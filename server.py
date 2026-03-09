"""
WiLoR Hand Detection Server (External Worker Protocol)

Implements the external worker handshake protocol for integration with
the teleop framework. Detects hands using YOLO and publishes bounding boxes
for downstream 3D reconstruction (POEM).

Protocol:
    1. Connect DEALER socket to master's ROUTER
    2. Send "starting" status
    3. Load YOLO detector
    4. Send "model_loaded" status, wait for init config
    5. Receive camera_info from master
    6. Send "ready" status
    7. Main loop: subscribe sync images, detect hands, publish bboxes

Author: Yiqian Gao, Xinyu Zhan
"""
from __future__ import annotations

import os
import json
import time
import atexit
import signal
import threading
import argparse
import logging
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

import zmq
import numpy as np
import msgpack
import msgpack_numpy

# Import YOLO detector
from ultralytics import YOLO

# Import utility functions
from server_tool import log_util
# Parse ZMQ endpoints
from server_tool import zmq_channel_util

_logger = logging.getLogger("wilor")


def calculate_area(bbox):
    """Calculate bounding box area."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def calculate_center(bbox):
    """Calculate bounding box center point."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_center_distance(bbox1, bbox2):
    """Calculate Euclidean distance between two bbox centers."""
    c1 = calculate_center(bbox1)
    c2 = calculate_center(bbox2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def get_max_area_hands(bboxes, hand_types, scores):
    """
    Get the largest left hand and right hand by area.
    
    Returns:
        (max_left, max_right): Each is None or (bbox, hand_type, score, area)
    """
    max_left = None
    max_right = None

    for bbox, hand_type, score in zip(bboxes, hand_types, scores):
        area = calculate_area(bbox)

        if hand_type == 0:  # left_hand
            if max_left is None or area > max_left[3]:
                max_left = (bbox, hand_type, score, area)
        else:  # right_hand
            if max_right is None or area > max_right[3]:
                max_right = (bbox, hand_type, score, area)

    return max_left, max_right


def select_best_hands(
    bboxes,
    hand_types,
    scores,
    prev_left_bbox=None,
    prev_right_bbox=None,
    distance_weight: float = 0.6,
    max_track_distance: float = 200.0,
):
    """
    Select best left and right hand considering both area and distance to previous detection.
    
    Uses a weighted scoring: score = (1 - distance_weight) * norm_area + distance_weight * (1 - norm_distance)
    Falls back to area-only selection when no previous bbox or distance exceeds threshold.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        hand_types: List of hand types (0=left, 1=right)
        scores: List of detection confidence scores
        prev_left_bbox: Previous frame's left hand bbox (or None)
        prev_right_bbox: Previous frame's right hand bbox (or None)
        distance_weight: Weight for distance term (0-1, higher = prefer closer to previous)
        max_track_distance: Max pixel distance for tracking; beyond this, use area-only
    
    Returns:
        (best_left, best_right): Each is None or (bbox, hand_type, score, area)
    """
    # Separate detections by hand type
    left_candidates = []
    right_candidates = []

    for bbox, hand_type, score in zip(bboxes, hand_types, scores):
        area = calculate_area(bbox)
        if hand_type == 0:  # left_hand
            left_candidates.append((bbox, hand_type, score, area))
        else:  # right_hand
            right_candidates.append((bbox, hand_type, score, area))

    def select_best(candidates, prev_bbox):
        """Select best candidate using combined area + distance scoring."""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # If no previous bbox, fall back to max area
        if prev_bbox is None:
            return max(candidates, key=lambda x: x[3])  # x[3] = area

        # Calculate areas and distances for normalization
        areas = [c[3] for c in candidates]
        distances = [calculate_center_distance(c[0], prev_bbox) for c in candidates]

        max_area = max(areas) if max(areas) > 0 else 1.0
        min_area = min(areas)

        best_candidate = None
        best_score = -float('inf')

        for candidate, area, dist in zip(candidates, areas, distances):
            # If distance exceeds threshold, this candidate gets area-only scoring
            if dist > max_track_distance:
                # Normalize area to [0, 1]
                norm_area = (area - min_area) / (max_area - min_area) if max_area > min_area else 1.0
                combined_score = norm_area
            else:
                # Normalize area to [0, 1]
                norm_area = (area - min_area) / (max_area - min_area) if max_area > min_area else 1.0
                # Normalize distance to [0, 1] where 0 = at prev position, 1 = at max_track_distance
                norm_distance = dist / max_track_distance
                # Combined score: higher is better
                combined_score = (1 - distance_weight) * norm_area + distance_weight * (1 - norm_distance)

            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate

        return best_candidate

    best_left = select_best(left_candidates, prev_left_bbox)
    best_right = select_best(right_candidates, prev_right_bbox)

    return best_left, best_right


def decode_sync_message(msg: bytes, video_shape: Tuple[int, int]):
    """
    Decode synchronized message from SyncUnit.
    
    Args:
        msg: Raw message bytes
        video_shape: (width, height) tuple
    
    Returns:
        tuple: (rgb_images, timestamp) or (None, None) on failure
    """
    try:
        data = msgpack.unpackb(msg, object_hook=msgpack_numpy.decode, raw=False)
        synced_data = data.get("synced_data")
        synced_ts = data.get("synced_ts")

        if synced_data is None or synced_ts is None:
            return None, None

        # Extract RGB images from synced_data
        # Skip non-camera payloads (e.g., mocap) by checking for "color" key
        rgb_images = []
        for payload in synced_data:
            if isinstance(payload, dict) and "color" in payload:
                color_img = payload["color"].reshape(video_shape[1], video_shape[0], 3)
                rgb_images.append(color_img)

        return rgb_images, synced_ts

    except Exception as e:
        _logger.warning(f"Failed to decode sync message: {e}")
        return None, None


def detect_hands_batch(
    detector,
    rgb_images: List[np.ndarray],
    conf_threshold: float = 0.3,
    prev_bbox_info: Optional[Dict[str, List]] = None,
    distance_weight: float = 0.6,
    max_track_distance: float = 200.0,
):
    """
    Run hand detection on a batch of images with optional tracking.
    
    Args:
        detector: YOLO detector instance
        rgb_images: List of RGB images (one per camera)
        conf_threshold: Detection confidence threshold
        prev_bbox_info: Previous frame's bbox_info for tracking (optional)
        distance_weight: Weight for distance in tracking score (0-1)
        max_track_distance: Max pixel distance for tracking association
    
    Returns:
        bbox_info: Dict with 'lh' and 'rh' keys, each containing list of bboxes (one per camera)
    """
    # Run batch detection
    batch_results = detector(rgb_images, conf=conf_threshold, verbose=False)

    # Initialize bbox_info structure
    bbox_info = {
        "lh": [],  # List of left hand bboxes, one per camera
        "rh": []  # List of right hand bboxes, one per camera
    }

    # Process results for each camera
    for cam_idx, results in enumerate(batch_results):
        bboxes = []
        hand_types = []
        scores = []

        if getattr(results, "boxes", None) is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                bboxes.append([x1, y1, x2, y2])
                hand_types.append(cls)  # 0: left_hand, 1: right_hand
                scores.append(conf)

        # Get previous bboxes for this camera (if available)
        prev_left_bbox = None
        prev_right_bbox = None
        if prev_bbox_info is not None:
            prev_left_bbox = prev_bbox_info["lh"][cam_idx]
            prev_right_bbox = prev_bbox_info["rh"][cam_idx]

        # Select best hands using tracking-aware selection
        best_left, best_right = select_best_hands(
            bboxes,
            hand_types,
            scores,
            prev_left_bbox=prev_left_bbox,
            prev_right_bbox=prev_right_bbox,
            distance_weight=distance_weight,
            max_track_distance=max_track_distance,
        )

        # Append bbox to respective lists
        if best_left is not None:
            bbox_info["lh"].append(best_left[0])  # Just the bbox coordinates
        else:
            bbox_info["lh"].append(None)

        if best_right is not None:
            bbox_info["rh"].append(best_right[0])
        else:
            bbox_info["rh"].append(None)

    return bbox_info


def main(
    video_shape: Tuple[int, int],
    sync_channel: str,
    pub_channel: str,
    cmd_channel: str,
    detector_model_path: str = "./pretrained_models/detector.pt",
    conf_threshold: float = 0.3,
    identity: str = "wilor-0",
    bbox_timeout_ms: float = 200.0,
    distance_weight: float = 0.6,
    max_track_distance: float = 200.0,
):
    """
    Main WiLoR detection server with external worker protocol.
    
    Args:
        video_shape: (width, height) tuple
        sync_channel: ZMQ channel to subscribe synchronized camera data
        pub_channel: ZMQ channel to publish bbox results
        cmd_channel: ZMQ channel for handshake with master (DEALER socket)
        detector_model_path: Path to YOLO detector model
        conf_threshold: Detection confidence threshold
        identity: ZMQ identity for DEALER socket
        bbox_timeout_ms: Timeout in ms to clear cached bbox
        distance_weight: Weight for distance in tracking (0=area only, 1=distance only)
        max_track_distance: Max pixel distance for tracking association
    """
    _logger.info("WiLoR Hand Detection Server starting...")

    def parse_channel(channel: str) -> str:
        endpoint = zmq_channel_util.channel_name_to_endpoint(channel, "/dev/shm/hcc_demo")
        if zmq_channel_util.is_ipc_endpoint(endpoint):
            os.makedirs(os.path.dirname(zmq_channel_util.ipc_to_filepath(endpoint)), exist_ok=True)
        return endpoint

    ctx = zmq.Context()

    # ========== Phase 1: Handshake with master ==========

    # Command socket (DEALER) for async communication with master node
    cmd_socket = ctx.socket(zmq.DEALER)
    cmd_socket.setsockopt(zmq.IDENTITY, identity.encode())
    cmd_endpoint = parse_channel(cmd_channel)
    cmd_socket.connect(cmd_endpoint)
    _logger.info(f"Command socket (DEALER) connected to: {cmd_endpoint} with identity: {identity}")

    # Report starting status
    cmd_socket.send_string(json.dumps({
        "status": "starting",
        "msg": "WiLoR server starting, loading detector...",
    }))
    _logger.info("Reported 'starting' status to master node")

    # Load YOLO detector
    _logger.info(f"Loading detector from {detector_model_path}")
    detector = YOLO(detector_model_path)

    # Warmup detector
    _logger.info("Warming up detector...")
    dummy = np.zeros((video_shape[1], video_shape[0], 3), dtype=np.uint8)
    detector([dummy], conf=conf_threshold, verbose=False)
    _logger.info("Detector loaded and warmed up")

    # Report model_loaded status and wait for init command
    cmd_socket.send_string(json.dumps({
        "status": "model_loaded",
        "msg": "Detector loaded, waiting for config...",
    }))
    _logger.info("Reported 'model_loaded' status, waiting for init command...")

    # Wait for init command with camera_info from master node
    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)
    camera_info = None

    while camera_info is None:
        try:
            socks = dict(poller.poll(timeout=100))  # 100ms poll
            if cmd_socket in socks:
                cmd_msg = cmd_socket.recv_string()
                cmd_data = json.loads(cmd_msg)
                if cmd_data.get("cmd") == "init":
                    camera_info = cmd_data.get("camera_info", {})
                    _logger.info(f"Received camera_info: {camera_info}")
                elif cmd_data.get("cmd") == "ping":
                    cmd_socket.send_string(json.dumps({
                        "status": "pong",
                        "msg": "waiting for init",
                    }))
                else:
                    _logger.warning(f"Unknown command while waiting for init: {cmd_data.get('cmd')}")
        except Exception as e:
            _logger.error(f"Error receiving init command: {e}")

    camera_name_list = list(camera_info.values())
    num_cameras = len(camera_name_list)
    _logger.info(f"Configured for {num_cameras} cameras: {camera_name_list}")

    # ========== Phase 2: Setup data channels ==========

    # Subscribe to synchronized images
    sync_socket = ctx.socket(zmq.SUB)
    sync_socket.setsockopt(zmq.SUBSCRIBE, b"")
    sync_socket.setsockopt(zmq.RCVHWM, 1)
    sync_socket.setsockopt(zmq.CONFLATE, 1)
    sync_endpoint = parse_channel(sync_channel)
    sync_socket.connect(sync_endpoint)
    _logger.info(f"Subscribed to sync channel: {sync_endpoint}")

    # Publish bbox results
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.SNDHWM, 1)
    pub_socket.setsockopt(zmq.CONFLATE, 1)
    pub_endpoint = parse_channel(pub_channel)
    pub_socket.bind(pub_endpoint)
    _logger.info(f"Publishing bbox results to: {pub_endpoint}")

    # Report ready status
    cmd_socket.send_string(json.dumps({
        "status": "ready",
        "msg": "WiLoR server ready",
    }))
    _logger.info("Reported 'ready' status to master node")

    # ========== Phase 3: Main detection loop ==========

    # Graceful shutdown flag — set by SIGTERM/SIGHUP handler so the main loop
    # can exit cleanly and run cleanup.  Without this, Python's default
    # behaviour terminates immediately without running atexit handlers.
    _shutdown_event = threading.Event()

    def _sigterm_handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        _logger.info(f"Received {sig_name}, requesting graceful shutdown...")
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGHUP, _sigterm_handler)

    _cleanup_done = False

    def cleanup():
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        _logger.info("Cleaning up...")
        cmd_socket.close()
        sync_socket.close()
        pub_socket.close()
        ctx.term()
        _logger.info("WiLoR server stopped")

    atexit.register(cleanup)

    # Initialize empty bbox template
    empty_bbox_info = {
        "lh": [None for _ in range(num_cameras)],
        "rh": [None for _ in range(num_cameras)],
    }

    # Stats tracking
    frame_count = 0
    detection_times = deque(maxlen=100)
    last_valid_bbox_info = None
    last_valid_time = None  # Timestamp of last valid bbox detection
    prev_bbox_info = None  # Previous frame's bbox for tracking

    _logger.info(f"Starting detection loop... (bbox_timeout: {bbox_timeout_ms}ms, "
                 f"distance_weight: {distance_weight}, max_track_distance: {max_track_distance}px)")

    while True:
        try:
            # --- Check shutdown conditions ------------------------------------
            # 1) SIGTERM / SIGHUP via signal handler
            if _shutdown_event.is_set():
                _logger.info("Shutdown flag set (signal), stopping main loop...")
                break

            # 2) ZMQ shutdown command from master node
            try:
                cmd_msg = cmd_socket.recv_string(zmq.NOBLOCK)
                cmd_data = json.loads(cmd_msg)
                if cmd_data.get("cmd") == "shutdown":
                    _logger.info("Received shutdown command from master, stopping...")
                    break
                else:
                    _logger.debug(f"Ignoring cmd during main loop: {cmd_data}")
            except zmq.Again:
                pass

            # Receive synchronized images (non-blocking)
            try:
                msg = sync_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue

            rgb_images, timestamp = decode_sync_message(
                msg,
                video_shape=video_shape,
            )

            if rgb_images is None:
                continue

            frame_count += 1

            # Run detection with tracking
            det_start = time.time()
            bbox_info = detect_hands_batch(
                detector,
                rgb_images,
                conf_threshold,
                prev_bbox_info=prev_bbox_info,
                distance_weight=distance_weight,
                max_track_distance=max_track_distance,
            )
            det_time = (time.time() - det_start) * 1000  # ms
            detection_times.append(det_time)

            # Update tracking state for next frame
            prev_bbox_info = bbox_info

            # Check detection validity
            valid_lh = sum(b is not None for b in bbox_info["lh"])
            valid_rh = sum(b is not None for b in bbox_info["rh"])
            has_usable_bbox = (valid_lh >= 2) or (valid_rh >= 2)

            # Determine what to publish
            current_time = time.time()
            if has_usable_bbox:
                bbox_to_publish = bbox_info
                last_valid_bbox_info = bbox_info
                last_valid_time = current_time
            elif last_valid_bbox_info is not None and last_valid_time is not None:
                # Check if last valid bbox has expired
                elapsed_ms = (current_time - last_valid_time) * 1000
                if elapsed_ms <= bbox_timeout_ms:
                    # Reuse last valid bbox to maintain continuity
                    bbox_to_publish = last_valid_bbox_info
                else:
                    # Timeout expired, clear cached bbox
                    bbox_to_publish = empty_bbox_info
                    last_valid_bbox_info = None
                    last_valid_time = None
            else:
                bbox_to_publish = empty_bbox_info

            # Publish bbox results
            pub_msg = msgpack.packb({
                "sync_timestamp": timestamp,
                "bbox": bbox_to_publish,
            },
                                    default=msgpack_numpy.encode)
            pub_socket.send(pub_msg)

            # Log periodically
            if frame_count % 30 == 0:
                avg_det_time = np.mean(detection_times) if detection_times else 0
                _logger.info(f"Frame {frame_count} | Det: {det_time:.1f}ms (avg: {avg_det_time:.1f}ms) | "
                             f"LH views: {valid_lh} | RH views: {valid_rh}")

        except KeyboardInterrupt:
            _logger.info("Received keyboard interrupt, stopping...")
            break
        except Exception as e:
            _logger.error(f"Error in detection loop: {e}", exc_info=True)
            time.sleep(0.01)

    cleanup()


if __name__ == "__main__":
    log_util.log_init()
    log_util.enable_console()

    parser = argparse.ArgumentParser(description="WiLoR Hand Detection Server (External Worker)")
    parser.add_argument("--server.video_shape",
                        type=str,
                        default="1280x720",
                        help="Video resolution (default: 1280x720)")
    parser.add_argument("--server.sync_channel",
                        type=str,
                        required=True,
                        help="ZMQ channel to subscribe synchronized camera data")
    parser.add_argument("--server.pub_channel", type=str, required=True, help="ZMQ channel to publish bbox results")
    parser.add_argument("--server.cmd_channel",
                        type=str,
                        required=True,
                        help="ZMQ channel for handshake with master (DEALER socket)")
    parser.add_argument("--detector_model_path",
                        type=str,
                        default="./pretrained_models/detector.pt",
                        help="Path to YOLO detector model")
    parser.add_argument("--conf_threshold",
                        type=float,
                        default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--bbox_timeout_ms",
                        type=float,
                        default=200.0,
                        help="Timeout in ms to clear cached bbox when hand leaves frame (default: 200)")
    parser.add_argument("--identity", type=str, default="wilor-0", help="ZMQ identity for DEALER socket")
    parser.add_argument("--distance_weight",
                        type=float,
                        default=0.6,
                        help="Weight for distance in tracking score (0=area only, 1=distance only, default: 0.6)")
    parser.add_argument("--max_track_distance",
                        type=float,
                        default=200.0,
                        help="Max pixel distance for tracking association (default: 200)")

    args = parser.parse_args()

    # Parse video_shape
    video_shape_str = getattr(args, "server.video_shape")
    width, height = map(int, video_shape_str.split("x"))
    video_shape = (width, height)

    main(
        video_shape=video_shape,
        sync_channel=getattr(args, "server.sync_channel"),
        pub_channel=getattr(args, "server.pub_channel"),
        cmd_channel=getattr(args, "server.cmd_channel"),
        detector_model_path=args.detector_model_path,
        conf_threshold=args.conf_threshold,
        identity=args.identity,
        bbox_timeout_ms=args.bbox_timeout_ms,
        distance_weight=args.distance_weight,
        max_track_distance=args.max_track_distance,
    )
