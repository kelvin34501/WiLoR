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
from dataclasses import dataclass, field

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


@dataclass
class TrackObservation:
    bbox: List[float]
    conf: float
    timestamp: Optional[float] = None


@dataclass
class TrackState:
    max_history: int
    history: deque[TrackObservation] = field(default_factory=deque)
    lost_frames: int = 0

    def latest_bbox(self) -> Optional[List[float]]:
        if not self.history:
            return None
        return self.history[-1].bbox

    def has_history(self) -> bool:
        return bool(self.history)

    def add_observation(self, bbox: List[float], conf: float, timestamp: Optional[float] = None):
        if len(self.history) >= self.max_history:
            self.history.popleft()
        self.history.append(TrackObservation(bbox=list(bbox), conf=conf, timestamp=timestamp))
        self.lost_frames = 0

    def reset_with(self, bbox: List[float], conf: float, timestamp: Optional[float] = None):
        self.history.clear()
        self.add_observation(bbox=bbox, conf=conf, timestamp=timestamp)

    def mark_lost(self, max_lost_frames: int):
        self.lost_frames += 1
        if self.lost_frames > max_lost_frames:
            self.history.clear()


def build_track_state_info(num_cameras: int, history_size: int) -> Dict[str, List[TrackState]]:
    return {
        "lh": [TrackState(max_history=history_size) for _ in range(num_cameras)],
        "rh": [TrackState(max_history=history_size) for _ in range(num_cameras)],
    }


def calculate_center(bbox):
    """Calculate bounding box center point."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_center_distance(bbox1, bbox2):
    """Calculate Euclidean distance between two bbox centers."""
    c1 = calculate_center(bbox1)
    c2 = calculate_center(bbox2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes."""
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2

    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def group_candidates_by_hand(
    bboxes: List[List[float]],
    hand_types: List[int],
    scores: List[float],
) -> Dict[str, List[Tuple[List[float], float]]]:
    grouped_candidates = {"lh": [], "rh": []}
    for bbox, hand_type, score in zip(bboxes, hand_types, scores):
        hand_key = "lh" if hand_type == 0 else "rh"
        grouped_candidates[hand_key].append((bbox, score))
    return grouped_candidates


def best_history_iou(candidate_bbox: List[float], track_state: TrackState) -> float:
    if not track_state.history:
        return 0.0

    best_iou = 0.0
    for age, observation in enumerate(reversed(track_state.history)):
        recency_weight = 0.85**age
        weighted_iou = calculate_iou(candidate_bbox, observation.bbox) * recency_weight
        if weighted_iou > best_iou:
            best_iou = weighted_iou
    return best_iou


def select_best_candidate(
    candidates: List[Tuple[List[float], float]],
    track_state: TrackState,
    init_conf_threshold: float = 0.25,
    track_conf_threshold: float = 0.20,
    max_track_distance: float = 200.0,
    min_track_iou: float = 0.05,
    track_iou_weight: float = 0.7,
    track_conf_weight: float = 0.3,
) -> Tuple[Optional[Tuple[List[float], float]], str]:
    """Select a candidate using confidence-only init/reinit and IoU-based tracking."""
    if not candidates:
        return None, "missing"

    init_candidates = [candidate for candidate in candidates if candidate[1] >= init_conf_threshold]
    track_candidates = [candidate for candidate in candidates if candidate[1] >= track_conf_threshold]
    highest_init_conf_candidate = max(init_candidates, key=lambda item: item[1]) if init_candidates else None

    if not track_state.has_history():
        if highest_init_conf_candidate is None:
            return None, "missing"
        return highest_init_conf_candidate, "init"

    latest_bbox = track_state.latest_bbox()
    if latest_bbox is None:
        if highest_init_conf_candidate is None:
            return None, "missing"
        return highest_init_conf_candidate, "init"

    tracked_candidates = []
    for bbox, conf in track_candidates:
        center_distance = calculate_center_distance(bbox, latest_bbox)
        if center_distance > max_track_distance:
            continue

        history_iou = best_history_iou(bbox, track_state)
        if history_iou < min_track_iou:
            continue

        combined_score = track_iou_weight * history_iou + track_conf_weight * conf
        tracked_candidates.append((bbox, conf, combined_score))

    if tracked_candidates:
        best_bbox, best_conf, _ = max(tracked_candidates, key=lambda item: (item[2], item[1]))
        return (best_bbox, best_conf), "tracked"

    if highest_init_conf_candidate is None:
        return None, "missing"
    return highest_init_conf_candidate, "reinit"


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
    init_conf_threshold: float = 0.25,
    track_conf_threshold: float = 0.20,
    track_state_info: Optional[Dict[str, List[TrackState]]] = None,
    timestamp: Optional[float] = None,
    max_track_distance: float = 200.0,
    min_track_iou: float = 0.05,
    max_lost_frames: int = 3,
    track_iou_weight: float = 0.7,
    track_conf_weight: float = 0.3,
):
    """
    Run hand detection on a batch of images with optional tracking.
    
    Args:
        detector: YOLO detector instance
        rgb_images: List of RGB images (one per camera)
        init_conf_threshold: Confidence threshold for init/reinit candidate selection
        track_conf_threshold: Confidence threshold for tracked candidate selection
        track_state_info: Per-camera tracking state for lh/rh (optional)
        timestamp: Frame timestamp used when appending history
        max_track_distance: Max pixel distance for tracking association
        min_track_iou: Minimum weighted IoU required to continue tracking
        max_lost_frames: Number of missing frames to retain history before clearing
        track_iou_weight: Weight for IoU term during tracked selection
        track_conf_weight: Weight for confidence term during tracked selection
    
    Returns:
        bbox_info: Dict with 'lh' and 'rh' keys, each containing list of bboxes (one per camera)
    """
    detector_conf_threshold = min(init_conf_threshold, track_conf_threshold)

    # Run batch detection
    batch_results = detector(rgb_images, conf=detector_conf_threshold, verbose=False)

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

        grouped_candidates = group_candidates_by_hand(bboxes, hand_types, scores)

        left_track_state = track_state_info["lh"][cam_idx] if track_state_info is not None else None
        right_track_state = track_state_info["rh"][cam_idx] if track_state_info is not None else None

        best_left = None
        if left_track_state is not None:
            best_left, left_mode = select_best_candidate(
                grouped_candidates["lh"],
                left_track_state,
                init_conf_threshold=init_conf_threshold,
                track_conf_threshold=track_conf_threshold,
                max_track_distance=max_track_distance,
                min_track_iou=min_track_iou,
                track_iou_weight=track_iou_weight,
                track_conf_weight=track_conf_weight,
            )
            if best_left is not None:
                left_bbox, left_conf = best_left
                if left_mode == "tracked":
                    left_track_state.add_observation(left_bbox, left_conf, timestamp=timestamp)
                else:
                    left_track_state.reset_with(left_bbox, left_conf, timestamp=timestamp)
            else:
                left_track_state.mark_lost(max_lost_frames=max_lost_frames)
        elif grouped_candidates["lh"]:
            init_candidates = [
                candidate for candidate in grouped_candidates["lh"] if candidate[1] >= init_conf_threshold
            ]
            if init_candidates:
                best_left = max(init_candidates, key=lambda item: item[1])

        best_right = None
        if right_track_state is not None:
            best_right, right_mode = select_best_candidate(
                grouped_candidates["rh"],
                right_track_state,
                init_conf_threshold=init_conf_threshold,
                track_conf_threshold=track_conf_threshold,
                max_track_distance=max_track_distance,
                min_track_iou=min_track_iou,
                track_iou_weight=track_iou_weight,
                track_conf_weight=track_conf_weight,
            )
            if best_right is not None:
                right_bbox, right_conf = best_right
                if right_mode == "tracked":
                    right_track_state.add_observation(right_bbox, right_conf, timestamp=timestamp)
                else:
                    right_track_state.reset_with(right_bbox, right_conf, timestamp=timestamp)
            else:
                right_track_state.mark_lost(max_lost_frames=max_lost_frames)
        elif grouped_candidates["rh"]:
            init_candidates = [
                candidate for candidate in grouped_candidates["rh"] if candidate[1] >= init_conf_threshold
            ]
            if init_candidates:
                best_right = max(init_candidates, key=lambda item: item[1])

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
    init_conf_threshold: float = 0.25,
    track_conf_threshold: float = 0.20,
    identity: str = "wilor-0",
    bbox_timeout_ms: float = 200.0,
    max_track_distance: float = 200.0,
    track_history_size: int = 5,
    max_lost_frames: int = 3,
    min_track_iou: float = 0.05,
    track_iou_weight: float = 0.7,
    track_conf_weight: float = 0.3,
):
    """
    Main WiLoR detection server with external worker protocol.
    
    Args:
        video_shape: (width, height) tuple
        sync_channel: ZMQ channel to subscribe synchronized camera data
        pub_channel: ZMQ channel to publish bbox results
        cmd_channel: ZMQ channel for handshake with master (DEALER socket)
        detector_model_path: Path to YOLO detector model
        init_conf_threshold: Confidence threshold for init/reinit selection
        track_conf_threshold: Confidence threshold for tracked selection
        identity: ZMQ identity for DEALER socket
        bbox_timeout_ms: Timeout in ms to clear cached bbox
        max_track_distance: Max pixel distance for tracking association
        track_history_size: Number of per-camera history observations to retain
        max_lost_frames: Number of missing frames before track history is cleared
        min_track_iou: Minimum weighted IoU to keep tracking instead of reinitializing
        track_iou_weight: IoU weight for tracked candidate scoring
        track_conf_weight: Confidence weight for tracked candidate scoring
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
    detector_conf_threshold = min(init_conf_threshold, track_conf_threshold)

    # Warmup detector
    _logger.info("Warming up detector...")
    dummy = np.zeros((video_shape[1], video_shape[0], 3), dtype=np.uint8)
    detector([dummy], conf=detector_conf_threshold, verbose=False)
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
    track_state_info = build_track_state_info(num_cameras, history_size=track_history_size)

    _logger.info(
        "Starting detection loop... (bbox_timeout: %.1fms, max_track_distance: %.1fpx, "
        "history_size: %d, max_lost_frames: %d, min_track_iou: %.3f, init_conf: %.2f, track_conf: %.2f)",
        bbox_timeout_ms,
        max_track_distance,
        track_history_size,
        max_lost_frames,
        min_track_iou,
        init_conf_threshold,
        track_conf_threshold,
    )

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
                init_conf_threshold,
                track_conf_threshold,
                track_state_info=track_state_info,
                timestamp=timestamp,
                max_track_distance=max_track_distance,
                min_track_iou=min_track_iou,
                max_lost_frames=max_lost_frames,
                track_iou_weight=track_iou_weight,
                track_conf_weight=track_conf_weight,
            )
            det_time = (time.time() - det_start) * 1000  # ms
            detection_times.append(det_time)

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
    parser.add_argument("--init_conf_threshold",
                        type=float,
                        default=0.275,
                        help="Confidence threshold for init/reinit selection (default: 0.275)")
    parser.add_argument("--track_conf_threshold",
                        type=float,
                        default=0.178,
                        help="Confidence threshold for tracked selection (default: 0.178)")
    parser.add_argument("--bbox_timeout_ms",
                        type=float,
                        default=200.0,
                        help="Timeout in ms to clear cached bbox when hand leaves frame (default: 200)")
    parser.add_argument("--identity", type=str, default="wilor-0", help="ZMQ identity for DEALER socket")
    parser.add_argument("--max_track_distance",
                        type=float,
                        default=200.0,
                        help="Max pixel distance for tracking association (default: 200)")
    parser.add_argument("--track_history_size",
                        type=int,
                        default=5,
                        help="Number of history frames retained per camera/hand track (default: 5)")
    parser.add_argument("--max_lost_frames",
                        type=int,
                        default=3,
                        help="Number of consecutive misses before clearing track history (default: 3)")
    parser.add_argument("--min_track_iou",
                        type=float,
                        default=0.05,
                        help="Minimum weighted IoU required to continue a track (default: 0.05)")
    parser.add_argument("--track_iou_weight",
                        type=float,
                        default=0.7,
                        help="Weight of IoU when scoring tracked candidates (default: 0.7)")
    parser.add_argument("--track_conf_weight",
                        type=float,
                        default=0.3,
                        help="Weight of confidence when scoring tracked candidates (default: 0.3)")

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
        init_conf_threshold=args.init_conf_threshold,
        track_conf_threshold=args.track_conf_threshold,
        identity=args.identity,
        bbox_timeout_ms=args.bbox_timeout_ms,
        max_track_distance=args.max_track_distance,
        track_history_size=args.track_history_size,
        max_lost_frames=args.max_lost_frames,
        min_track_iou=args.min_track_iou,
        track_iou_weight=args.track_iou_weight,
        track_conf_weight=args.track_conf_weight,
    )
