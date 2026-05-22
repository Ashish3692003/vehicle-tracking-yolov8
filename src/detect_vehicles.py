"""
Real-Time Vehicle Tracking & Counting using YOLOv8 + ByteTrack
================================================================

A deep learning-based traffic monitoring system that processes video streams
to detect, track, and count vehicles in real-time.

Author : V. Ashish Nadh
Course : MLSAVA - ML for Speech & Video Analysis
Institution: GITAM University Visakhapatnam

Usage:
    python src/detect_vehicles.py --input data/car-detection.mp4 --output results/output_counted.mp4
"""

import cv2
from ultralytics import YOLO
import argparse
import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# 🚗 COCO Class Filter: Only track relevant vehicle classes
# ─────────────────────────────────────────────────────────────────────────
# COCO IDs:
#   0: person   1: bicycle  2: car   3: motorcycle  4: airplane  5: bus
#   6: train    7: truck    8: boat  ... (rest excluded)

VEHICLE_IDS = {2, 5, 7}  # Car=2, Bus=5, Truck=7 | (Motorcycles excluded per project spec)

CLASS_NAMES = {
    2: "Car",
    5: "Bus",
    7: "Truck"
}

# ─────────────────────────────────────────────────────────────────────────
# 🎨 Color Mapping per Vehicle Class
# ─────────────────────────────────────────────────────────────────────────
COLORS = {
    2: (0, 255, 0),    # Green  - Car
    5: (0, 255, 255),  # Cyan   - Bus
    7: (0, 0, 255),    # Red    - Truck
}

# ─────────────────────────────────────────────────────────────────────────
# 📐 Crossing Line Configuration
# ────────────────────
# A virtual horizontal line drawn across the video frame.
# Vehicles crossing this line (from top to bottom) are counted once.
# ─────────────────────────────────────────────────────────────────────────
CROSSING_LINE_Y = None  # Set dynamically based on frame height


def load_model():
    """Load pretrained YOLOv8n model from Ultralytics hub."""
    model = YOLO("yolov8n.pt")
    return model


def draw_crossing_line(frame):
    """Draw a horizontal crossing line across the frame."""
    global CROSSING_LINE_Y
    h = frame.shape[0]
    CROSSING_LINE_Y = int(h * 0.6)  # Line at 60% from top of frame
    color = (0, 255, 255)  # Yellow
    thickness = 2
    cv2.line(frame, (0, CROSSING_LINE_Y), (frame.shape[1], CROSSING_LINE_Y), color, thickness)
    cv2.putText(frame, f"Crossing Line (y={CROSSING_LINE_Y})", (10, CROSSING_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def count_vehicle(frame, counted_ids, track_id, center_y):
    """
    Check if vehicle has crossed the line. If yes, add its track_id to counted set.
    Returns True if counted, False otherwise.
    """
    if track_id in counted_ids:
        return False  # Already counted this vehicle

    if CROSSING_LINE_Y is not None and center_y > CROSSING_LINE_Y:
        counted_ids.add(track_id)
        return True
    return False


def process_detection(frame, results, counted_ids, vehicle_counts):
    """
    Process YOLOv8 detection results:
    - Annotate frame with bounding boxes, class names, and track IDs
    - Count vehicles crossing the line
    """
    for box in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        # Filter: Only process vehicle classes (cars, buses, trucks)
        if class_id not in VEHICLE_IDS:
            continue

        # Extract track ID if tracking is used
        track_id = None
        if hasattr(box, "id") and box.id is not None:
            track_id = int(box.id[0])

        # Compute center of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Get class name and color
        label = CLASS_NAMES.get(class_id, f"ID:{class_id}")
        color = COLORS.get(class_id, (255, 0, 0))
        thickness = 2

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label text with background
        text = f"{label} {confidence:.2f}"
        if track_id is not None:
            text += f"  Track:{track_id}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Draw center point
        cv2.circle(frame, (center_x, center_y), 5, color, -1)

        # Count vehicle if crossing the line
        if track_counting_enabled and track_id is not None:
            if count_vehicle(frame, counted_ids, track_id, center_y):
                class_name = CLASS_NAMES.get(class_id, "Unknown")
                vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

    return frame


def display_counts(frame, vehicle_counts):
    """
    Display running vehicle count on the top-right of the frame.
    """
    y_offset = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8

    # Background panel
    h = len(vehicle_counts) * 40 + 60
    cv2.rectangle(frame, (450, 20), (frame.shape[1] - 10, 20 + h), (0, 0, 0), -1)
    cv2.putText(frame, "--- Vehicle Count ---", (470, 45), font, font_scale, (255, 255, 255), 2)

    for idx, (cls_name, count) in enumerate(vehicle_counts.items()):
        color = COLORS.get({2: "Car", 5: "Bus", 7: "Truck"}.get(cls_name), (255, 255, 255))
        y_pos = 90 + idx * 40
        cv2.putText(frame, f"{cls_name}: {count}", (470, y_pos),
                    font, 0.7, color, 2)

    return frame


def main():
    global track_counting_enabled
    parser = argparse.ArgumentParser(
        description="Real-Time Vehicle Tracking & Counting using YOLOv8 + ByteTrack"
    )
    parser.add_argument("--input", type=str, default="data/car-detection.mp4",
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default="results/output_counted.mp4",
                        help="Path to output video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Path to YOLOv8 model (default: yolov8n.pt)")
    parser.add_argument("--track", action="store_true", default=True,
                        help="Enable multi-object tracking with ByteTrack")
    args = parser.parse_args()

    # Toggle tracking mode
    track_counting_enabled = args.track

    # ──────────── 1. Load Model ─────────────
    print("[INFO] Loading YOLOv8n model... (COCO pretrained weights)")
    model = YOLO(args.model)
    print("[INFO] Model loaded successfully!")
    print("[INFO] Vehicle Classes: Car(2), Bus(5), Truck(7)")
    print("[INFO] Motorcycles and other COCO classes excluded.")

    # ──────────── 2. Open Video ─────────────
    print(f"[INFO] Opening video: {args.input}")
    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("ERROR: Could not open video file.")
        print("       Ensure your input video is placed in the 'data/' folder.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video properties: {width}x{height} @ {fps} FPS | {total_frames} frames")

    # ──────────── 3. Setup Output Writer ─────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"[INFO] Output video will be saved to: {args.output}")

    # ──────────── 4. Tracking State ─────────────
    counted_ids = set()      # Set of vehicle track_ids already counted
    vehicle_counts = {}      # Running count per class

    frame_num = 0

    # ──────────── 5. Process Each Frame ─────────────
    print("\n[INFO] Starting vehicle detection & tracking...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % int(fps) == 0:  # Print progress every second
            elapsed = frame_num / fps
            print(f"[PROGRESS] Frame {frame_num}/{total_frames} | {elapsed:.1f}s elapsed")

        # ── Run YOLOv8 Inference ──
        if track_counting_enabled:
            # model.track() enables ByteTrack SORT/ByteTrack persistent tracking
            results = model.track(frame, persist=True, classes=list(VEHICLE_IDS))
        else:
            results = model.predict(frame, classes=list(VEHICLE_IDS))

        # ── Draw Crossing Line (for counting reference) ──
        frame = draw_crossing_line(frame)

        # ── Process & Annotate Detections ──
        frame = process_detection(frame, results, counted_ids, vehicle_counts)

        # ── Display Running Counts ──
        frame = display_counts(frame, vehicle_counts)

        # ── Write to output video ──
        out.write(frame)

        # Optional: press 'q' to quit (for local display)
        # cv2.imshow("Vehicle Tracking & Counting", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # ──────────── 6. Cleanup ─────────────
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("[SUCCESS] Processing Complete!")
    print("=" * 50)
    print(f"Total vehicles counted:")
    for cls, count in sorted(vehicle_counts.items()):
        print(f"  {cls}: {count}")
    print(f"Output video saved: {args.output}")


if __name__ == "__main__":
    main()
