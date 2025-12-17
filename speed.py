import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque

# =====================
# HELPER: DRAW READABLE LABEL
# =====================
def draw_label(
    image,
    text,
    x,
    y,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.9,
    thickness=2,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0)
):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        image,
        (x, y - th - baseline - 6),
        (x + tw + 6, y),
        bg_color,
        -1
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - 4),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

# =====================
# LOAD MODEL
# =====================
model = YOLO("yolov8n.pt")

# =====================
# VIDEO SETUP
# =====================
VIDEO_PATH = r"C:\Users\visha\vehicle_speed_estimation\vehicles.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video resolution: {w} x {h}, FPS: {fps}")

OUTPUT_FPS = 60

out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    OUTPUT_FPS,
    (w, h)
)


# =====================
# PERSPECTIVE TRANSFORM
# =====================
SOURCE = np.array([
    [1252, 787],     # top-left
    [2298, 803],     # top-right
    [5039, 2159],    # bottom-right
    [-550, 2159]     # bottom-left
], dtype=np.float32)

TARGET_WIDTH = 25     # meters
TARGET_HEIGHT = 250   # meters

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(SOURCE, TARGET)

# =====================
# TRACKER
# =====================
tracker = sv.ByteTrack(frame_rate=fps)

# Store Y positions (meters)
history = defaultdict(lambda: deque(maxlen=int(fps)))

# =====================
# VEHICLE CLASSES (COCO)
# =====================
VEHICLE_CLASSES = np.array([2, 3, 5, 7])  # car, motorcycle, bus, truck

# =====================
# DISPLAY SETTINGS
# =====================
DISPLAY_SCALE = 0.5  # adjust if needed for your screen
cv2.namedWindow("Vehicle Speed Estimation", cv2.WINDOW_NORMAL)

# =====================
# PROCESS VIDEO
# =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference (FULL RESOLUTION)
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter vehicle classes (NumPy-safe)
    class_mask = np.isin(detections.class_id, VEHICLE_CLASSES)
    detections = detections[class_mask]

    # Update tracker
    detections = tracker.update_with_detections(detections)

    for track_id, box in zip(detections.tracker_id, detections.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Bottom-center anchor point
        cx = int((x1 + x2) / 2)
        cy = int(y2)

        # Perspective transform → meters
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, M)[0][0]
        y_meters = warped[1]

        history[track_id].append(y_meters)

        speed_kmh = 0
        if len(history[track_id]) >= int(fps / 2):
            distance = history[track_id][-1] - history[track_id][0]
            time = len(history[track_id]) / fps
            speed_kmh = abs(distance / time) * 3.6

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Color-code label by speed
        if speed_kmh < 40:
            bg = (0, 128, 0)      # green
        elif speed_kmh < 80:
            bg = (0, 165, 255)    # orange
        else:
            bg = (0, 0, 255)      # red

        label = f"ID {track_id} | {int(speed_kmh)} km/h"
        draw_label(
            frame,
            label,
            x1,
            y1,
            font_scale=0.9,   # increase to 1.1 for very large videos
            thickness=2,     # use 3 for 4K
            bg_color=bg
        )

    # Draw calibration polygon (debug)
    cv2.polylines(
        frame,
        [SOURCE.astype(int)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2
    )

    # Save full-resolution frame
    out.write(frame)

    # Display resized frame (fixes zoom / push issue)
    display_frame = cv2.resize(
        frame,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )
    cv2.imshow("Vehicle Speed Estimation", display_frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
out.release()
cv2.destroyAllWindows()
