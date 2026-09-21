"""
Build the two clip-4 variants used to separate two explanations for its coverage.

Kept so a negative result stays reproducible, the same reason the abandoned audio
detector survives behind --use-audio. Both variants made clip 4 worse, which is
the finding.

    python3 make_clip4_variants.py

Writes fencing_clip4_masked.mp4 and fencing_clip4_crop.mp4 beside the source clip.
"""
import sys
import numpy as np
import cv2
from ultralytics import YOLO

video = "fencing_clip4.mp4"
model = YOLO("yolov8n.pt")

# --- derive the boundary from the two clusters, not from a frame -----------
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
boxes = []
for i in range(0, total, 30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    if not ok:
        continue
    for b in model.predict(f, conf=0.3, iou=0.5, classes=[0],
                           verbose=False)[0].boxes.xyxy.cpu().numpy():
        boxes.append(b)
cap.release()
b = np.array(boxes)
h = b[:, 3] - b[:, 1]

# The height histogram is bimodal: small distant boxes and fencer-sized ones. The
# split is taken at the trough between them rather than at a round number.
small = b[h < 40]          # distant gallery figures
fencer = b[(h >= 80) & (h <= 125)]   # the fencer-sized cluster
spectator_bottom = np.percentile(small[:, 3], 99)
fencer_top = np.percentile(fencer[:, 1], 1)
print(f"spectator cluster n={len(small)}, bottom edge p99 = {spectator_bottom:.1f}")
print(f"fencer cluster    n={len(fencer)}, top edge p1     = {fencer_top:.1f}")

if fencer_top <= spectator_bottom:
    print(f"WARNING: the clusters overlap by {spectator_bottom - fencer_top:.1f} px, "
          f"so no horizontal line separates them cleanly.")
boundary = int(np.floor(min(fencer_top, spectator_bottom + 1)))
# never cut into a fencer: back off to the highest fencer top actually observed
boundary = int(min(boundary, np.floor(fencer[:, 1].min())))
print(f"boundary chosen: y = {boundary} "
      f"(keeps every observed fencer box intact)")
removed = (b[:, 3] < boundary).sum()
print(f"detections entirely above it: {removed} of {len(b)} "
      f"({100*removed/len(b):.0f}%)")

# --- write the variants ---------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
mw = cv2.VideoWriter("fencing_clip4_masked.mp4", fourcc, fps, (W, H))
cw = cv2.VideoWriter("fencing_clip4_crop.mp4", fourcc, fps, (W, H - boundary))
n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    m = f.copy()
    m[:boundary, :] = 0
    mw.write(m)
    cw.write(f[boundary:, :])
    n += 1
cap.release(); mw.release(); cw.release()
print(f"wrote {n} frames to fencing_clip4_masked.mp4 ({W}x{H}) "
      f"and fencing_clip4_crop.mp4 ({W}x{H-boundary})")
