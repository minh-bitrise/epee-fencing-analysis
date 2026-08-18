#!/bin/bash
# Resolution ablation for touch detection.
#
# Downsamples one clip to a range of resolutions and runs the full pipeline on
# each, so that touch-detection quality can be compared against the SAME ground
# truth with resolution as the only variable.
#
# This exists because the original observation was confounded. Clip 4 scored
# F1 0.60 against 0.80-0.86 for the 720p clips, and resolution was the obvious
# explanation, but clip 4 also differs in venue, framing, referee presence and
# bout dynamics. Downsampling an existing clip changes one thing and nothing
# else, and reuses labels already produced.
#
# Clip 3 is the subject: 14 labelled touches, the most of any clip, so it gives
# the most statistical power.
set -u
SRC="fencing_clip3.mp4"
OUT="results_ablation"
mkdir -p "$OUT"

for H in 540 360 270 180; do
  SCALED="ablation_${H}p.mp4"
  if [ ! -f "$SCALED" ]; then
    echo "=== downscaling to ${H}p ==="
    ffmpeg -y -i "$SRC" -vf "scale=-2:${H}" -c:v libx264 -preset fast -crf 23 \
           -c:a copy "$SCALED" 2>&1 | tail -1
  fi
  echo "=== processing ${H}p ==="
  python3 run_detection.py --video "$SCALED" --output "$OUT" 2>&1 | tail -12
  echo "=== detecting touches at ${H}p ==="
  python3 detect_touches.py --csv "$OUT/ablation_${H}p_distance.csv" 2>&1 | tail -3
done
echo "=== ablation complete ==="
