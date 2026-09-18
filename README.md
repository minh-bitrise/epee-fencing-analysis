# AI-Assisted Epee Fencing Bout Analysis

BSc Final Year Project - CM3020 Artificial Intelligence, University of London

## Overview
A web application that helps fencers and coaches analyse epee bout videos. Users upload a recording; the system uses pre-trained AI models to extract movement data, flag probable touch events, and generate a tactical profile. Users verify and annotate the AI suggestions rather than labelling everything from scratch.

## Features
- Fencer detection and tracking (YOLOv8 + ByteTrack)
- Pose estimation (MediaPipe Pose)
- Inter-fencer distance analysis over time
- Assisted event annotation (confirm / correct AI suggestions)
- Touch attribution from the scoring lamps, and lunge proposals calibrated per bout
- Bout statistics and tactical profile
- LLM-generated written summary, with an automated check that every figure in it came from the data
- A review queue and a manual-logging mode, which are the two conditions of the effort measurement

## Running the application
The whole workflow runs in a browser: upload a bout, watch it process, review the result.

### Requirements
- Python 3.10+, and `pip install -r code/prototype/requirements.txt`
- Node 18+ (only to build the interface; see the fallback below if you would rather not)
- `ffmpeg` on the PATH, for converting the annotated render into something browsers play

### Run
```bash
cd code/frontend && npm install && npm run build
cd ../backend && python3 -m uvicorn app:app --port 8000
```
Then open **http://localhost:8000/app/**. The `/app/` matters: the bare root serves an
older no-build-step page kept as a fallback, which has only the four original annotation
actions and none of the later features.

The summary button needs an API key for the language-model provider. The server
looks for it once at startup, first in the `ANTHROPIC_API_KEY` environment
variable and then in the macOS Keychain under the service `anthropic-api-key`
(both names are fixed by the provider's SDK), so on a machine where the key is
already in the
Keychain there is nothing to do. Without a key everything else works and the
button explains itself.

Without the Node build step the server still works: it falls back to a no-build-step
interface that reviews already-processed bouts, also available at `/legacy`. That page
needs nothing but Python.

For front-end development, `npm run dev` in `code/frontend` serves on port 5173 and
proxies the API to the backend, so the two restart independently and a page reload never
interrupts a running job.

### Tests

```bash
cd code/prototype && python3 -m pytest -q                  # pipeline: 440
cd code/backend   && python3 -m pytest -q -m "not slow"   # API: 189
cd code/backend   && python3 -m pytest -q test_end_to_end.py  # slow, loads models: 2
cd code/frontend  && npm test                             # interface: 146
```

The end-to-end tests are marked slow and excluded from the fast run. They drive
the real pipeline on a small synthetic video to check that the stages hand their
outputs to each other, which is the part unit tests structurally cannot reach:
the stages are joined by filename conventions rather than return values.

### Running the pipeline directly
Every stage is a command-line program, and the web layer invokes exactly these commands
rather than reimplementing them:
```bash
cd code/prototype
python3 derive_piste.py    --video <bout.mp4> --output piste.json   # optional
python3 run_detection.py   --video <bout.mp4> --output results/ [--piste-config piste.json]
python3 detect_touches.py  --csv results/<bout>_distance.csv
python3 detect_scorer.py   --video <bout.mp4> --touches <touches.csv>
python3 detect_lunges.py   --csv results/<bout>_distance.csv --annotations <annotations.json>
python3 fencer_profile.py  --metrics results/<bout>_distance.csv --touches <touches.csv>
python3 generate_summary.py --csv results/<bout>_distance.csv --touches <touches.csv>
```

### Reproducing the evaluation
These are the commands behind the figures in Chapter 5 of the report. Each scores
something against hand-labelled ground truth in `code/prototype/ground_truth/`, and
each prints what it cannot establish as well as what it can.
```bash
cd code/prototype
python3 evaluate_touches.py --truth ground_truth/fencing_clip3_touches.csv \
                            --candidates results_current/fencing_clip3_distance_touches.csv
python3 evaluate_lunges.py --csv results_pose/fencing_clip3_distance.csv \
                           --bout results_pose:fencing_clip3   # calibration vs transfer
python3 evaluate_closing_share.py --power        # what the metric can and cannot resolve
python3 evaluate_pose.py --results results_posecmp   # pose against the bounding box
python3 audit_summary.py --summary results_current/fencing_clip3_distance_summary.md
python3 touch_features.py --results results_current  # labelled candidates, and the ceiling
python3 train_touch.py --results results_current --curve --ablation   # learned proposer vs rule
```
`train_touch.py` needs `scikit-learn`, which nothing in the running system imports:
the shipped detector is the hand-set rule, because the comparison found the rule
better. `evaluate_pose.py` needs a results directory carrying the `distance_bbox_m`
column, which means a run of `run_detection.py` from this version onwards.

## Project structure
```
code/prototype/   The AI pipeline: detection, tracking, pose, distance, touches, summary
code/backend/     FastAPI application: upload, job runner, annotation API
code/frontend/    React client (Vite), with its own test suite
code/var/         Runtime data: uploads, job records, their outputs (gitignored)
uni_modules/      Read-only university materials
proposal/         Project proposal submission deliverables
preliminary/      Preliminary report submission deliverables
final/            Final report submission (in progress), its figures, and the checking tools
```
