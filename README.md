# AI-Assisted Epee Fencing Bout Analysis

BSc Final Year Project - CM3020 Artificial Intelligence, University of London

## Overview
A web application that helps fencers and coaches analyse epee bout videos. Users upload a recording; the system uses pre-trained AI models to extract movement data, flag probable touch events, and generate a tactical profile. Users verify and annotate the AI suggestions rather than labelling everything from scratch.

## Features
- Fencer detection and tracking (YOLOv8 + ByteTrack)
- Pose estimation (MediaPipe Pose)
- Inter-fencer distance analysis over time
- Assisted event annotation (confirm / correct AI suggestions)
- Bout statistics and tactical profile
- LLM-generated written summary

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
Then open http://localhost:8000.

The summary button needs an Anthropic API key. The server looks for it once at
startup, first in `ANTHROPIC_API_KEY` and then in the macOS Keychain under the
service `anthropic-api-key`, so on a machine where the key is already in the
Keychain there is nothing to do. Without a key everything else works and the
button explains itself.

Without the Node build step the server still works: it falls back to a no-build-step
interface that reviews already-processed bouts, also available at `/legacy`. That page
needs nothing but Python.

For front-end development, `npm run dev` in `code/frontend` serves on port 5173 and
proxies the API to the backend, so the two restart independently and a page reload never
interrupts a running job.

### Running the pipeline directly
Every stage is a command-line program, and the web layer invokes exactly these commands
rather than reimplementing them:
```bash
cd code/prototype
python3 derive_piste.py    --video <bout.mp4> --output piste.json   # optional
python3 run_detection.py   --video <bout.mp4> --output results/ [--piste-config piste.json]
python3 detect_touches.py  --csv results/<bout>_distance.csv
python3 generate_summary.py --csv results/<bout>_distance.csv --touches <touches.csv>
```

## Project structure
```
code/prototype/   The AI pipeline: detection, tracking, pose, distance, touches, summary
code/backend/     FastAPI application: upload, job runner, annotation API
code/frontend/    React client (Vite)
code/var/         Runtime data: uploads, job records, their outputs (gitignored)
uni_modules/      Read-only university materials
proposal/         Project proposal submission deliverables
preliminary/      Preliminary report submission deliverables
final/            Final report submission (in progress)
```
