# AI-Assisted Epee Fencing Bout Analysis

BSc Final Year Project — CM3020 Artificial Intelligence, University of London

## Overview
A web application that helps fencers and coaches analyse epee bout videos. Users upload a recording; the system uses pre-trained AI models to extract movement data, flag probable touch events, and generate a tactical profile. Users verify and annotate the AI suggestions rather than labelling everything from scratch.

## Features
- Fencer detection and tracking (YOLOv8 + ByteTrack)
- Pose estimation (MediaPipe Pose)
- Inter-fencer distance analysis over time
- Assisted event annotation (confirm / correct AI suggestions)
- Bout statistics and tactical profile
- LLM-generated written summary

## Prototype
The current prototype demonstrates person detection, tracking, and distance measurement on a fencing video clip.

### Requirements
- Python 3.10+
- See `code/prototype/requirements.txt`

### Run
```bash
cd code/prototype
pip install -r requirements.txt
python run_detection.py --video <path_to_video>
```

## Project structure
```
code/             ALL project code (prototype + future backend/frontend)
uni_modules/      Read-only university materials
proposal/         Project proposal submission deliverables
preliminary/      Preliminary report submission deliverables
final/            Final report submission (in progress)
```
