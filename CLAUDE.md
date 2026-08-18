# Project: AI-Assisted Epee Fencing Bout Analysis

## What this project is
A web application for post-bout analysis of epee fencing videos. Users upload a bout video; the system uses pre-trained AI models to detect/track fencers, estimate poses, extract distance and movement metrics, propose probable touch events, and generate a written tactical summary. Users confirm or correct AI suggestions through an annotation interface. Outputs: bout statistics, distance analysis, tactical profile, LLM-generated summary.

**University module:** CM3020 Artificial Intelligence - Project Idea 1 (Orchestrating AI models to achieve a goal)
**Student:** Nguyen Anh Minh (minh-bitrise on GitHub)

## Stack
- **Frontend:** React (single-page app)
- **Backend:** Python, FastAPI
- **AI models:** YOLOv8 + ByteTrack (detection/tracking), MediaPipe Pose (pose estimation), LLM via API (summary generation)
- **Data:** distance/movement derived geometrically from tracked positions

## Repo structure
```
code/prototype/  - Python/AI code for the prototype (will sit alongside backend/ and frontend/ in code/ later)
uni_modules/     - read-only university materials (syllabus, instructions, transcripts)
proposal/        - project proposal submission deliverables
preliminary/     - preliminary report submission deliverables
final/           - final report submission deliverables (in progress)
```

## Prototype scope (Chapter 4 of prelim report)
Person detection + tracking + inter-fencer distance measurement.
- Input: a fencing bout video clip
- Output: annotated video (bounding boxes + fencer IDs + distance overlay) + distance-over-time data

## Key design decisions
- AI-assisted not fully automatic - user confirms/corrects all AI suggestions
- Scoped to epee only, uploaded video (not real-time), single camera
- Pose estimation treated as informative but potentially noisy - not used for hard decisions
- Event detection (touching/scoring) is a stretch goal, not core

## Branching convention
- `main` - stable, working code only
- `feature/<name>` - new feature or experiment
- `fix/<name>` - bug fixes

## Notes for Claude
- User is an undergrad, not a professional dev - keep code simple and well-commented
- Prototype must be demonstrable in a 3-5 min video
- Report word limits: Intro 1000w, Lit Review 2500w, Design 2000w, Prototype 1500w
- No long dashes (em dashes) in report - use short hyphens only
- Preliminary report file: preliminary/prelim_report.docx
- Final report file: final/final_report.md
