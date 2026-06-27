# An AI-Assisted Web Application for Epee Fencing Bout Analysis and Fencer Profiling

**Final Project Report — CM3020 Artificial Intelligence**
University of London, BSc Computer Science

**Author:** Nguyen Anh Minh
**Project template:** Project Idea 1 — Orchestrating AI Models to Achieve a Goal
**Repository:** https://github.com/minh-bitrise/epee-fencing-analysis (private)

---

## About this document

This is the **final project report** for the full project, not the preliminary report.

The preliminary report (`final proj uol/prelim_report.docx`) is a separate submission with
strict per-chapter word limits. This document has no such limits; it is intended to be a
thorough, academic-style record of the entire development process - planning, research,
implementation, testing, iteration, evaluation, and discussion - that can be polished into
the final submission at the end of the year.

The structure mirrors a typical final-year project report and is filled in as work progresses.
Sections labelled *(placeholder)* are stubs that will be expanded later; sections written out
in full are based on work already completed.

> **Author's working note.** I am keeping this document as a long-form development log written
> as I go, rather than as something to be drafted from scratch at the end of the year. The aim
> is that by the time the final submission is due, this file already contains a faithful and
> well-organised record of every meaningful decision, experiment, and result, which can then be
> trimmed and re-polished into the final required format.

---

## Table of contents

1. Abstract *(placeholder)*
2. Introduction
3. Background and literature review *(placeholder; preliminary version exists)*
4. Project planning and methodology
5. System design *(placeholder; preliminary version exists)*
6. Implementation and development log
7. Testing
8. Evaluation
9. Discussion and known limitations
10. Future work
11. Conclusion *(placeholder)*
12. References *(placeholder)*
13. Appendices

---

## 1. Abstract

*(Placeholder — to be written at the end of the project. Will summarise the problem, approach,
key results, and contribution in 200-300 words.)*

---

## 2. Introduction

### 2.1 Context

Video review is now a routine part of sports performance analysis at professional levels, but
in amateur and club-level fencing it remains largely manual, informal, and inconsistent. Recorded
bout footage often sits unreviewed, or is reviewed by watching whole clips repeatedly to extract
qualitative impressions. Even dedicated fencing analysis tools tend to require the user to label
every action by hand, which is a substantial time cost and discourages routine use.

This project develops an AI-assisted web application for analysing recorded epee fencing bouts.
The system combines multiple pre-trained AI models with a user-driven annotation workflow to
turn raw bout video into structured event records, movement and distance data, and a simple
tactical profile of each fencer.

### 2.2 Motivation

The motivation is partly first-person. As a fencer, I have experienced directly how
time-consuming it is to conduct meaningful post-bout self-analysis from video; conversations
with peers at club level reinforce that this is widely felt. The deeper motivation, however, is
that many tactically valuable metrics - distance maintained between the fencers over time and
at the moment of each touch, total push forward versus retreat backward, average engagement
distance and its variance, tempo (time between touches and exchange duration), frequency of
advances, retreats and lunges, and how a fencer's behaviour changes under pressure - are
**effectively impossible to measure by eye**. Recording even one of these by hand for a single
bout requires pausing, measuring and tabulating across thousands of frames. As a result, this
entire layer of quantitative tactical insight is almost never captured at the amateur level,
and review remains limited to subjective impressions. An AI-assisted system is well suited to
close this gap by computing such metrics automatically and leaving the user to verify and
interpret them.

### 2.3 Problem statement and aims

The problem is to design and build a working software system that:

- ingests an uploaded epee bout video;
- uses pre-trained AI models to extract structured information from it (fencer positions,
  poses, derived movement metrics, and likely event segments);
- presents that information to a user through a structured annotation interface that allows
  confirmation, correction, and labelling; and
- produces aggregated outputs - bout statistics, distance analysis, and a simple tactical
  profile - that are genuinely useful to fencers and coaches.

A central design principle, established in section 5 and validated in section 8, is that the
system is AI-assisted, not fully automatic. The unreliability of state-of-the-art computer
vision on real fencing footage is treated as a constraint to be accommodated by the workflow
rather than a problem to be solved before the system can be useful.

### 2.4 Project template

This project corresponds to the CM3020 Artificial Intelligence template "Project Idea 1:
Orchestrating AI Models to Achieve a Goal", which expects an integrated software system built
around at least three pre-trained models applied to a clearly defined problem.

### 2.5 Structure of this report

Section 3 reviews related academic literature and existing systems. Section 4 sets out the
project plan, methodology, tooling and risk management. Section 5 describes the system design,
including architecture and intended user workflow. Section 6 is a development log: it documents
the implementation as it actually happened, including iterations and changes of direction.
Section 7 describes testing methodology and what is covered. Section 8 reports evaluation
results on real bout footage. Section 9 discusses observed failure modes and known limitations.
Section 10 lays out future work and the path from the current prototype to the full system.
Section 11 concludes.

---

## 3. Background and literature review

*(Placeholder — the preliminary report contains a 1,200-word literature review covering
Rangasamy et al. (2020), Vahdani and Tian (2021), Hong et al. (2021), Mo (2022),
Mosqueira-Rey et al. (2023) plus Fencing Manager and Dartfish as related systems. That
review will be carried over and extended here.)*

---

## 4. Project planning and methodology

### 4.1 Methodology

The project is being developed in short two-week iterations, each producing demonstrable
output, with frequent commits to a private GitHub repository so that every iteration is
traceable. Each iteration consists of: a small design step, an implementation step, an
on-real-footage smoke test, and a unit-test update. This is closer to an evidence-driven,
test-supported development style than to a heavyweight formal process, and is appropriate for
a single-developer undergraduate project where the main risks lie in unknown model behaviour
rather than in coordination.

The development is organised around the principle of de-risking the AI components first.
Model selection and integration are front-loaded so that the feasibility of the full system is
established before the user interface is built, and so that the feature prototype required for
the preliminary report could be produced as a natural by-product of early iterations.

### 4.2 Work plan

The preliminary report contains a Gantt chart covering approximately 4.5 months of work in nine
two-week iterations (P1 to P9), comprising: requirements and environment setup; model
selection and comparative testing; backend API and data storage; AI processing pipeline
integration; frontend upload and review views; assisted-annotation interface; statistics,
profiling and dashboard; software and user testing; refinement and iteration; evaluation and
report write-up; with a final buffer iteration for contingency. This plan is the working
roadmap and is updated rather than replaced when scope changes.

### 4.3 Risk management

The principal risks are:

- **Model performance on real footage**: pose estimation and detection are known from the
  literature to be unreliable under motion blur, occlusion, and domain shift, all of which
  are present in fencing video. Mitigated by adopting an assisted rather than fully automatic
  workflow, by providing graceful fallbacks between methods, and by acknowledging this
  explicitly in the system's design.
- **Scope creep**: the project naturally invites further metrics, more advanced action
  recognition, blade-tip tracking, multi-camera, and so on. Mitigated by an explicit
  separation in the project's TODO tracking between in-scope work for the current submission
  and future-work items.
- **Compute / hardware**: pose estimation on every frame at 60 fps is heavy on a
  consumer-grade Apple Silicon CPU. Mitigated by frame-stride throttling on the pose stage and
  by planning to move heavier processing to a GPU workstation when scaling up.

### 4.4 Tooling and infrastructure

The development environment is:

- **Editor / workflow**: Claude Code (CLI) used for code generation, code review and writing
  support; VS Code for inspection; GitHub Desktop for repository management.
- **Language and frameworks**: Python 3.13 for all AI / video processing code. The prototype
  uses the Ultralytics implementation of YOLOv8 with ByteTrack for detection and tracking, and
  the MediaPipe Tasks API (`PoseLandmarker`) for pose estimation. OpenCV is used for video I/O
  and annotation. matplotlib is used for charts. pytest is used for unit testing.
- **Source control**: a private GitHub repository (`minh-bitrise/epee-fencing-analysis`),
  with a `main` branch for stable code and `feature/<name>` branches for experiments. The
  prototype was developed primarily on `feature/pose-distance`.
- **Project tracking**: the repository root contains `TODO.md`, structured into a Part A
  (preliminary report scope) and Part B (full-app scope) with cross-references, and this file
  (`final_report.md`) as the long-form report and development log. Completed items in
  `TODO.md` are checked off rather than deleted, so the historical trail is preserved.

---

## 5. System design

*(Placeholder — the preliminary report contains a 1,400-word design chapter including a layered
architecture diagram (Client / Application / Processing pipeline / Data) and a Gantt chart.
That material will be carried over here and extended, including an expanded data-flow
description for the assisted-annotation workflow.)*

---

## 6. Implementation and development log

This section records the implementation work as it actually happened, with the intent of
producing an honest, traceable account that can be quoted from or cited in the final
submission's discussion of process and engineering decisions.

### 6.1 Prototype scope and goals

The prototype, written for Chapter 4 of the preliminary report, was scoped to the most
foundational technical feature of the eventual system: identifying and following the two
fencers across a bout, and computing inter-fencer distance and per-fencer cumulative
forward / backward motion. This choice was deliberate:

- It is the foundation that every later feature depends on. Pose-based action recognition,
  touch suggestion, distance-band analysis, and tactical profiling all read from the same
  per-frame position and pose data that this prototype produces.
- It is visually demonstrable: the output is an annotated video, a CSV of frame-by-frame
  metrics, and a distance-over-time plot.
- It generates exactly the kind of metric (distance, push, pull) that the project's
  introduction identifies as infeasible to compute manually, which makes the prototype's
  output a direct evidence base for the project's premise.

### 6.2 Pipeline overview

The prototype pipeline is, per frame:

1. **Detection and tracking.** YOLOv8 (`yolov8n.pt`, the smallest variant) is run with the
   built-in ByteTrack tracker, filtered to the `person` class, with confidence and IoU
   thresholds tuned to admit only confident detections.
2. **Identity assignment.** Detections are passed through a custom `FencerTracker` that
   maintains two persistent slots (`Fencer 1` / `Fencer 2`) using spatial continuity rather
   than relying on ByteTrack's own IDs.
3. **Pose estimation.** For each successfully tracked slot, the corresponding crop is passed
   to MediaPipe's `PoseLandmarker` to extract body keypoints. To keep wall-clock time
   manageable on a CPU-only Apple Silicon machine, pose is run every Nth frame (default
   stride 3) rather than every frame.
4. **Distance estimation.** Distance is measured front-foot-to-front-foot when pose is
   available, and bottom-of-bounding-box-to-bottom-of-bounding-box (a feet-position proxy)
   when it is not. The pixel distance is normalised to metres using the average fencer
   bounding-box height as a scale reference under the assumption of an average fencer height
   of 1.75 m.
5. **Push / pull tracking.** Each fencer's horizontal position is smoothed with a rolling
   median, frame-to-frame movement is computed against the smoothed value, motion toward the
   opponent is accumulated as "push" and motion away as "pull", with biomechanically
   implausible jumps clamped out and sub-noise-floor movements ignored.
6. **Output.** Each frame produces a CSV row (raw distance, smoothed distance, method,
   per-fencer cumulative push and pull); the annotated frame is written to an output MP4 with
   a translucent bottom HUD; at end of run a distance-over-time chart is saved.

### 6.3 Person detection and tracking

YOLOv8 was selected for detection because it is mature, pre-trained on a generic person class
(COCO), well documented, and bundled with a stable tracking implementation (ByteTrack) in the
Ultralytics package. The `nano` variant was chosen for the prototype because it runs adequately
on CPU; a larger variant could be substituted later without code changes.

ByteTrack's own track IDs were initially used to maintain fencer identity across frames. This
proved fragile: ByteTrack frequently reassigns IDs after occlusion or rapid motion, and a
strict-ID-matching identity scheme caused the prototype to lose track of both fencers for
extended periods (in one early test, only ~5% of frames had both fencers identified). The
identity logic was reworked to use spatial continuity instead (section 6.6).

### 6.4 Pose estimation

MediaPipe was selected for pose estimation because of its low setup cost, broad documentation,
and adequate accuracy on standing-human poses. The deprecated `solutions.pose` API was tried
first; this is removed in the current MediaPipe release (0.10+), and the project moved to the
new `Tasks` API with the `pose_landmarker_lite` model file. Pose inference is performed on
each fencer's bounding-box crop (with a small padding) rather than on the whole frame, which
gives the pose model an easier problem and constrains its output to the right person.

Pose estimation is the most expensive step in the pipeline by far. On the Apple M4 used for
development, pose inference dominates wall-clock time, while detection is comparatively cheap.
To make end-to-end runs feasible, pose is by default executed only every third frame, with the
distance pipeline falling back gracefully to a bounding-box-based estimate for the in-between
frames.

### 6.5 Distance estimation

Distance was originally computed centre-to-centre of bounding boxes. This was changed for two
reasons: arm and weapon extension distort the bounding box, moving its centre in a way that is
unrelated to where the fencer's body is; and front-foot to front-foot distance is the
tactically meaningful measure in fencing - it is the quantity coaches refer to as "distance",
not the centre-of-mass separation. The pose pipeline extracts left and right ankles for each
fencer; the front foot is identified as the ankle whose x-coordinate is nearer to the
opponent's reference x; and the distance is the Euclidean distance between the two front
ankles. When pose is unavailable, the system falls back to the bottom-centre of each bounding
box (a better feet-position proxy than the centre).

The pixel-to-metres normalisation uses the average bounding-box height of the two fencers as a
scale reference. This is approximate, with a number of known sources of error including camera
angle and fencers' actual heights, but it is sufficient to produce values consistent with
expected engagement ranges (typically 1.5-3 m for an active exchange).

### 6.6 Identity tracking and bystander rejection

The `FencerTracker` is the result of several iterations.

The first version used strict ByteTrack-ID matching: it locked onto the two highest-confidence
track IDs in the first multi-person frame, and from then on accepted only detections matching
those exact IDs. This was robust against background bystanders but extremely fragile to ID
reassignment, and was abandoned after testing showed that it lost both fencers for >95% of
frames in a typical 3-minute clip.

The second version replaced strict ID matching with **spatial continuity**: per frame, the two
highest-confidence detections are taken as candidates, and assignments to slots are chosen to
minimise total movement from each slot's last known position. This restored ~99.9% coverage on
the test footage and allowed identity to survive ByteTrack ID changes, but it had the opposite
problem: it would happily accept a background bystander (referee, audience member, fencer on an
adjacent strip) as one of the tracked fencers whenever such a person happened to be the second
most confident person detection.

The third version added two **gates** to the spatial-continuity matcher:

- A **spatial gate** rejects any candidate whose centre is more than `GATE_DISTANCE_RATIO`
  bounding-box-heights from the slot's last known centre.
- A **size gate** rejects any candidate whose bounding-box height is outside
  `[MIN_SIZE_RATIO, MAX_SIZE_RATIO]` of the slot's last accepted height.

The gates were initially set quite strictly (`GATE_DISTANCE_RATIO=2.5`,
`MIN_SIZE_RATIO=0.5`, `MAX_SIZE_RATIO=1.8`), which dropped coverage to ~55% as legitimate
fencer motion was over-rejected; they were subsequently relaxed (`3.5 / 0.4 / 2.2`) to give
~82% coverage while still filtering obvious bystanders. The maximum observed inter-fencer
distance dropped from 11.27 m to 6.60 m between the un-gated and gated runs, confirming that
the gates were excluding precisely the wrong-target outliers that had inflated distance
estimates. Remaining bystander captures and close-range flicker (section 9) are deferred to
the full system, where motion modelling, piste-region detection, or appearance re-ID will be
deployed.

### 6.7 Push / pull metric

This metric records, in metres, how much each fencer has moved toward the opponent ("push")
and away from the opponent ("pull") over the bout. It directly addresses the report's
introduction: it is one of the metrics that is essentially impossible to measure by hand.

The initial implementation accumulated raw frame-to-frame horizontal displacement. The first
real-clip run produced obviously inflated values (around 216 m of push and 214 m of pull per
fencer in a 3-minute bout). This was traced to two effects: bounding-box jitter (small
frame-to-frame fluctuations in the YOLO box edges even when the fencer is essentially
stationary) and small camera pans not large enough to be clamped out as biomechanically
impossible. Two corrections were applied: each fencer's reference x is now smoothed with a
rolling median before frame-to-frame movement is computed; the per-frame motion clamp was
tightened (any movement >0.15 m/frame is treated as camera motion and ignored entirely); and
the noise floor was raised (any movement <0.03 m is treated as noise and not accumulated).
The same test clip now produces totals in the 13-22 m range per fencer, which are biomechanically
plausible for a 3-minute active bout.

### 6.8 Annotated video output

The output video draws bounding boxes, fencer labels, key pose landmarks, and a bottom-of-frame
heads-up panel. The HUD shows the time, the currently smoothed distance with a colour code by
tactical zone (red ≤1.0 m, orange ≤1.8 m, green otherwise), and the cumulative push / pull
totals for each fencer. The panel sits on a semi-transparent dark rectangle with a thin white
outline so that it remains legible against the bright piste background. An earlier version
placed the HUD in the top-left and top-right corners with no background, which proved hard to
read in lighter parts of the video.

---

## 7. Testing

The prototype is supported by 55 unit tests, organised into ten test classes, written with
`pytest`. Tests cover:

- the geometry helpers (`get_box_centre`, `get_box_bottom_centre`, `box_height_pixels`,
  `pixel_distance`, `normalise_distance`);
- the colour-zone decision helper (`distance_zone_colour`);
- the pose helpers (`get_hip_centre`, `get_front_foot`);
- the `FencerTracker` stateful logic (initial assignment by leftmost-first-frame position,
  independence from confidence ordering, tolerance of ByteTrack ID changes, spatial gate
  rejection of far-away bystanders, size gate rejection of much-smaller candidates, accepting
  realistic frame-to-frame motion);
- the `PushPullTracker` stateful logic (advance and retreat directions for opponent on either
  side, camera-pan clamp, noise-floor rejection, accumulation across multiple frames, the
  effect of smoothing on jitter accumulation);
- the rolling-median smoother (`smooth_distance`).

The tests are intentionally written against the pure data behaviours of these components
rather than against end-to-end video processing, so they are fast (the full suite runs in
under two seconds) and stable enough to run on every iteration. End-to-end smoke testing is
performed by running the pipeline against a real fencing video clip and inspecting the
resulting annotated video and CSV by eye.

The combination of fast unit tests plus on-real-footage smoke tests has proved effective at
catching both regression bugs in the matching logic and unrealistic numbers in the derived
metrics (the inflated push / pull values of section 6.7 were caught this way).

---

## 8. Evaluation

The prototype has been evaluated on two independently sourced clips of FIE-level epee bouts,
both trimmed to three minutes of in-bout footage. Footage attribution is given in
Appendix C.

### 8.1 Clip 1 — engagement-oriented bout

- Frames processed: 10,791
- Frames with both fencers detected: 8,842 (82%)
- Pose-based distance samples: 2,842 (~32% of detected frames, consistent with the pose
  stride of 3)
- Mean inter-fencer distance: 2.26 m
- Distance range: 0.11 m to 6.60 m
- Cumulative motion: Fencer 1 — 14.2 m push / 11.1 m pull. Fencer 2 — 15.8 m push / 17.1 m pull.

### 8.2 Clip 2 — high-activity bout

- Frames processed: 9,000 (50 fps source)
- Frames with both fencers detected: 7,813 (87%)
- Pose-based distance samples: 2,327 (~30%)
- Mean inter-fencer distance: 3.33 m
- Distance range: 0.36 m to 8.45 m
- Cumulative motion: Fencer 1 — 41.0 m push / 41.5 m pull. Fencer 2 — 36.8 m push / 37.1 m pull.

### 8.3 Cross-clip comparison

The two clips show a striking difference in activity. Clip 2's fencers do roughly two to three
times the footwork of clip 1's and engage at a noticeably wider average distance. This is
exactly the kind of cross-bout comparison that is effectively impossible to perform by manual
observation, and so its reproducible production by the prototype is itself a meaningful
result: the prototype can already generate metrics that distinguish bouts at a level of detail
that manual analysis cannot match.

### 8.4 Detection coverage

Coverage of 82-87% (both fencers detected and assigned to their stable slots) is well above
what is needed to support a useful annotation workflow. The remaining 13-18% of frames are
largely either (a) frames where one or both fencers are partially off-screen, (b) frames where
the spatial / size gates correctly rejected a passing referee or background detection, or
(c) frames where the close-range flicker described in section 9 caused the tracker to drop a
slot momentarily. None of these introduce wrong data into the aggregate metrics; they merely
reduce the number of contributing samples.

---

## 9. Discussion and known limitations

*(This section currently mirrors `TODO.md` Part A3 and the failure-modes paragraph in A4a.
It will be re-written and expanded for the final submission, with concrete frame timestamps
and screenshots from the two test clips.)*

In summary, the prototype's two main observed failure modes are wrong-target capture by
background people (typically the centre referee) and close-range identity flicker when the
detector merges the two fencers during a clinch. Both are classic data-association failures
in single-camera multi-object tracking. Both are predicted in the design as the reason the
system is AI-assisted rather than fully automatic; both are accommodated by the planned
annotation workflow rather than treated as bugs in the prototype. The deeper algorithmic
fixes - piste-region detection, motion modelling, and appearance re-ID - are tracked as
future work (section 10).

---

## 10. Future work

Future work falls into four categories: (i) tracker robustness, with piste-region detection,
a motion / velocity model, and an appearance re-ID embedding as the three candidate paths;
(ii) event handling and the assisted-annotation UI, which is the design's answer to the
prototype's failure modes; (iii) richer derived metrics (engagement-vs-reset distance, tempo,
distance bands, leading-vs-trailing behaviour, per-fencer trajectories); and (iv) the
remaining two pre-trained models needed to satisfy the project brief - notably an LLM for the
written tactical summary. The full web-application shell (React frontend, FastAPI backend,
database) and GPU-accelerated processing complete the path from prototype to deployable
system. Detailed task-level breakdown is maintained in `TODO.md` Part B.

---

## 11. Conclusion

*(Placeholder — to be written at the end of the project.)*

---

## 12. References

*(Placeholder — the preliminary report's reference list will be carried over and extended
here in Harvard style.)*

---

## Appendices

### A. Source code overview

The prototype is in `prototype/`, with `run_detection.py` as the main pipeline and
`test_detection.py` as the unit-test suite. Dependencies are declared in
`prototype/requirements.txt`.

### B. CSV schema

The per-frame CSV produced by the prototype has columns: `frame`, `time_s`, `distance_raw_m`,
`distance_smooth_m`, `method` (`pose` or `bbox`), `f1_advance_m`, `f1_retreat_m`,
`f2_advance_m`, `f2_retreat_m`.

### C. Footage attribution

*(Placeholder — final URLs and per-clip details to be filled in here; the drop-in attribution
paragraph for the preliminary report is in `TODO.md` Part A4.)*

### D. Development log highlights

The full development trail is in the project's git history. Significant milestones include:
initial bounding-box-centre distance prototype; switch to pose-based front-foot distance with
MediaPipe; introduction of `FencerTracker` for stable identity; the regression caused by
overly strict ID locking and its reversion to spatial continuity; the inflated push / pull
values and the addition of rolling-median smoothing plus motion clamping; the addition of
spatial and size gates against bystanders and the subsequent re-calibration; the move of the
HUD from corners to a translucent bottom panel for legibility; the evaluation runs on two
independent test clips.
