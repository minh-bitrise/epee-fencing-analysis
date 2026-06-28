# An AI-Assisted Web Application for Epee Fencing Bout Analysis and Fencer Profiling

## About this document

This is the **final project report** for the full project, not the preliminary report.

The preliminary report (`preliminary/prelim_report.docx`) is a separate, earlier submission
with strict per-chapter word limits. This document has no such limits; it is a thorough,
academic-style record of the entire project that will be polished into the final submission.

The structure follows the **six chapters required by the module** for the final report:
Introduction, Literature Review, Design, Implementation, Evaluation, and Conclusion, preceded
by an abstract and followed by references and appendices. Chapters are numbered; subsections
are titled rather than numbered, so the document can be reordered and extended without
renumbering churn. Sections marked *(placeholder)* are stubs to be expanded; the rest is based
on work already completed.

> **Author's working note.** I keep this as a long-form development log written as I go, rather
> than something to be drafted from scratch at the end. The aim is that by submission time this
> file already holds a faithful, well-organised record of every meaningful decision, experiment
> and result, which can then be trimmed into the final required format. Reminders on house style
> for the final write-up (past tense, minimal first person, avoid passive voice, justify every
> claim with evidence) are tracked in `TODO.md`.

## Table of contents

Front matter: Abstract

1. Introduction
2. Literature Review
3. Design  (domain & users, architecture, technologies, work plan, evaluation plan, ethics, methodology, tooling)
4. Implementation  (development log of the prototype, plus testing)
5. Evaluation  (results on real footage, summative assessment against the aims, limitations)
6. Conclusion  (summary, future work, self-evaluation)

Back matter: References, Appendices

## Abstract

*(Placeholder — to be written at the end of the project. Will summarise the problem, approach,
key results, and contribution in 200-300 words.)*

---


## 1. Introduction

### Context

Video review is now a routine part of sports performance analysis at professional levels, but
in amateur and club-level fencing it remains largely manual, informal, and inconsistent. Recorded
bout footage often sits unreviewed, or is reviewed by watching whole clips repeatedly to extract
qualitative impressions. Even dedicated fencing analysis tools tend to require the user to label
every action by hand, which is a substantial time cost and discourages routine use.

This project develops an AI-assisted web application for analysing recorded epee fencing bouts.
The system combines multiple pre-trained AI models with a user-driven annotation workflow to
turn raw bout video into structured event records, movement and distance data, and a simple
tactical profile of each fencer.

### Motivation

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

### Problem statement and aims

The problem is to design and build a working software system that:

- ingests an uploaded epee bout video;
- uses pre-trained AI models to extract structured information from it (fencer positions,
  poses, derived movement metrics, and likely event segments);
- presents that information to a user through a structured annotation interface that allows
  confirmation, correction, and labelling; and
- produces aggregated outputs - bout statistics, distance analysis, and a simple tactical
  profile - that are genuinely useful to fencers and coaches.

A central design principle, established in the Design chapter and validated in the Evaluation chapter, is that the
system is AI-assisted, not fully automatic. The unreliability of state-of-the-art computer
vision on real fencing footage is treated as a constraint to be accommodated by the workflow
rather than a problem to be solved before the system can be useful.

### Project template

This project corresponds to the CM3020 Artificial Intelligence template "Project Idea 1:
Orchestrating AI Models to Achieve a Goal", which expects an integrated software system built
around at least three pre-trained models applied to a clearly defined problem.

### Structure of this report

The Literature Review chapter reviews related academic literature and existing systems. The
Design chapter sets out the system design (architecture, technologies, intended user workflow),
the work plan and evaluation plan, ethics, and the development methodology and tooling. The
Implementation chapter is a development log: it documents the prototype as it was actually
built, including iterations and changes of direction, and the testing that supports it. The
Evaluation chapter reports results on real bout footage, assesses how well the work meets its
aims, and discusses observed failure modes and known limitations. The Conclusion summarises the
project, lays out future work, and reflects on what was learned.

---


## 2. Literature Review

*(Placeholder — the preliminary report contains a 1,200-word literature review covering
Rangasamy et al. (2020), Vahdani and Tian (2021), Hong et al. (2021), Mo (2022),
Mosqueira-Rey et al. (2023) plus Fencing Manager and Dartfish as related systems. That
review will be carried over and extended here.)*

---


## 3. Design

*(Placeholder — the preliminary report contains a 1,400-word design chapter including a layered
architecture diagram (Client / Application / Processing pipeline / Data) and a Gantt chart.
That material will be carried over here and extended, including an expanded data-flow
description for the assisted-annotation workflow.)*

---



### Methodology

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

### Work plan

The preliminary report contains a Gantt chart covering approximately 4.5 months of work in nine
two-week iterations (P1 to P9), comprising: requirements and environment setup; model
selection and comparative testing; backend API and data storage; AI processing pipeline
integration; frontend upload and review views; assisted-annotation interface; statistics,
profiling and dashboard; software and user testing; refinement and iteration; evaluation and
report write-up; with a final buffer iteration for contingency. This plan is the working
roadmap and is updated rather than replaced when scope changes.

### Risk management

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

### Tooling and infrastructure

This section records the development environment and supporting infrastructure in detail. The
intent is that the engineering process behind the project be visibly professional and
reproducible, not just the final artefacts. Each tool and convention is listed with a brief
rationale so that another developer (or marker) could continue the project from the same
starting point.

**Editor and workflow.** Claude Code (CLI) was used for code generation, code review and
writing support; VS Code for visual inspection; GitHub Desktop for branch and commit
visualisation when terminal git was inconvenient. The CLI / IDE split was practical rather
than ideological: writing and refactoring code happened in the CLI session, reading and
spot-checking the diffs in VS Code, and inspecting commit history in GitHub Desktop.

**Language, runtime and dependencies.** All AI and video-processing code is in Python 3.13.
The prototype uses the Ultralytics implementation of YOLOv8 with the built-in ByteTrack
tracker for detection and identification; the MediaPipe Tasks API (`PoseLandmarker`,
`pose_landmarker_lite` model) for pose estimation; OpenCV for video I/O, frame manipulation
and on-frame annotation; matplotlib for the distance-over-time chart; and pytest for unit
testing. Direct dependencies are pinned with version floors in `code/prototype/requirements.txt`
so that a fresh checkout can be installed with one command. Heavy artefacts (model weights
`*.pt`, MediaPipe `.task` files, raw and trimmed videos, and all generated CSV / MP4 / PNG
output) are excluded from the repository via `.gitignore` to keep clones small.

**Source control: branching and commit discipline.** The project is hosted in a private
GitHub repository, `minh-bitrise/epee-fencing-analysis`. A simple branching model is used:
`main` holds stable, working code only; experiments and new features live on `feature/<name>`
branches (for example, the entire pose-based distance work happened on
`feature/pose-distance`, where every prototype iteration was committed). Commits are made at
green-test states - that is, only when the relevant unit-test suite passes - and follow a
multi-line convention with a short imperative subject line and a longer body explaining
*why* the change was made and what was measured or observed, rather than merely *what* the
change was. Several commits in this project are explicit checkpoints attached to specific
quantitative findings (for example, the move from strict-ID matching to spatial continuity
is recorded with the coverage drop that motivated it). This commit style is more verbose
than is strictly necessary but it is what makes the development log (the Implementation chapter) possible:
the commit history is itself the primary record of what was tried and why.

**Test discipline.** Each new component is accompanied by its unit tests in the same commit
where the component is introduced; tests are added for regressions before, not after, the
underlying fix. The full pytest suite is run before every commit and currently completes in
under two seconds, which is short enough that running it adds no friction to the iteration
cycle. Stateful components (`FencerTracker`, `PushPullTracker`) are tested directly via their
public interfaces; pure helpers are tested with table-style cases.

**Project tracking.** The repository root contains `TODO.md`, structured into a Part A
(preliminary report scope, time-bounded by the imminent submission) and a Part B (full-app
scope, with a substructure mirroring the eventual chapters of the final report) with
cross-references between the two parts so that limitations acknowledged in the report point
directly to the implementation tasks that will resolve them. A separate file,
`final_report.md` (this document), is the long-form development log and report,
intentionally maintained as the project unfolds rather than written from scratch at the end.
Completed items in `TODO.md` are checked off but not deleted, preserving the audit trail of
what the project actually went through.

**Documentation.** The repository's `README.md` describes the project at a high level and
gives setup instructions for the prototype. The `CLAUDE.md` file holds short-form context
used during development. The development log (this section's later siblings) and the
`TODO.md` cross-references together replace what would otherwise be a separate design
journal or wiki.

**Repository hygiene.** The `.gitignore` excludes Python build artefacts (`__pycache__/`,
`*.egg-info/`, `.venv/`, `dist/`, `build/`), Node artefacts left over from the early docx
generation tooling, OS files (`.DS_Store`, `Thumbs.db`), large media (`*.mp4`, `*.avi`,
`*.mov`, `*.mkv`), model weights (`*.pt`, `*.pth`, `*.onnx`, `*.task`), the generated
results folder, and any `.env` or environment-secret files. This keeps the repository to
source code, tests, the report files, and the project documentation.

---


### Ethics

The project is subject to the module's research-ethics requirements. The analysis is performed
on recorded fencing footage of identifiable athletes, so two ethical dimensions apply even
though no human-participant experiment is run on vulnerable groups.

First, **copyright and source**: the evaluation footage is publicly available video used solely
for non-commercial academic evaluation, not redistributed, with the original rights retained by
the uploaders (see the footage-attribution appendix). Second, **personal data and likeness**:
the footage shows identifiable individuals, and the system extracts position and pose data about
them. For the prototype and report, no attempt is made to identify individuals by name from the
video, no biometric identity model is used, and the derived data is used only to demonstrate the
analysis pipeline. If the full system were to store per-fencer profiles tied to named
individuals, informed consent and a data-handling/retention policy would be required, and UK
data-protection norms (the strictest applied by the module regardless of country) would govern
storage and sharing.

Any user testing of the application will be carried out with consenting healthy adults; the
project deliberately avoids children, vulnerable adults, medical patients, and animals, in line
with the module's ethics guidance.

## 4. Implementation

This section records the implementation work as it actually happened, with the intent of
producing an honest, traceable account that can be quoted from or cited in the final
submission's discussion of process and engineering decisions.

### Prototype scope and goals

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

### Pipeline overview

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

### Person detection and tracking

YOLOv8 was selected for detection because it is mature, pre-trained on a generic person class
(COCO), well documented, and bundled with a stable tracking implementation (ByteTrack) in the
Ultralytics package. The `nano` variant was chosen for the prototype because it runs adequately
on CPU; a larger variant could be substituted later without code changes.

ByteTrack's own track IDs were initially used to maintain fencer identity across frames. This
proved fragile: ByteTrack frequently reassigns IDs after occlusion or rapid motion, and a
strict-ID-matching identity scheme caused the prototype to lose track of both fencers for
extended periods (in one early test, only ~5% of frames had both fencers identified). The
identity logic was reworked to use spatial continuity instead (see the identity-tracking discussion below).

### Pose estimation

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

### Distance estimation

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

### Identity tracking and bystander rejection

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
estimates. Remaining bystander captures and close-range flicker (discussed in the Evaluation chapter) are deferred to
the full system, where motion modelling, piste-region detection, or appearance re-ID will be
deployed.

### Push / pull metric

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

### Annotated video output

The output video draws bounding boxes, fencer labels, key pose landmarks, and a bottom-of-frame
heads-up panel. The HUD shows the time, the currently smoothed distance with a colour code by
tactical zone (red ≤1.0 m, orange ≤1.8 m, green otherwise), and the cumulative push / pull
totals for each fencer. The panel sits on a semi-transparent dark rectangle with a thin white
outline so that it remains legible against the bright piste background. An earlier version
placed the HUD in the top-left and top-right corners with no background, which proved hard to
read in lighter parts of the video.

---


### Testing

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
metrics (the inflated push / pull values discussed earlier were caught this way).

---


## 5. Evaluation

The prototype has been evaluated on two independently sourced clips of FIE-level epee bouts,
both trimmed to three minutes of in-bout footage. Footage attribution is given in
Appendix C.

### Clip 1 — engagement-oriented bout

- Frames processed: 10,791
- Frames with both fencers detected: 8,842 (82%)
- Pose-based distance samples: 2,842 (~32% of detected frames, consistent with the pose
  stride of 3)
- Mean inter-fencer distance: 2.26 m
- Distance range: 0.11 m to 6.60 m
- Cumulative motion: Fencer 1 — 14.2 m push / 11.1 m pull. Fencer 2 — 15.8 m push / 17.1 m pull.

### Clip 2 — high-activity bout

- Frames processed: 9,000 (50 fps source)
- Frames with both fencers detected: 7,813 (87%)
- Pose-based distance samples: 2,327 (~30%)
- Mean inter-fencer distance: 3.33 m
- Distance range: 0.36 m to 8.45 m
- Cumulative motion: Fencer 1 — 41.0 m push / 41.5 m pull. Fencer 2 — 36.8 m push / 37.1 m pull.

### Cross-clip comparison

The two clips show a striking difference in activity. Clip 2's fencers do roughly two to three
times the footwork of clip 1's and engage at a noticeably wider average distance. This is
exactly the kind of cross-bout comparison that is effectively impossible to perform by manual
observation, and so its reproducible production by the prototype is itself a meaningful
result: the prototype can already generate metrics that distinguish bouts at a level of detail
that manual analysis cannot match.

### Detection coverage

Coverage of 82-87% (both fencers detected and assigned to their stable slots) is well above
what is needed to support a useful annotation workflow. The remaining 13-18% of frames are
largely either (a) frames where one or both fencers are partially off-screen, (b) frames where
the spatial / size gates correctly rejected a passing referee or background detection, or
(c) frames where the close-range flicker described in the Evaluation chapter caused the tracker to drop a
slot momentarily. None of these introduce wrong data into the aggregate metrics; they merely
reduce the number of contributing samples.

---



This section documents every observed limitation of the prototype and the design decisions
that drive how the full system accommodates them. It is deliberately exhaustive at this
stage; some material will be trimmed and re-balanced for the final submission, but the
intention is that everything be recorded now while the engineering reasoning is fresh.

### Observed failure modes

Prototype evaluation on two independently sourced FIE-level bout clips reproducibly revealed
two specific failure modes. Both are well-known data-association problems in single-camera
multi-object tracking and both are predicted in the design chapter as the reason the system
is AI-assisted rather than fully automatic. Their presence in the prototype is therefore not
a contradiction of the design but evidence supporting it.

**Failure mode 1: wrong-target capture by background people.** The tracker maintains stable
identity for each fencer by spatial continuity, gated by (a) a maximum allowed jump from the
slot's last known centre, expressed as a multiple of the slot's last accepted bounding-box
height, and (b) a plausible band on the bounding-box height itself. The gates correctly
reject the obvious cases - bystanders walking far across the frame, small distant figures in
the audience - and reduced the maximum observed inter-fencer distance on the first test clip
from 11.27 m (un-gated) to 6.60 m (gated) by filtering false matches that were inflating the
estimate. However, when a non-fencer such as the centre referee or a fencer on an adjacent
piste passes through a tracked slot's last known position, the candidate detection is close
enough and similar enough in size to pass both gates. Once accepted, the slot's history
updates to that wrong target; the next frame compares against the updated history; and the
slot remains locked on the wrong person until the gate eventually fails - sometimes for
several seconds. This was reproducibly observed in the second clip, where the centre referee
stepped into Fencer 2's last position during a walkback and the tracker followed the referee
rather than the fencer until they separated.

**Failure mode 2: close-range identity flicker.** When the two fencers cross or clinch within
touch range, the YOLO detector frequently merges them into a single bounding box rather than
returning two. The tracker then has only one detection to allocate, the other slot receives no
update for those frames, and when the fencers separate again the two new boxes can be
mis-assigned (in particular if the cost-minimising assignment happens to be the swapped
one). During those chaotic moments background detections (referee, adjacent-piste fencers,
audience members near the camera) are also more likely to be admitted, because the slot whose
own fencer was momentarily lost is hungry for any nearby candidate.

### Algorithmic fixes deferred to the full system

The deeper fixes for both failure modes are scoped to the full system rather than to the
prototype:

- **Piste-region detection.** The piste is visually distinctive in fencing footage (a
  rectangular, coloured area surrounded by a contrasting border). Detecting it once per
  video, by Hough-line analysis, colour segmentation, or first-frame manual selection, and
  rejecting any detection that does not overlap with it, would eliminate referees standing
  in front of the strip, audience members, and fencers on adjacent pistes in a single step.
  This is the most likely first deeper fix because the piste is the natural region of
  interest for the whole problem.
- **Motion / velocity model.** Track each slot's velocity, not just its position, and require
  candidates to be consistent with predicted motion. This catches the "moving fencer vs
  stationary referee" case that pure-position matching cannot, because the referee fails the
  velocity test even if they pass the spatial gate. This also helps the close-range flicker
  case: the system remembers the direction of motion immediately before the clinch and
  prefers the post-separation detection consistent with that direction.
- **Appearance / re-identification embedding.** A small person re-ID model could verify that
  a candidate looks like the same person previously tracked (e.g. white uniform vs the
  referee's dark suit). Heaviest of the three but the most general; useful in combination
  with the lighter techniques above.

### The assisted-annotation workflow as the design's answer

The prototype's failures do not undermine the project concept. They are exactly the
situations the proposed assisted-annotation workflow is designed to handle, by exposing a
small number of high-level user actions that resolve each observed problem with one or two
interactions rather than per-frame correction.

The actions the planned UI must expose are:

1. **Confirm or correct AI-suggested touches.** Each touch the system suggests can be
   accepted, adjusted to a different timestamp, re-assigned to the other fencer, or rejected.
   Confirming touches has a structural effect on the rest of the analysis: it establishes the
   boundaries of each in-play exchange, so that **reset and walkback periods are
   automatically excluded from aggregate metrics**. These are precisely the situations in
   which the centre referee enters the frame and triggers the bystander failure, so the
   single act of confirming touches removes most bystander-contaminated data from the
   aggregates without the user having to think about tracking at all.
2. **Add a touch the AI missed.** Manual entry of timestamp and scoring fencer, with an
   optional action label, for cases where the AI's suggestion engine misses an event.
3. **Mark a segment as tracking-unreliable.** The user selects a short time range on the
   timeline and excludes it from aggregate calculations while the raw per-frame data is
   preserved. This is the user's response to close-range flicker: a few seconds of confused
   data are simply taken out of the totals.
4. **Re-anchor a slot.** When tracking is wrong but not chaotically so, the user scrubs to a
   problem frame, indicates which fencer the system has confused (typically with a click on
   the correct fencer in the video), and the tracker resets its slot history to the user's
   selection. From that frame onwards, tracking continues with the corrected anchor.

The unifying principle behind all four actions is that the user never has to correct the
system frame by frame. Each action operates at the level at which the user already thinks
about a bout - exchanges, touches, segments, who is who - and propagates downward to fix
many frames' worth of derived metrics in one interaction.

### Distance estimation: design decisions and limitations

Distance is the single most important derived metric in the system, both as the headline
indicator on the dashboard and as the input to several other metrics (engagement distance,
tempo bands, push/pull). The choices made for it during the prototype are:

- **Measured front-foot to front-foot when pose is available.** This is the tactically
  meaningful definition of distance in fencing - coaches refer to "distance" as the gap
  between front feet in en-garde - rather than centre-of-mass separation. The pose pipeline
  identifies the front foot as the ankle whose x-coordinate is nearer to the opponent's
  reference x, where the opponent's reference x is the opponent's mid-hip if pose is
  available for the opponent and the bottom-centre of their bounding box otherwise.
- **Fallback to bounding-box bottom-centre when pose is unavailable.** This was a deliberate
  change from an earlier centre-of-box fallback: the centre of a bounding box is
  contaminated by arm and weapon extension, which can move the centre even when the fencer's
  body has not moved, whereas the bottom-centre is approximately at foot level and is
  considerably more stable.
- **Normalised to metres using fencer height as a scale reference.** The system assumes an
  average fencer height of 1.75 m and uses the mean of the two fencers' bounding-box heights
  as the pixel-to-metre scale for that frame. This is approximate - actual heights vary,
  camera angle distorts the projection, perspective changes as fencers move toward and away
  from the camera - but it is sufficient to produce values consistent with expected
  engagement ranges (1.5-3 m for active fencing) and, importantly, it is *consistent enough
  for relative comparison across a single bout*, which is the main use case.
- **Rolling-median smoothing for the displayed value.** The raw frame-to-frame distance is
  noisy because the underlying bounding-box edges and pose landmarks both have small
  fluctuations even when the fencers are essentially stationary. A rolling-median smoother
  with a window of five frames is applied to the displayed number; the raw values are still
  written to the CSV for any downstream analysis that wants them.
- **Colour-coded tactical zones.** The on-screen distance value is shown in red (≤1.0 m,
  touch range), orange (≤1.8 m, engagement range), or green (further out), which gives an
  immediate visual signal about the current tactical situation as the video plays.

Known limitations of the distance estimate that follow from these choices:

- It is **only metrically accurate to within roughly one fencer-height** because the scale
  reference is itself approximate. It should be reported in the UI as an *estimated*
  distance, not a precise measurement.
- It assumes **a roughly side-on camera view**. Substantial camera angle relative to the
  piste will systematically distort the metric value. Large camera pans are clamped out
  separately for the push/pull metric but the distance value itself is computed per-frame
  and is therefore sensitive to camera motion within a single frame.
- It does not yet **distinguish between in-play distance and reset distance**. This is by
  design at the prototype stage; in the full system, distance is averaged and analysed only
  over in-play segments determined by the user's confirmed touches (see the assisted-annotation workflow discussion).
- It is **front-foot-to-front-foot only in a horizontal/2D sense**; depth differences
  between the two fencers (one closer to the camera than the other) are not modelled
  because we only have a single camera. This is acceptable for a side-on view where both
  fencers lie on the same plane.

### Identity and matching: design decisions and limitations

The `FencerTracker` design - spatial continuity rather than strict ByteTrack-ID matching,
with spatial and size gates - is the result of three distinct iterations during development
(documented in the identity-tracking discussion) and represents a deliberate engineering tradeoff between
robustness to ID changes and rejection of obvious bystanders. The current gate settings
(`GATE_DISTANCE_RATIO=3.5`, `MIN_SIZE_RATIO=0.4`, `MAX_SIZE_RATIO=2.2`) were chosen
empirically to give ~82% detection coverage on the first test clip while still meaningfully
reducing wrong-target captures (max distance from 11.27 m to 6.60 m). Tighter settings
improved bystander rejection but dropped coverage below 60%; looser settings restored
coverage but admitted obvious bystanders. The choice was made to err slightly on the side of
admitting borderline cases (because aggregate metrics will be cleaned up by the
assisted-annotation workflow anyway) rather than to err on the side of dropping coverage.

### Push / pull metric: design decisions and limitations

The push / pull metric records, in metres, how much each fencer has moved toward the
opponent ("push") and away from the opponent ("pull") over the bout. It exists because the
preliminary report's introduction promises metrics that are essentially impossible to
measure by hand; push and pull are the clearest such example.

Design decisions made for this metric:

- **Smoothed reference x.** Each fencer's horizontal position is smoothed with a rolling
  median (default window 5) before frame-to-frame movement is computed. Without this,
  bounding-box edge jitter accumulated into clearly inflated totals: an early test run
  produced 216 m of push and 214 m of pull per fencer in a 3-minute bout, which is
  physically impossible given a 14 m piste and represents jitter being summed by the
  accumulator.
- **Per-frame motion clamp.** Any frame-to-frame movement greater than 0.15 m is treated as
  camera motion or detection jitter and ignored entirely. This corresponds to a
  biomechanical limit of about 9 m/s, which exceeds the speed of even fast lunges and so
  does not reject legitimate fencer motion.
- **Noise floor.** Any frame-to-frame movement smaller than 0.03 m is treated as noise and
  not accumulated. This prevents thousands of tiny jitters from summing into a misleading
  total.
- **Reference x is mid-hip when pose is available, bbox-bottom-centre otherwise.** Using
  hips is more stable than using the centre of a bounding box because the hip is unaffected
  by arm and weapon extension.

After the smoothing, clamp and noise-floor changes the same 3-minute test clip produced push
and pull totals in the 13-22 m range per fencer, which is biomechanically plausible for
active fencing. The two test clips also produced very different push / pull totals,
demonstrating that the metric *does* distinguish between bouts and is not simply a noise
floor - the high-activity clip 2 produced 41 m of push per fencer, versus 14 m in the more
static clip 1.

Known limitations:

- The metric **accumulates during walkback and reset periods**. This is the same issue as
  with distance and will be addressed in the same way: in the full system, push and pull
  are aggregated only over in-play segments determined by confirmed touches.
- The metric **inherits the approximation of the pixel-to-metres scale**. It is more
  trustworthy as a *relative* indicator (this fencer pushed twice as much as the other,
  or this bout was twice as active as that one) than as a precise metric value.
- The metric **is sensitive to camera motion**. Large pans are clamped out by the per-frame
  motion ceiling but smaller, smoother pans are not detected and can inflate both push and
  pull symmetrically.

### Pose estimation: design decisions and limitations

Pose is the most expensive step in the pipeline and contributes to the prototype's wall-clock
processing time more than any other component. The decisions taken in the prototype are:

- **MediaPipe `pose_landmarker_lite` Tasks API.** Chosen for low setup cost, broad
  documentation, and adequate accuracy on standing-human poses. The earlier
  `mediapipe.solutions.pose` API is removed in current MediaPipe releases (0.10+); the
  project migrated to the new Tasks API.
- **Run on cropped bounding-box regions.** Running pose on each fencer's crop separately
  (with small padding) rather than on the whole frame both constrains the pose model to the
  right person and gives it an easier image to work with.
- **Pose stride.** Pose is by default run every third frame. This trades pose-sample
  frequency (20 Hz at 60 fps, instead of 60 Hz) for a roughly threefold reduction in total
  wall-clock time, and was the change that brought a 3-minute clip's processing time from
  ~38 minutes (under memory pressure) to a few minutes once memory was free.
- **Graceful fallback.** When pose fails (occlusion, motion blur, partial view) the distance
  computation falls back to bounding-box bottom-centre. The CSV records the method used per
  frame.

Known limitations:

- Pose **fails noticeably under fast motion and occlusion**. This is consistent with the
  literature (Hong et al., 2021) and was an explicit design assumption from the outset.
  Hence the fallback strategy and the assisted-annotation workflow.
- Pose **is computed on a fixed-shape crop**. If the bounding box is wrong (e.g. includes
  parts of another fencer during a clinch), the pose may attach to the wrong body. This
  is part of why pose alone is not used as the sole identity signal.

### Engineering and process

A few observations about the development process itself, worth recording while still fresh:

- The fast unit-test suite (currently 55 tests across the pure utility functions and the
  stateful trackers) caught regressions multiple times during this project, most notably
  the inflated push / pull values, which were initially flagged not by the tests
  themselves but by the obvious implausibility of the numbers; that observation then
  motivated the tests for jitter accumulation. This pattern of *observation on real footage
  motivates a regression test* is one of the more valuable workflows in this project.
- The cycle of *run on a real clip, find a failure mode, design a small targeted fix, add
  a test for it, re-run* has worked well at the prototype scale; whether it scales to the
  full system is an open question.
- Memory pressure on the development machine (Apple Silicon, 16 GB) is the single biggest
  factor in wall-clock processing time. Closing browser tabs and other heavy applications
  reduced clip processing time from ~38 minutes to a few minutes without any code change.
  This is a useful operational note for anyone reproducing the prototype.

---


## 6. Conclusion

*(Placeholder — to be written at the end of the project.)*

---


### Future work

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


## References

*(Placeholder — the preliminary report's reference list will be carried over and extended
here in Harvard style.)*

---


## Appendices

### A. Source code overview

The prototype is in `code/prototype/`, with `run_detection.py` as the main pipeline and
`test_detection.py` as the unit-test suite. Dependencies are declared in
`code/prototype/requirements.txt`.

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
