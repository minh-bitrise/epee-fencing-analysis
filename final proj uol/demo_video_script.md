# Preliminary Report - Demo Video Script

**Target length: 4 min (within the 3-5 min requirement). Required format: MP4.**

This script is a working draft. It is structured as a shot list with timing on the left and
spoken narration on the right. Read it through aloud once before recording to check pacing; the
spoken text below comfortably fits inside four minutes at a normal pace.

The aim is to hit four marking criteria explicitly:
1. clearly show what the prototype does (effective and impactful demonstration);
2. show that it is technically substantive (three pre-trained / AI components orchestrated);
3. evaluate it honestly, including its failure modes (suitable improvements identified);
4. tie it back to the project's central design choice (AI-assisted, not fully automatic).

---

## Recording setup (before you press record)

- Have the annotated clip 1 (`fencing_clip_annotated.mp4`) open in a full-screen video player.
  Pre-scrub to a known good engagement section so the first thing the viewer sees is the system
  working well.
- Have the distance plot (`fencing_clip_distance_plot.png`) and the first few rows of the CSV
  (`fencing_clip_distance.csv`) ready in separate windows for cuts.
- Have one screenshot of a failure-mode frame ready (e.g. the referee captured as Fencer 2),
  so the "honest about failure" section has something visual to point at.
- Use screen recording at 1080p with the system microphone; clear background, no music.

---

## Section 1 - Title and what this project is (0:00 - 0:30)

**Shot:** Title slide or the first frame of the annotated video, paused.

> *"My name is Nguyen Anh Minh and this is the feature-prototype demonstration for my CM3020
> Artificial Intelligence final project, on AI-assisted analysis of epee fencing bouts. The
> project template is Project Idea 1 - Orchestrating AI Models to Achieve a Goal. The aim of the
> full project is a web application that takes a recorded bout video and turns it into useful
> tactical information for fencers and coaches. This demo shows the prototype: the part of the
> AI pipeline I have built so far."*

---

## Section 2 - The problem in one sentence (0:30 - 0:50)

**Shot:** Still on the title or the paused first frame.

> *"Fencing video review is currently manual, slow, and inconsistent, and most of the
> tactically interesting numbers - the distance between fencers over time, how much each fencer
> pushed forward versus retreated, the engagement distance at the moment of a touch - are
> effectively impossible to measure by eye. The prototype shows that an AI pipeline can produce
> exactly those numbers automatically."*

---

## Section 3 - The pipeline, briefly (0:50 - 1:20)

**Shot:** Brief diagram or bullets on screen, OR just the video paused. Keep the
visual simple - the voice does the work here.

> *"The prototype is a Python pipeline that does four things, per frame of video. First, it
> uses YOLOv8 with ByteTrack to detect and follow people on screen. Second, it identifies
> which two of those are the actual fencers using a custom matcher that maintains a stable
> Fencer 1 and Fencer 2 identity by spatial continuity, and rejects obvious bystanders by a
> spatial gate and a size gate. Third, it runs MediaPipe pose estimation on each fencer to find
> body keypoints, including the ankles. And fourth, it measures front-foot to front-foot
> distance, normalised to metres, and accumulates how much each fencer has moved toward and
> away from the opponent."*

---

## Section 4 - Live walkthrough of the annotated video (1:20 - 2:40)

**Shot:** Play the annotated `fencing_clip_annotated.mp4`. Talk over what is happening on
screen. Let the visuals do the demonstration - this is the most important section of the
video.

> *(As the boxes appear)*
> *"Each fencer has a coloured box - orange for Fencer 1, green for Fencer 2 - with their
> label sticking with them as they move. The small dots are the pose keypoints: the ankles,
> hips and shoulders the system is using for distance and for footwork analysis."*
>
> *(As the bottom HUD updates)*
> *"At the bottom of the frame is the readout. On the left, the current distance between the
> two fencers, colour-coded: green when they are at safe distance, orange when they are inside
> the engagement range of about 1.8 metres, and red when they are within touch range. The
> word `pose` or `bbox` next to the value tells you whether this frame used pose-based or
> bounding-box-based measurement."*
>
> *(Pause as fencers close in and the distance turns red)*
> *"Watch the colour change - they are closing the distance into touch range here. This is the
> kind of moment a coach wants to find quickly when reviewing a bout."*
>
> *(Point at the right-hand stats)*
> *"On the right, the running totals: how much each fencer has pushed forward toward the
> opponent, and how much they have pulled back, both in metres. This number is impossible to
> work out by eye and it is one of the most tactically informative things a fencer can know
> about their own footwork over a bout."*

---

## Section 5 - The data output (2:40 - 3:00)

**Shot:** Cut briefly to the distance-over-time plot, then to the CSV.

> *"The pipeline also produces a per-frame CSV with raw and smoothed distance, the method
> used, and the running push and pull totals for each fencer. And a distance-over-time chart
> that lets you see engagement patterns at a glance. These are the building blocks of the full
> system's dashboard."*

---

## Section 6 - Honest about failure modes (3:00 - 3:30)

**Shot:** Pre-cued frame from one of the test clips that shows a failure - for example
the moment the referee is captured as Fencer 2 in clip 2.

> *"I want to be honest about what the prototype gets wrong. Two failure modes came up
> reproducibly in testing. The first is wrong-target capture: when a background person like the
> centre referee passes close to a fencer's last known position, the matcher can lock onto
> them, as you can see here. The second is close-range flicker: when the two fencers cross or
> clinch within touch range, the detector sometimes merges them into one box, and on
> separation the labels can briefly swap. Both are well-known data-association problems in
> single-camera tracking and are documented in the report."*

---

## Section 7 - How the design handles this (3:30 - 3:50)

**Shot:** Back on the annotated video, or on a simple list of the four actions.

> *"And this is exactly why the system is AI-assisted rather than fully automatic. The full
> application's annotation interface lets the user confirm touches, which automatically
> excludes walkback and reset periods from the metrics - which is where most of these failures
> happen. The user can also mark a short segment as unreliable, or click on the correct fencer
> to re-anchor a slot. So the user fixes a few high-level boundaries; they don't correct frames
> one by one."*

---

## Section 8 - Closing (3:50 - 4:00)

**Shot:** Final frame or simple closing slide.

> *"That is the feature prototype: detection, tracking, pose, distance, and cumulative push
> and pull, with the failure modes acknowledged and the design's response to them in place.
> Thank you for watching."*

---

## Backup / extension lines

If timing comes in short, drop into Section 3 or 6:

- **Tech-depth filler:** *"All three of the pre-trained models the project uses come from
  different domains - YOLO is a CNN-based object detector, MediaPipe Pose is a regression-based
  keypoint model, and the planned language model will operate on text - so the project
  satisfies the template's requirement of orchestrating models across different data types."*
- **Evaluation filler:** *"On the test clip, 82% of frames had both fencers correctly tracked,
  pose was available for about a third of those, mean engagement distance was 2.26 metres, and
  the spatial and size gates dropped the maximum reported distance from 11.27 metres down to
  6.60 metres by filtering wrong-target outliers."*

If timing runs long, the simplest cuts are:
- Section 5 (data output) - can be one sentence over a quick image flip
- The technical details in Section 3 - can be shortened to "three pre-trained models in
  a pipeline"
