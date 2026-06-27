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

> *"Hi, my name is Nguyen Anh Minh, and in this demo I'll walk you through the feature
> prototype for my CM3020 Artificial Intelligence final project: an AI-assisted analysis tool
> for epee fencing bouts. The project template I'm working under is Project Idea 1,
> Orchestrating AI Models to Achieve a Goal. What I'm aiming for in the full project is a web
> application that takes a recorded bout video and turns it into useful tactical information
> for fencers and coaches. What I'll show you today is the part of the AI pipeline I've built
> so far."*

---

## Section 2 - The problem in one sentence (0:30 - 0:50)

**Shot:** Still on the title or the paused first frame.

> *"The reason I picked this problem is that fencing video review, at the amateur and club
> level I come from myself, is still done by hand. It's slow, it's inconsistent, and most of
> the tactically interesting numbers - the distance between the two fencers over time, how much
> each fencer pushed forward versus retreated, the engagement distance at the moment of a
> touch - are effectively impossible to measure by eye. What I want to show in this prototype
> is that an AI pipeline can produce exactly those numbers automatically."*

---

## Section 3 - The pipeline, briefly (0:50 - 1:20)

**Shot:** Brief diagram or bullets on screen, OR just the video paused. Keep the
visual simple - the voice does the work here.

> *"What I built is a Python pipeline that does four things, per frame of video. First, I use
> YOLOv8 with ByteTrack to detect and follow people on screen. Second, I run those detections
> through a custom matcher I wrote that keeps a stable Fencer 1 and Fencer 2 identity by
> spatial continuity, and that filters out obvious bystanders using a spatial gate and a size
> gate. Third, I run MediaPipe pose estimation on each fencer to extract body keypoints,
> including the ankles. And fourth, I measure front-foot to front-foot distance, normalised to
> metres, and accumulate how much each fencer has moved toward and away from the opponent."*

---

## Section 4 - Live walkthrough of the annotated video (1:20 - 2:40)

**Shot:** Play the annotated `fencing_clip_annotated.mp4`. Talk over what is happening on
screen. Let the visuals do the demonstration - this is the most important section of the
video.

> *(As the boxes appear)*
> *"You can see each fencer has a coloured box: orange for Fencer 1, green for Fencer 2, and
> the labels stick with them as they move along the piste. The small dots inside each box are
> the pose keypoints I'm extracting: ankles, hips, and shoulders - those are what I use for
> distance and for footwork analysis."*
>
> *(As the bottom HUD updates)*
> *"At the bottom of the frame is the readout I designed. On the left is the current distance
> between the two fencers, colour-coded by tactical range: green when they're at safe
> distance, orange when they're inside the engagement range of about 1.8 metres, and red when
> they're within touch range. The `pose` or `bbox` label next to the number tells you whether
> the system used the pose-based measurement on that frame or the bounding-box fallback."*
>
> *(Pause as fencers close in and the distance turns red)*
> *"Watch the colour change here - they're closing into touch range. This is exactly the kind
> of moment a coach wants to be able to jump to quickly when reviewing a bout."*
>
> *(Point at the right-hand stats)*
> *"On the right are the running totals: how much each fencer has pushed forward toward the
> opponent, and how much they've pulled back, both in metres. This is the number I'm most
> proud of, honestly, because it's one I genuinely couldn't work out by eye as a fencer
> myself, and it's one of the most tactically informative things you can know about your own
> footwork over a bout."*

---

## Section 5 - The data output (2:40 - 3:00)

**Shot:** Cut briefly to the distance-over-time plot, then to the CSV.

> *"On top of the annotated video, the pipeline also writes out a per-frame CSV with raw and
> smoothed distance, the method used, and the running push and pull totals for each fencer.
> Plus a distance-over-time chart so you can see engagement patterns at a glance. These are
> what I'll build the full system's dashboard on top of."*

---

## Section 6 - Honest about failure modes (3:00 - 3:30)

**Shot:** Pre-cued frame from one of the test clips that shows a failure - for example
the moment the referee is captured as Fencer 2 in clip 2.

> *"I do want to be upfront about what the prototype gets wrong, because I think the failures
> are as informative as the successes. Two failure modes came up reproducibly when I tested
> on real footage. The first is wrong-target capture: when a background person, typically the
> centre referee, passes close to a fencer's last known position, the matcher can lock onto
> them - you can see that happening in this frame. The second is close-range flicker: when the
> two fencers cross or clinch within touch range, the detector sometimes merges them into one
> box, and on separation the labels can briefly swap. Both of these are well-known
> data-association problems in single-camera tracking, and I documented them in the report."*

---

## Section 7 - How the design handles this (3:30 - 3:50)

**Shot:** Back on the annotated video, or on a simple list of the four actions.

> *"And honestly, this is exactly why I designed the system as AI-assisted rather than fully
> automatic. In the full application, the annotation interface will let the user confirm
> touches, which automatically excludes the walkback and reset periods from the metrics, and
> that's precisely where most of these failures happen. The user will also be able to mark a
> short segment as unreliable, or click on the correct fencer to re-anchor a slot. So the
> user is fixing a few high-level boundaries, not correcting frames one by one."*

---

## Section 8 - Closing (3:50 - 4:00)

**Shot:** Final frame or simple closing slide.

> *"And that's the prototype: detection, tracking, pose, distance, and cumulative push and
> pull, with the failure modes acknowledged and a clear design response to them. Thanks for
> watching."*

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
