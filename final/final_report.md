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

*(Placeholder - to be written at the end of the project. Will summarise the problem, approach,
key results, and contribution in 200-300 words.)*

---


## 1. Introduction

### Context

Video review is now routine in sports performance analysis, but in fencing the review itself has
remained manual across the whole sport rather than only at its amateur end. Coaches at elite,
FIE and Olympic level still scrub footage by hand and tag actions themselves; no automated
pipeline is in widespread use anywhere in the sport. What differs by level is the resources
brought to the same manual task, not the task. At club level the consequence is that recorded
bouts often sit unreviewed altogether, or are reviewed by watching whole clips repeatedly for
qualitative impressions, and even dedicated fencing analysis tools require the user to label
every action by hand, which is a time cost that discourages routine use.

The addressable need is therefore broader than accessibility for amateurs, although amateur
accessibility is this project's primary user focus: a club fencer has no analyst and no budget,
so they are the user for whom automation makes the difference between analysis happening and
not happening.

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

### Contributions

The project delivers four things, listed here so that the chapters that follow can be read
against them.

**A working end-to-end system.** A bout video is uploaded through a browser, processed by a
background job runner, and reviewed through an annotation interface, with no terminal involved.
Three pre-trained models are chained rather than demonstrated separately: person detections
bound the region pose estimation works in, pose landmarks supply the ground reference the
distance series is measured from, that series drives touch proposal, confirmed touches scope
every aggregate, and those aggregates are what a language model is given to interpret.

**A geometric touch detector, evaluated against hand-labelled ground truth.** Twenty-seven
touches were labelled across four independently sourced bouts, with clips held back from tuning
so that generalisation could be tested rather than assumed. An audio-based detector was built
first, shown to fail on held-back footage, and replaced.

**Two capabilities the detector could not supply on its own.** Attribution of a touch to a
fencer, read from the scoring machine's lamps, and lunge proposals calibrated per bout from
lunges the user has already confirmed. Both are reported with the limits measurement established
rather than as finished features.

**A methodological result about measuring one's own system.** Repeatedly during this project a
metric survived its own derivation and then failed a test of what it could actually detect: a
classifier using invented rather than derived thresholds, a movement total that accumulated its
own smoothing, an ablation that varied a quantity the models never see, a spatial filter that
improved a headline number while making the underlying result worse, and a tracker choosing
between candidates on differences indistinguishable from noise. The Evaluation chapter treats
that pattern as a finding in its own right, because the diagnosis in each case came from
measurement against an external constraint rather than from further reasoning.

### Scope

Deliberately out of scope: refereeing decisions and right-of-way, which epee does not require
and which the system makes no attempt to judge; identification of individual athletes, which
raises personal-data questions the Design chapter's ethics section addresses and which the
system avoids by tracking anonymous slots; and multi-camera or three-dimensional reconstruction,
which would resolve several measurement limitations and is treated as further work.

### Project template

This project corresponds to the CM3020 Artificial Intelligence template "Project Idea 1:
Orchestrating AI Models to Achieve a Goal", which expects an integrated software system built
around at least three pre-trained models applied to a clearly defined problem.

### Structure of this report

The Implementation chapter is written as a development log rather than a description of a
finished artefact: it records what was built, in what order, and where the direction changed,
because on this project the changes of direction carry most of the evidence. The Evaluation
chapter reports results on real footage and judges them against the aims above.

---


## 2. Literature Review

This chapter reviews academic literature and existing systems relevant to the proposed
application across four themes: the broad application of deep learning to sports video
analysis; temporal action detection in untrimmed video; human pose estimation for
fine-grained sports actions; and the human-in-the-loop paradigm that underpins the project's
assisted-annotation design. It then examines existing fencing- and sports-analysis systems in
order to position the project relative to current practice. Throughout, the emphasis falls on
identifying not only what prior work achieves but also the limitations and gaps that motivate
the present project. Where the implementation reported in Chapter 4 has since tested a claim
from this literature against real footage, this chapter notes the outcome, because a
literature review that informed a build should be readable alongside what the build found.

### 2.1 Deep learning in sports video analysis

Rangasamy et al. (2020) provide a useful entry point, contrasting traditional
handcrafted-feature approaches with deep learning methods for human activity recognition in
sport. Their review establishes that convolutional neural networks have become the dominant
approach for frame-level understanding, with recurrent and temporal models capturing motion
across sequences. The value of this work for the present project is methodological: it
confirms that deep learning is the appropriate family of techniques for extracting structured
information from sports footage. However, the review treats analysis largely as a fully
automatic classification problem, and it is necessarily broad rather than fencing-specific. It
does not address how such systems should behave in domains where reliable full automation
remains out of reach, which is precisely the situation this project confronts.

### 2.2 Temporal action detection in untrimmed video

Vahdani and Tian (2021) survey deep-learning approaches to temporal action detection, the task
of localising when actions occur within long, untrimmed footage. This bears directly on the
proposed system, because a fencing bout is a continuous untrimmed recording in which touches
and exchanges are sparse and separated by footwork, preparation and pauses. Their taxonomy of
supervision levels, ranging from fully supervised to weakly, self- and semi-supervised, is
particularly instructive. The fully supervised methods that dominate published benchmarks
depend on dense, frame-level boundary annotations that are expensive to produce and that do
not exist for any fencing-specific dataset. The survey is also explicit that temporal action
detection remains an open research problem rather than a solved capability.

Two implications follow. First, automatic event localisation should highlight probable
segments for human review rather than serve as a reliable end in itself. Second, the absence
of annotated fencing data makes a fully supervised, fully automatic system infeasible within
the scope of an undergraduate project. The realistic and well-grounded contribution is
therefore to reduce manual review time through assisted highlighting.

### 2.3 Pose estimation for fine-grained sports actions

Hong et al. (2021), presented at ICCV 2021, introduce Video Pose Distillation and demonstrate
that human pose is a strong cue for fine-grained sports action recognition. Equally
importantly, they show that off-the-shelf pose estimators degrade in sports footage as a
result of motion blur, occlusion and domain shift. This finding applies acutely to fencing,
where two athletes overlap on a narrow piste and move explosively, producing exactly these
failure conditions. Their conclusion that pose is informative yet unreliable supports a design
in which pose-derived features inform suggestions that a user subsequently confirms, rather
than driving fully automatic decisions.

The prototype reported in Chapter 4 bears this out concretely. Pose succeeded on roughly 30 to
32 per cent of frames where both fencers were tracked, a figure governed mainly by the
deliberate pose stride of three frames, and it failed most often during precisely the
clinches and rapid exchanges Hong et al. identify. The pipeline therefore falls back to a
bounding-box geometry estimate on those frames rather than treating a missing pose as missing
data. Hong et al.'s distillation method is, however, a model-training contribution aimed at
improving recognition accuracy on curated datasets; it does not deliver an end-user analysis
tool, and reproducing its training pipeline lies beyond this project's scope. This project
therefore adopts pose estimation as a pragmatic, pre-trained component within an assisted
workflow rather than attempting to advance pose recognition itself.

### 2.4 Fencing-specific computer vision

The closest domain-specific academic work is Mo (2022), whose "Allez Go" system applies pose
estimation and a lightweight temporal convolutional network, augmented with audio analysis, to
referee fencing bouts. It reports approximately 89 to 90 per cent accuracy in classifying
which fencer scored, using a custom dataset of around 4,000 international-level clips, with
audio cues detecting blade contact. This is valuable evidence that pose-based fencing analysis
is feasible and that audio can usefully complement visual features.

Two contrasts with the present project matter. First, Allez Go pursues automatic refereeing, a
high-stakes binary decision, whereas the proposed system pursues assisted profiling and
review, a lower-stakes task that tolerates imperfect automation far better. Second, Allez Go
trains on elite, well-filmed competition footage. It is worth being precise about what this
does and does not imply. It would be wrong to suggest that manual review is a problem only for
amateurs: post-bout analysis remains largely manual across the whole sport, including at FIE
and Olympic level, where coaches still scrub footage by hand and tag actions themselves, and
no automated pipeline has been widely adopted. What elite footage does provide is better
conditions for automation, namely multiple fixed camera positions, consistent framing and
professional lighting. Club and amateur recordings are typically single-camera, less
standardised and noisier, so they weaken the case for full automation further still. Allez Go
thus simultaneously demonstrates technical feasibility and reinforces the rationale for an
assisted rather than autonomous design, with the assisted approach mattering most where
footage quality is lowest.

### 2.5 Human-in-the-loop machine learning

The assisted-annotation workflow at the centre of the project is not merely a pragmatic
compromise but an established methodology. Mosqueira-Rey et al. (2023) survey human-in-the-loop
machine learning, in which humans and models collaborate so that human input corrects and
guides model output. This provides theoretical legitimacy for a system whose AI suggestions the
user reviews, confirms or corrects, reframing manual annotation from a weakness into a
deliberate strategy that improves reliability and yields verified data. Their review
distinguishes active learning, interactive machine learning and machine teaching; the proposed
system most closely resembles interactive machine learning, in which the user iteratively
refines model outputs.

This grounding also shapes how Chapter 5 evaluates the system. The appropriate criteria are
human-in-the-loop criteria: the reduction in manual effort relative to fully manual review,
the quality of the resulting annotations, and the number of interactions a user needs in order
to repair a given class of error. That last criterion is what makes the tracking failures
reported in Chapter 5 tolerable rather than fatal, since each one is designed to be
correctable by a single high-level user action rather than frame-by-frame editing.

### 2.6 Existing systems

Beyond the academic literature, two existing tools illustrate the practical landscape.
Fencing Manager (2026) is a fencing-specific application that allows users to upload footage
and manually tag actions. Its domain alignment confirms genuine demand for structured fencing
analysis, but it offers no AI assistance, requires the user to watch entire bouts and label
every event by hand, and provides no derived analytics such as inter-touch timing, distance
analysis or opponent profiling. Dartfish, a mature general-purpose sports video-analysis
platform, demonstrates the proven value of video analysis across many sports, but it is not
fencing-specific, depends heavily on manual operation, and assumes a degree of analyst
expertise that club-level users frequently lack.

Together these systems define the gap this project addresses: Fencing Manager offers domain
focus without AI assistance, while Dartfish offers analytical sophistication without fencing
specificity or accessibility. Neither computes the geometric metrics that motivate this
project. Inter-fencer distance over time and cumulative advance and retreat per fencer are not
merely absent from these tools; they are impractical to produce by hand at all, which is the
substantive argument for automating the measurement layer even when event interpretation stays
with the user.

### 2.7 Synthesis

Taken together, the literature establishes three points that justify the project's concept and
scope. First, deep-learning-based video analysis is the appropriate methodological foundation
(Rangasamy et al., 2020), and pose estimation in particular is an informative cue for
fine-grained sports actions (Hong et al., 2021). Second, the techniques most relevant to
fencing, namely temporal action detection and sports pose estimation, are powerful but
unreliable on realistic, unconstrained footage (Vahdani and Tian, 2021; Hong et al., 2021),
which makes fully automatic analysis an unsuitable goal. Third, both the human-in-the-loop
literature (Mosqueira-Rey et al., 2023) and the limitations of existing systems point toward
an assisted approach that combines automated suggestion with human verification.

The proposed application occupies the gap left by prior work. It brings AI assistance to a
fencing-specific, club-accessible niche that Fencing Manager and Dartfish leave unserved,
while adopting the realistic, human-in-the-loop stance that the technical literature implies is
necessary. One limitation of this review should be acknowledged: the fencing-specific
literature is thin, resting substantially on a single system (Mo, 2022), so several arguments
here transfer findings from adjacent sports rather than resting on fencing-specific evidence.
The prototype results in Chapter 5 provide a small amount of direct fencing evidence to set
against that gap.

---


## 3. Design

*(Placeholder - the preliminary report contains a 1,400-word design chapter including a layered
architecture diagram (Client / Application / Processing pipeline / Data) and a Gantt chart.
That material will be carried over here and extended, including an expanded data-flow
description for the assisted-annotation workflow. The revised, submission-ready version of
chapters 1 to 6 lives in `final/draft_report.md`; this file remains the long-form log and holds
the fuller rationale that the word-limited chapters can only summarise.)*

### Touch detection: design rationale

This section records the design reasoning for the touch-detection stage before it was built, so
that the justification survives independently of the code. Much of the domain reasoning below
came from the author's own competitive knowledge rather than from the literature, and it
materially changed the design.

#### Why a single signal cannot work

The obvious signal is the scoring machine's buzzer, and Mo (2022) validates audio as a cue for
fencing analysis. On our footage the buzzer is measurable: band-passing clip 3's audio around
3.2 kHz isolates eighteen narrow-band events in three minutes, distinguishable by duration into
sustained tones and short transients.

Audio alone is nevertheless insufficient, for three reasons that are specific to how fencing is
actually filmed and practised.

1. **Beeps from adjacent pistes.** At a competition, several bouts run simultaneously within
   earshot. The microphone cannot tell which scoring machine fired.
2. **Deliberate weapon testing.** Fencers test a blade by striking the floor or a guard, which
   registers on the machine and produces an identical beep. This happens during breaks, not
   during fencing.
3. **Blade contact.** Parries and beats produce high-frequency metallic transients that resemble
   a short buzzer in the spectral domain.

#### Multi-signal candidate scoring

The design therefore scores each candidate on several features rather than thresholding one.
Each feature is included because it discriminates a specific false positive above, which is the
test a feature must pass to earn its place:

| Feature | Source | False positive it rules out |
|---|---|---|
| Sustained tone vs short transient | audio duration | blade contact |
| Fencers within touch-possible distance | existing distance metric | adjacent-piste beeps |
| Distance was closing before the event | existing distance metric | ambient noise |
| Both fencers halt afterwards | tracked positions | weapon testing |
| Event is temporally isolated | audio event clustering | weapon testing (it comes in bursts) |
| Scoring lamp flash or score change | frame region analysis | all of the above |

Two consequences follow. First, the existing distance and position pipeline is reused as a
filter rather than extended, so touch detection adds a stage without adding a model. Second,
because the design only needs to *propose* candidates for user confirmation, recall matters
considerably more than precision, and a permissive detector with a confidence score is
preferable to a strict one. This is a materially easier target than Mo (2022), whose system
decides *which* fencer scored and reports 89 to 90 per cent accuracy on that harder task.

**Double touches are an attribution problem, not a detection problem.** In epee both fencers
score if they land within the machine's lockout interval. The event is detected identically; only
the assignment differs, and assignment is already the user's responsibility.

#### A capability hierarchy rather than a single accuracy figure

Available signals differ systematically by footage type, so the design degrades in tiers rather
than failing:

- **Tier 1, score change.** Broadcast score overlays and venue scoring machines are often
  visible. Reading them gives both timing and attribution.
- **Tier 2, buzzer with geometric corroboration.** Detects hits; the user resolves validity and
  attribution.
- **Tier 3, geometry alone.** For club footage with no visible machine and no usable audio.
  Weakest, and requires the most correction.

Reporting the system's capability per tier is more honest than a single number, because the
number would otherwise average over footage types that offer categorically different information.

#### The annulment argument

Tier 1 is not merely the most convenient signal but the only one that reflects the referee's
decision. A scoring machine registers a valid electrical contact; a score display registers an
awarded point. These differ whenever the referee annuls a touch, for corps-a-corps, for covering
target, or for any other non-valid action.

This yields a stronger justification for the assisted-annotation design than the one given in
the literature review. That argument rests on models being unreliable on real footage, which
invites the reply that better models would remove the need for user involvement. The annulment
case is not of that kind. When a referee annuls a touch, the hit occurred, the machine fired, and
no point was awarded: **the information distinguishing those outcomes is absent from the video and
audio entirely.** It is a refereeing judgement, not a physical event, and no model of any quality
can recover it from the recording.

The touch detector therefore cannot be correct in principle, only useful. That is not a
limitation to apologise for; it is the reason the system is designed as an assistant. Where a
score display is readable the system can observe the consequence of the referee's decision
without modelling the decision itself, which is the closest automation can come. Everywhere else,
a human supplies what the recording does not contain.

This argument should replace the weaker "models are noisy" framing wherever it appears in the
literature review and design chapters.

### Touch detection: implementation and first evaluation

Implemented as `detect_touches.py`, with `evaluate_touches.py` scoring its output against
hand-labelled ground truth. Ground truth for clip 3 (fourteen awarded touches, four of them
doubles) was labelled by the author from audio and video together and is stored in
`ground_truth/`.

**Result at the tuned operating point:** precision 0.79, recall 0.79, F1 0.79, with eleven of
fourteen touches found. Expressed as user effort, six corrections against a manual baseline of
fourteen entries.

Per-clip band calibration proved necessary rather than merely tidy. The buzzer's pitch differs by
venue and machine: 1200-1700 Hz on clip 3, 2700-3200 Hz on clip 2, 1700-2200 Hz on clip 4. A
hardcoded frequency would have worked on at most one of them.

#### Two features that failed, and why

Both failures are worth recording because in each case the code was correct and the reasoning
behind it was not.

**The halt feature measured the wrong thing.** The design predicted that fencers stop after a
touch, so the first implementation tested whether combined push and pull movement fell to near
zero afterwards. Measured against ground truth this separated real touches from false positives
by a factor of 1.04, which is to say not at all. Two causes compounded. Push and pull are
inflated by camera panning, so on hand-held footage the metric never goes quiet even when the
fencers do, meaning a known defect in one metric silently disabled a feature in a later stage.
More fundamentally the premise was wrong: a referee's halt does not make the fencers still, it
ends the phrase and sends them back to their guard lines, which is movement, not its absence.

Replacing it with post-event **separation**, the change in mean distance across the event,
measures the reset directly. It is also largely immune to panning, because a pan shifts both
fencers together and distance is a difference between them. Measured separation is +0.47 m median
after a real touch against -0.03 m for a false positive, and recall rose from 0.50 to 0.79.

The general lesson is that a feature can be implemented exactly as specified and still measure
nothing, and that only ground truth reveals which case one is in. The unit tests for the halt
feature passed throughout.

**The band-calibration metric discarded amplitude.** Candidate frequency bands were scored by the
kurtosis of their energy envelope, standardised per band. Standardising divides out amplitude, so
a band containing only faint spectral leakage against near-silence scored higher than the band
containing the actual tone. Weighting peakiness by absolute peak fixed it and improved F1 from
0.73 to 0.79 on its own.

#### Honest limits on these numbers

The operating point is tuned on a single clip with fourteen touches, so overfitting is a genuine
risk and a second labelled bout is needed before the settings can be called general. The
separation feature is on firmer ground than the thresholds around it, because it follows from how
fencing is refereed rather than from the fit.

The three missed touches are instructive. One falls at 179 s in a 180 s clip, so the window used
to measure separation runs past the end of the recording and the strongest feature cannot fire at
all. That is an edge effect of the method rather than a detection failure, and it would disappear
on footage that continues past the final touch.

Finally, the corrections metric flatters nothing but is still not a fair measure. It counts a
rejection and a manual addition as equally expensive. Rejecting a proposal is one click on an
event already located and timestamped; adding a missed touch requires scrubbing the recording to
find it, which is exactly the labour the system exists to remove. Recall is therefore worth more
than the raw count implies, and the six-corrections figure is a lower bound on the benefit rather
than a measurement of it. Establishing the real ratio requires the user study set out in the
design chapter, which has not been carried out.

### Touch detection: the audio approach failed to generalise, and was replaced

The figures above describe the audio-based detector evaluated on the clip it was tuned on. A
second bout was then labelled and deliberately held back, and the tuned operating point was
applied to it without retuning. It failed.

| | clip 3 (tuned on) | clip 2 (held back) |
|---|---|---|
| Precision | 0.79 | 0.21 |
| Recall | 0.79 | 1.00 |
| F1 | 0.79 | 0.35 |
| Corrections vs manual | 6 vs 14 | 15 vs 4 |

Fifteen corrections against a manual baseline of four means the detector produced nearly four
times more work than labelling by hand. Confidence also lost all ranking power: the four real
touches scored 0.74 to 0.79 while false positives reached 0.91, so filtering at 0.80 gave
precision 0.00.

#### Why audio could not be rescued

Three diagnostics established that this was not a badly chosen constant.

**The noise floor differs by an order of magnitude between recordings.** Clip 3 is a quiet club
hall: median band score 0.00007 against touches of 0.008 to 0.21, so touch-to-noise runs 114x to
3000x. Clip 2 is a broadcast with crowd and commentary: median 0.00079 against touches of 0.005 to
0.025, so touch-to-noise is 6x to 32x. Five threshold rules were tested (percentile, median
multiple, fraction of maximum, median plus MAD, and calibration from user-confirmed touches) and
each either floods the noisier clip or finds nothing in it.

**A percentile threshold flags a fixed fraction of frames rather than a number of events.** Five
per cent of 9,000 envelope frames is always about 450, merging to 40 or 50 candidates on any clip.
Against fourteen touches that ratio works; against four it cannot. Candidate count followed
recording length rather than how much happened, which is a structural defect rather than a tuning
error.

**No frequency band works, even with the answer in hand.** Searching every band and scoring each by
how well it separated the weakest real touch from strong background gave a best ratio below 1.0 on
both clips (0.45 and 0.92), meaning the weakest touch is quieter than ordinary background even in
the optimal band. Spectral tonality, which responds to tone rather than loudness and should suit a
buzzer better, was worse (0.21 and 0.55). Most tellingly, on clip 3 the loudest tonal component at
each touch sat at a different frequency every time. A scoring machine has one pitch. That scatter
indicates the buzzer is not reliably present in the recording at all, and that what the detector
had been finding was blade contact and exchange noise which happens to correlate with touches.

#### Geometry alone is better, and it transfers

Testing what the geometric features could do without audio produced a detector that scores the
same on both clips using identical settings:

| | audio and geometry | geometry alone |
|---|---|---|
| clip 3 (tuned on) | F1 0.79, 6 corrections | **F1 0.86, 4 corrections** |
| clip 2 (held back) | F1 0.35, 15 corrections | **F1 0.86, 1 correction** |

*Clip 2's geometry-alone figure was later re-measured at F1 0.80 with two
corrections, after a frame-rate dependency was corrected in the local-minimum
window. Recall rose from 0.75 to 1.00 in the same change, so the fall in F1 is a
shift from missing a touch to admitting two false positives rather than a
regression in what the detector finds.*

The audio was not merely unhelpful but actively harmful: it generated candidates that the geometry
then had to filter, and on the noisier recording it flooded the timeline. Removing it also removed
the ffmpeg dependency, the per-clip band calibration, and the threshold that would not transfer.

The geometric signature follows from how fencing works rather than from a fit. A touch requires
closing to scoring distance, and the referee's halt afterwards sends both fencers back to their
guard lines, so they separate. The detector looks for a prominent local minimum in inter-fencer
distance followed by sustained separation. Distance is also robust to camera panning, because a pan
shifts both fencers together and distance is a difference between them.

Candidate rate now tracks content instead of recording length: 0.7 per minute on clip 1, 1.0 on
clip 2, 4.7 on clip 3 and 4.4 on clip 4.

#### What this episode demonstrates

Three points worth carrying into the evaluation chapter.

First, **a single-clip evaluation establishes nothing.** The audio detector's F1 of 0.79 looked like
a working feature and was an artefact of the clip it was tuned on. Only a held-back bout revealed
that, and the cost of finding out was one labelling session.

Second, **the more sophisticated approach was the worse one.** Audio detection followed the
literature (Mo, 2022), required band calibration, spectral analysis and an external decoding
dependency, and was beaten by a local minimum in a signal the pipeline already produced. The
simpler method won because it rests on a physical property of the sport rather than on a property
of one recording.

Third, **a negative result was necessary to reach the positive one.** Geometry-only detection was
not tried until audio had been shown to fail, and the failure is what prompted asking what the
remaining features could do unaided.

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

**Editor and workflow.** Development was split between a terminal and an IDE for practical
rather than ideological reasons: writing and refactoring in a shell session with the test
suite running alongside, reading and spot-checking diffs in VS Code, and inspecting commit
history in GitHub Desktop when a visual view of the branch graph was quicker than terminal
git.

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
gives setup instructions. `RESULTS.md` indexes every output directory and records which
figures should be quoted from where, which matters because a dozen of them accumulated during
development. The development log (this section's later siblings) and the `TODO.md`
cross-references together replace what would otherwise be a separate design journal or wiki.

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

### Piste-region filtering

The evaluation of the third `FencerTracker` iteration identified wrong-target capture by
background people as the most damaging remaining failure mode. The design chapter had already
named piste-region detection as the most promising fix, on the reasoning that the piste is the
natural region of interest for the entire problem: every legitimate detection is a person
standing on it, and every referee, coach, audience member and adjacent-piste fencer is not.

The implemented filter is a polygon in pixel coordinates, supplied per clip as a small JSON
file and loaded into a `PisteRegion` object. For each detection, the bounding box's
bottom-centre is taken as a feet-position proxy and tested against the polygon with OpenCV's
`pointPolygonTest`; detections whose feet fall outside are discarded. The filter runs *before*
the `FencerTracker` matching stage rather than after it, which matters for a specific reason:
the tracker's spatial and size gates depend on slot history, so on the first frame of a clip
they have nothing to compare against and will accept whatever the detector returns. Filtering
first means a bystander cannot be admitted as the initial anchor for a slot and then defended
by the gates on every subsequent frame.

Two design decisions are worth recording. First, the polygon is authored by hand rather than
detected automatically. Automatic piste detection by Hough-line analysis or colour
segmentation was considered and deferred: the manual polygon takes a few minutes per clip,
is completely reliable, and is in any case the behaviour the full system needs, because the
planned annotation interface will have the user confirm or adjust the piste region on the
first frame as part of upload. Building the automatic detector first would have meant
building the harder version of a feature whose easier version the user interface requires
anyway. Second, an arbitrary polygon is supported rather than a rectangle, because a piste
viewed from a raised broadcast camera is a trapezoid in the image plane, not a rectangle.

The filter is a strict improvement where the piste is correctly described and a severe
regression where it is not, and the evaluation chapter reports both cases.

### Constant-velocity motion model

The second fix named in the design chapter was a motion model: track each slot's velocity
rather than only its position, and require candidate detections to be consistent with where
the slot is predicted to be, not merely with where it last was. The motivating case is a
referee who walks into the position a fencer occupied a moment ago. Under position-only
gating that candidate is an excellent match, because the gate asks only "is this near where
the slot last was". Under prediction-based gating it is a poor match, because the fencer was
moving and the referee is not where that motion leads.

The implementation keeps the two most recent committed centres for each slot and estimates
velocity by differencing them. The predicted position is the last centre plus that velocity.
Both the distance gate and the assignment-cost calculation use the predicted position as
their anchor in place of the last-known position, which means the model improves both
rejection (the gate) and allocation (which detection goes to which slot) from a single
change. The allocation effect is what addresses close-range identity flicker: when two
fencers cross and the detector briefly merges them, the assignment chosen on re-separation is
the one consistent with each slot's prior direction of travel.

Two guards proved necessary, and both were added only after the first version failed on real
footage (the failure is documented in the evaluation chapter). Velocity is estimated only
when the two commits it is derived from are within three frames of each other; differencing
two positions recorded further apart measures total displacement over the gap rather than
per-frame velocity, and extrapolating from it produces predictions far outside the frame.
Separately, a slot that has gone unmatched for more than thirty consecutive frames discards
its history entirely so that it can re-acquire a fencer, because a slot whose stale history
rejects every candidate can otherwise never recover: nothing is committed, so the history
that is causing the rejections is never replaced.

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

### LLM tactical summary (third pre-trained model, basic implementation)

The third pre-trained model in the design - a large language model that turns the collected
statistics into a written tactical summary - was implemented as a separate command-line stage,
`generate_summary.py`, decoupled from the video pipeline. It reads the per-frame metrics CSV
that `run_detection.py` produces, aggregates it into bout-level statistics (mean, minimum,
maximum and standard deviation of distance; time spent in each tactical distance band;
per-fencer cumulative push and pull with percentage shares and net displacement; tracker
coverage), embeds that payload as JSON in a structured prompt, and calls a hosted
large-language-model API through its Python SDK. The output is constrained by the prompt to a fixed markdown structure:
a bout-summary paragraph, observed-tendencies bullets each tied to a concrete number,
suggestions to explore, and a data-caveats paragraph.

Two design points are worth recording. First, the prompt carries explicit **honesty
constraints**: the model is told that distances are estimates, that totals include resets and
walkbacks because the system does not yet know when touches happen, and that it must never
invent touches, scores or events that are not in the data. This mirrors the project-wide
stance of acknowledging measurement limitations rather than glossing over them, and directly
addresses the known risk of LLMs fabricating plausible-sounding specifics. Second, outputs are
**cached**: a SHA-256 hash of the model ID and both prompts is stored in a sidecar metadata
file, and the API call is skipped when nothing has changed, so re-running the stage does not
re-bill.

This is deliberately a basic first implementation - the inspiration is the way consumer sports
platforms (Garmin Connect, Strava, Fitbit) generate narrative insights from sensor data with a
pre-trained LLM behind a structured prompt. The summaries are currently thin because the
prototype only produces distance and push/pull data; the pipeline is unchanged as richer
inputs arrive (confirmed touch events, in-play-only metrics, tempo, cross-bout profiles), so
the payload grows rather than the architecture. The stage is covered by 17 unit tests over the
statistics computation, prompt construction and cache behaviour, with the API call mocked, so
the test suite needs neither a network connection nor an API key.

---


### The application layer

The pipeline was reachable only from a shell until late in the project: no upload, no way to
start a run, and no way for anyone but the author to use the system. The design chapter
describes a web application, so this was the largest distance between what the design claimed
and what existed.

Each pipeline stage runs as a SUBPROCESS supervised by a single background worker, never as a
function call inside the API process. Three reasons, all specific to this pipeline. The heavy
work is native code, and YOLO, OpenCV and MediaPipe can fail in ways that end the interpreter
rather than raise something catchable; in a thread that would take the API and every queued job
with it, whereas a subprocess failure is an exit code recorded against one job. The models cost
hundreds of megabytes of resident memory, which a subprocess returns on exit. And the pipeline
stays a working command line, invoked by exactly the commands used to produce the figures in
this report, so there is no second code path that could disagree with them.

Jobs queue at a concurrency of one, because detection is CPU-bound and two concurrent runs take
appreciably more than twice as long as one while making the machine unusable. Progress is parsed
from a machine-readable line the pipeline prints, rather than through a callback, which would
have required importing the pipeline and undone the separation above. Every job is a JSON file
on disk, so a restart loses nothing: jobs caught mid-run are marked interrupted rather than left
claiming to be running, and are not retried automatically, since a video that crashes the
pipeline would otherwise crash it on every start.

Two decisions in the interface are worth recording. Summary generation costs a paid API call, so
it is never automatic; what the application layer changed is that the user presses a button
rather than being handed a command to type. And re-anchoring, the one annotation action that
changes tracking rather than interpretation, can only take effect on a reprocess, so the
interface now queues that reprocess and writes the result to a NEW bout, leaving the original
intact for comparison.

### Interface design decisions

The interface is a working instrument rather than a demonstration, and two decisions in it are
worth stating because both were reversals of what the first version did.

The first is that colour carries meaning and the chrome does not use it. The original palette
took the Fencer 1 blue as its general accent, with the result that the primary buttons, the
progress bars, the timeline markers for hand-added touches and one of the two fencers were all
rendered in the same hue. Since the two fencers are identified by colour in the timeline, the
radar, the scoreline and the table stripes, that made the single most information-carrying
signal in the interface indistinguishable from decoration. Structural elements are now neutral,
so a saturated pixel anywhere always identifies a fencer or a state. The same reasoning applied
to the explanatory notes, every one of which carried the amber warning rule and so made genuine
warnings indistinguishable from paragraphs explaining how a panel worked.

The third decision, and the one that changed the most, was to stop presenting everything at
once. The screen showed six bordered panels of equal visual weight, so nothing was more
important than anything else and the single act the interface exists for, deciding about the
proposal on screen, competed with five things the user was not doing. The working column now
holds only the bout selector, the video, the timeline, the decision loop and the record it
produces, with no borders inside it, since those parts are one continuous task rather than five
announcements. Everything else moved into a tabbed side column grouped by the question it
answers, with the corrective tools in particular separated out because they are used rarely and
never during a review pass. The explanatory notes are folded behind single-line disclosures
rather than deleted: nearly every caveat in this interface was paid for by a measurement that
went wrong, so none can be dropped, but printed simultaneously they produced a screen that was
mostly prose, and a caveat inside a wall of text is read no more carefully than one that is
absent.

The second is that the interface is pitched at a coach rather than at an operator. An
intermediate version set the whole application in a technical register: ten-pixel uppercase
labels, hairline rules, monospaced figures throughout and three-pixel corners. It was internally
consistent, and it read as laboratory equipment. Since measurements are read by comparing them
down a column, which proportional figures prevent, the alignment that motivated the monospaced
face is real; but `font-variant-numeric: tabular-nums` supplies it without the register, so the
monospaced face is now used only for file paths. Type sizes were raised throughout, corners
rounded, and the primary action filled rather than outlined, an outlined primary having read as a
disabled control.

A fourth pass addressed the densest parts of the result. The per-fencer measurements had been a
flat list in which every row carried an "F1" or "F2" prefix, so comparing the two fencers meant
reading alternate lines and matching the words after the prefix, six rows carrying three
measurements. Presented as a two-column table the comparison becomes the layout itself. The share
of the bout each fencer spent leading became a single split bar, on the same reasoning: it is a
share of a whole, and as two separate percentages the reader had to work out that the two shares
plus the level time account for the bout. The proposal list stopped being a six-column table, in
which every cell carried equal weight and fourteen proposals were fourteen identical bands of
small text, and became rows with internal hierarchy: the timestamp first because it identifies
the proposal, the state as a coloured marker because that is what the user is changing, and the
confidence and separation in smaller muted text because they are consulted rather than scanned.

This is recorded because it took four attempts and the first three failed in instructive ways. The
first changed only the palette and the typeface and left the layout untouched, which addressed
none of the problem. The second restructured correctly but dressed the result in the wrong
register. The third fixed the register and left the dense lists inside it unchanged. None of the
failures was visible in any test: the interface passed its full suite in all four states, because
what the tests assert is that the right information is reachable, and in every version it was.
Presentation is the one part of this system that its tests cannot judge.

### Testing

The system is supported by 506 automated tests: 320 over the pipeline, 163 over the application
layer and 23 over the interface, plus two end-to-end tests that drive the real pipeline on a
synthetic video. The end-to-end pair exists for what unit tests structurally cannot reach: the
stages are joined by filename conventions rather than by return values, and a bout identifier is
assembled in one module and taken apart in another. It found two defects on its first run, one
of which returned an unhandled server error for any bout the tracker never held both fencers in.

The pipeline tests cover:

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

### Clip 1 - engagement-oriented bout

- Frames processed: 10,791
- Frames with both fencers detected: 8,842 (82%)
- Pose-based distance samples: 2,842 (~32% of detected frames, consistent with the pose
  stride of 3)
- Mean inter-fencer distance: 2.26 m
- Distance range: 0.11 m to 6.60 m
- Cumulative motion: Fencer 1 - 14.2 m push / 11.1 m pull. Fencer 2 - 15.8 m push / 17.1 m pull.

### Clip 2 - high-activity bout

- Frames processed: 9,000 (50 fps source)
- Frames with both fencers detected: 7,813 (87%)
- Pose-based distance samples: 2,327 (~30%)
- Mean inter-fencer distance: 3.33 m
- Distance range: 0.36 m to 8.45 m
- Cumulative motion: Fencer 1 - 41.0 m push / 41.5 m pull. Fencer 2 - 36.8 m push / 37.1 m pull.

### Cross-clip comparison

The two clips show a striking difference in activity. Clip 2's fencers do roughly two to three
times the footwork of clip 1's and engage at a noticeably wider average distance. This is
exactly the kind of cross-bout comparison that is effectively impossible to perform by manual
observation, and so its reproducible production by the prototype is itself a meaningful
result: the prototype can already generate metrics that distinguish bouts at a level of detail
that manual analysis cannot match.

### Attributing a touch to a fencer

The detector reports when a touch happened and never who scored, so the scoring
fencer was the one field of the design's touch record that could only originate
with the user. `detect_scorer.py` proposes it by reading the piste-side scoring
machine's lamps.

Lamps rather than a score overlay, because a broadcast usually carries both but
club footage carries only the machine, and the club case is the project's target
user. Whole-frame saturated-colour counts rather than a region around the machine,
because club footage is hand-held and follows the action, so the machine drifts
across the frame and out of it; a fixed region is empty much of the time. A local
baseline taken two to three seconds earlier removes what is permanently red in
shot, which on these clips means an EXIT sign, sponsor banners and the piste.

It is never permitted to propose touches, only to attribute ones already found.
The lamps fire whenever the circuit closes, which includes fencers testing weapons
against the piste or each other's guards, routinely just after a touch and before
coming back on guard. Taking touch times as given removes that entire failure
class rather than attempting to filter it.

Evaluated leave-one-out against the scorer column already present in all four
ground-truth files, so it required no additional labelling:

| clip | three-way (left/right/double) | green lamp alone |
|---|---|---|
| 1 | 3/3 | 3/3 |
| 2 | 2/4 | 4/4 |
| 3 | 9/14 | 14/14 |
| 4 | 4/6 | 6/6 |
| **all** | **18/27 (67%)** | **27/27 (100%)** |

The green lamp is reliable and the red lamp is not, and nearly every three-way
error is a red-lamp error: red is contaminated by everything permanently red in
frame and no local baseline removes it entirely when the camera pans. The claim
this supports is therefore narrower than "the system says who scored". It is that
**the system always identifies whether one named fencer was involved**, which
fully determines the 11 touches of 27 where they were not and reduces the other 16
from a three-way decision to a two-way one.

This is classical colour thresholding and adds no fourth pre-trained model. The
colour-to-side mapping is confirmed once per bout by the user, since nothing in
the image indicates whether the green lamp belongs to the fencer on the left.

### Lunge detection, calibrated per bout

B1j established that the stance ratio is not view-invariant: an operating point
fitted on clip 3 does not survive clip 2, reaching F1 0.22 and 0.29 while firing
on a quarter of all windows. `detect_lunges.py` stops attempting to transfer it
and calibrates instead from the first five lunges the user has confirmed on the
bout being watched, which reuses the interactive-correction mechanism the design
already depends on rather than adding a demand on the user.

The magnitude of the transfer problem, stated directly: clip 3 calibrates to a
stance ratio of 1.902 and clip 2 to 2.706, a 42 per cent difference in what
constitutes a lunge-like posture.

Calibrating on the first five confirmed lunges and testing only on those that
follow, so no lunge informs its own proposal:

| case | test lunges | precision | recall | F1 | lift over chance |
|---|---|---|---|---|---|
| clip 3, Fencer 2 | 25 | 0.73 | 0.76 | **0.75** | 9.3x |
| clip 2, Fencer 2 | 2 | 0.50 | 1.00 | 0.67 | 54x |
| cross-clip threshold (B1j) | - | 0.12 | 1.00 | **0.22** | 6.4x |

Only the clip-3 row carries evidential weight; clip 2's test sets contain two and
four lunges. Two caveats attach to every figure here. Precision is a lower bound,
because a firing on a real but unlabelled lunge counts against it, and B1h
measured roughly 40 wide-stance episodes per minute against about 10 labelled
lunges. And the result rests on two clips and one labeller, so the claim is not
that lunge detection works, but that per-bout calibration works where transfer
does not.

### Measuring the effort the design claims to save

The project's central claim is that confirming the system's proposals costs less effort than
labelling a bout from scratch. Every chapter to this point argues it and none of them measures
it. A user study would measure it properly and is out of scope for the reasons given in section
9; what follows is the cheaper thing that was available, which is to build the measurement into
the tool and take it on a single reviewer.

Two additions make it possible. The first is a review queue: rather than presenting every
proposal in a table and leaving the user to find the next undecided one, the interface holds the
playhead at each proposal in turn and takes a single keystroke as the answer. This is not a new
capability so much as the removal of a cost the earlier interface imposed by accident. The table
is the right shape for checking work already done and the wrong shape for doing it, because
finding the next row, clicking it, and waiting for the seek is repeated once per proposal and on
a bout with a hundred lunge proposals it is most of the work. It is also precisely the cost the
design claims assisted annotation removes, so leaving it in place would have understated the
system in its own evaluation.

The second is a manual mode: the same interface with the detector's proposals withheld. It is
built to be the control condition rather than an alternative way of working. The proposals are
withheld entirely rather than dimmed or collapsed, because a suggestion that is visible has
already been read by the time the user decides to ignore it, and the condition being measured is
labelling without them.

Both modes time themselves, and the timing is stored with the bout's annotations rather than
displayed and discarded, since a figure that does not survive a page reload cannot be quoted.
Three decisions about what the figure counts are worth recording. Skipped items are excluded from
the denominator, because they are not decisions and counting them would improve the measured rate
in proportion to how many questions went unanswered. The rate is per decision rather than per
bout, because the two conditions do not produce the same number of decisions: assisted review
answers one question per proposal, while manual logging creates one entry per touch the user
finds. And a ratio computed from a single pass in each mode is labelled an illustration rather
than a measurement, in the stored record and again in the interface, because a number displayed
in a box is read as a result unless it is explicitly told not to be.

At the time of writing no such pass has been recorded by the author, so this section reports a
method and not a result. That is the honest position: the instrument exists, it has been
exercised end to end, and the figure it is designed to produce requires a person who can tell a
lunge from a wide stance to sit down with a bout.

### Characterising a fencer without accumulating error

The withdrawn push / pull totals are the closest this project came to answering the question
a coach actually asks, which is not how far someone moved but what kind of fencer they were.
Withdrawing them removed the only per-fencer characterisation the system offered. The question
this section answers is whether any such characterisation survives the measurement failure that
sank those totals, or whether the failure was fatal to the whole class.

It was not fatal, and the reason is specific rather than general. Appendix E.2 traces the defect
to accumulation: the totals summed per-frame position deltas, and a fencer re-acquired at a new
position after a tracking dropout contributed a one-sided step that never cancelled. Clip 3's
Fencer 2 accumulated +23.01 m of movement against an endpoint difference of -0.94 m, and no clamp
threshold repairs it because the truncated steps reach 5.4 m in a single frame. What failed was
the operation, not the underlying position series. An instantaneous reading averaged over
thousands of frames absorbs a dropout that displaces a handful of them; a running sum banks it
permanently and carries it to the end of the bout.

`fencer_profile.py` therefore builds a six-axis profile in which no axis accumulates anything.
Three come from the position series directly, as a mean or a quantile, and three are counts over
the touches the user has confirmed:

| Axis | Derived from | Clip 3, Fencer 1 | Clip 3, Fencer 2 |
|---|---|---|---|
| Territory, metres up the strip from own end | mean position | 2.40 | 3.19 |
| Ground used, interquartile range of position | quantiles of position | 0.89 | 1.66 |
| Scoring share, doubles counted half | confirmed touches | 58.3% | 41.7% |
| Scoring range, mean distance when they scored | distance at those touches | 1.67 m | 1.42 m |
| Lunges per minute | confirmed lunges | not measured | not measured |
| Best run, consecutive touches | confirmed touches | 2 | 1 |

Measured over the twelve ground-truth touches that the detector proposed, the two fencers separate
on a reading no single number in the earlier evaluation expressed: Fencer 2 occupies nearly twice
the ground of Fencer 1, 1.66 m of interquartile range against 0.89 m, while Fencer 1 scores at
the longer distance, 1.67 m against 1.42 m. Whether that describes the bout is a question for
someone who has watched it, and it is recorded as an outstanding verification rather than a
finding.

Three decisions in the axis definitions are worth stating, because each rejects an obvious
alternative. Ground used is the interquartile range and not the full range, because one dropout
frame at the far end of the piste would set a full-range figure entirely by itself, and that frame
is the least trustworthy in the bout. Territory is measured from each fencer's own end of the
observed extent rather than as a raw coordinate, or the right-hand fencer would appear dominant in
every bout ever recorded. And the axes are scored against each other rather than against a
population: eight fencer-bouts drawn from one club and one broadcast is not a population, and
normalising against it would manufacture a typical epeeist out of this project's own evaluation
set. The midpoint of each axis is parity between the two fencers in that bout, and the measured
value is displayed beside it.

The profile also introduces the first outright refusal in the system. Every axis is per-fencer, so
all six assume slot identity held, and on clip 4 it did not: fourteen side swaps, with Fencer 1 on
the left in only 26.9 per cent of frames. A profile computed there would describe the tracker
rather than either fencer, and would look exactly as convincing as one that did not. Elsewhere the
project's practice is to display a figure alongside its caveat, and that practice is sound for a
number in a table. It is not sound for a shape: a radar with a footnote is still read as a radar.
The interface therefore shows the swap count and the reason in place of the diagram.

One defect in this feature was found by looking at the interface rather than by any test. With no
touches confirmed, the "best run" axis reported zero for both fencers and drew at parity, which
states that the two are evenly matched rather than that nothing has been reviewed. Every axis was
individually correct and the picture as a whole said something false. Unmeasured axes now report
absence explicitly, the spoke is dashed to distinguish it from a genuine tie, and both behaviours
carry regression tests confirmed by reintroducing the defect. It is a smaller instance of the
pattern this project has hit repeatedly, and recorded in section 9 as its central methodological
finding: a component can satisfy every test written for it while the thing it produces is wrong.

### What closing share can actually resolve

Closing share is reported here as one of the two reliable movement figures, on the
reasoning that counting the direction of each movement is robust where summing
magnitudes is not. That reasoning is sound, and it says nothing about resolution.
Across the evaluation set all eight fencer-clip values fall between 48.3 and 52.8
per cent, a spread of 4.5 points around 50, which is also what a metric dominated
by noise would produce.

Frames are strongly autocorrelated, because a fencer's direction of travel is
committed for a few tenths of a second, so treating each frame as an independent
draw gives a standard error under one point and would make every one of these
figures look highly significant. Resampling contiguous blocks of a third of a
second instead, converted to frames from each clip's own sample spacing:

| Clip | Fencer | Closing share | 95% interval | Differs from 50%? |
|---|---|---|---|---|
| 1 | F1 | 51.0% | 48.3 to 54.0% | no |
| 1 | F2 | 48.3% | 45.8 to 50.8% | no |
| 2 | F1 | 50.1% | 47.2 to 53.0% | no |
| 2 | F2 | 50.0% | 47.4 to 52.7% | no |
| 3 | F1 | 52.4% | 49.5 to 55.3% | no |
| 3 | F2 | 52.0% | 49.4 to 54.7% | no |
| 4 | F1 | 50.4% | 47.0 to 53.9% | no |
| 4 | F2 | 52.8% | 49.4 to 56.0% | no |

**Not one of the eight is distinguishable from a coin flip.** A null result is only
readable alongside what the method could have detected, so the same machinery was
run against series with known biases: on three minutes of footage it detects a
60/40 tendency in 100 per cent of trials, a 55/45 tendency in 75 per cent, and a
52/48 tendency in 30 per cent, against a false-positive rate of 11 per cent where
no bias exists. The intervals above are therefore mildly optimistic, and a true
interval is somewhat wider.

Every measured value sits in the 48 to 53 band, which is exactly where the method
detects a real tendency about a third of the time. **So the measurements are
consistent with these fencers being genuinely balanced and equally consistent with
real tendencies too small for three minutes of footage to resolve, and this
evidence cannot separate those.** Closing share should be reported with its
interval, and a difference between two fencers should not be read from it unless
the intervals separate. On this footage none do.

This is the fifth measurement in this project to survive its own reasoning and
fail a test of what it can resolve, and the pattern is now specific enough to
state as a rule: a metric's derivation establishes what it means, never what it
can detect, and the second question needs its own measurement. `evaluate_closing_share.py`
reproduces both tables.

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

### The candidate cut: root cause of both tracking failure modes

The two failure modes described below were treated for most of this project as
inherent limits of single-camera tracking, and as the standing justification for
an AI-assisted rather than automatic design. Investigating why a correctly aimed
user correction still repaired nothing showed that they shared a single cause,
upstream of the correction and of the gates, in code that had never been
questioned because it looked obviously right.

`FencerTracker.select` narrowed each frame to the two most CONFIDENT detections
and discarded the rest, before any anchor, gate or user correction was consulted.
The assumption is that the two fencers are the two people the detector is most
sure about. On competition footage that is measurably false, because the referee
stands ON the strip and so survives the piste-region filter by construction.

Measured across clip 2's six-second bystander capture, sampling every twentieth
frame: three detections lie inside the piste region in EVERY sampled frame, and
the fencer a user would click is detected in every one of them. The confidence cut
nonetheless discards that fencer in 9 of the 16 samples, their rank oscillating
between first and third on margins of about 0.01. At frame 8590, where the
capture becomes visible, the three confidences are 0.899 for the referee, 0.896
for the far fencer and 0.886 for the near one: the correct answer is discarded by
0.010. The tracker was arbitrating between three people using a quantity whose
differences are noise.

This also explains why the re-anchor action could not repair the failure, which
had until then been read as a limitation of the action. The cause is sustained
over six seconds and the correction is momentary: with the correction reaching the
right detection, the slot is handed the correct fencer at 171.80 s and has lost
them again by 171.82 s, because the next frame's cut discards that detection once
more. Teleport counts are identical with and without the correction.

**The fix follows from the diagnosis.** Confidence answers whether a person is
present, which all three detections satisfy. It does not answer which two of them
are the fencers being tracked. Proximity to each slot's predicted position does,
and costs nothing: the predictions already exist for the gates. Selecting the two
nearest candidates instead of the two most confident gives, across all four clips
with the hand-authored piste configurations:

| Clip | Coverage before / after | Tracking mix-ups before / after | Touch F1 before / after |
|---|---|---|---|
| 1 | 92.9% / 93.5% | 2 / **0** | 0.80 / 0.80 |
| 2 | 98.0% / 98.0% | 15 / **0** | 0.86 / 0.86 |
| 3 | 97.4% / 97.4% | 0 / 0 | 0.86 / 0.86 |
| 4 | 73.7% / **79.8%** | 114 / **54** | 0.60 / 0.60 |

Mix-ups are single-frame position jumps exceeding 1.5 m, which is a direct
signature of a slot switching person rather than a proxy for it: no fencer crosses
metres of piste between consecutive frames. Coverage and mix-ups improve or hold
on every clip, and the headline 720p result is unchanged, since clips 1 to 3 still
require six corrections against twenty-one manual entries.

**Clip 4's touch F1 does not move, and an earlier version of this section
reported it rising from 0.67 to 0.77.** That figure was read at clip 4's own
best-scoring confidence threshold rather than at the operating point tuned on
clip 3, which is the protocol this report states and applies elsewhere. Choosing
a threshold by looking at the held-back clip is the tuning-on-test-data error the
audio detector's failure was diagnosed with, and quoting a benefit obtained that
way would have repeated it in the project's own favour. At the tuned operating
point clip 4 scores F1 0.60 both before and after: the change improves how
reliably the fencers are followed, not how well touches are found on that clip.

Worth stating rather than quietly correcting, because it is another instance of
the pattern this project keeps finding in itself, this time in a measurement of
its own improvement.

**A prediction the footage makes was used to corroborate it.** Play resets to the
guard lines after every touch, so each fencer should finish a bout within about a
metre of where they started, and section 5.5 flagged clip 2's Fencer 1 net
displacement of +3.86 m as not believable. Under the corrected selection it
measures -0.00 m, and clip 4's Fencer 1 moves from +0.42 m to -0.05 m. This was
not the target of the change and is stronger evidence than the coverage figures
for that reason: it is a constraint imposed by the sport rather than a metric
being optimised. Clip 4's Fencer 2 remains implausible at +7.08 m, so that failure
has a different cause and stays open.

**What this costs the argument, stated plainly.** Wrong-target capture was
presented as evidence for the assisted-annotation design: computer vision fails in
this way, therefore the user must be able to repair it. On three of four clips it
now does not occur at all. It was a defect in candidate selection, not a limit of
single-camera tracking, and the honest reading is that this project spent a long
time designing around a bug. The design argument survives, since clip 4 still
shows 54 mix-ups and the correction mechanism remains necessary, but it must rest
on the failures that remain rather than on one that turned out to be fixable.

### Slot identity, and why one clip reported seven metres of drift

Clip 4's Fencer 2 reported +7.08 m of net displacement, which is not possible: play resets to
the guard lines after every touch, so each fencer should finish within about a metre of where
they started. The report recorded the cause as unidentified, and camera panning had already
been tested and ruled out because panning moves the two fencers' figures in opposite
directions.

The cause is that slot identity did not hold. Two fencers do not cross on a piste; one stays on
the referee's left for the whole bout and the other on the right. A sign change in the
difference between the two tracked positions is therefore not a fencing event but the tracker
exchanging which fencer each slot follows. Counting those separates the evaluation set
completely:

| Clip | Side swaps | Slot 1 on the left | Closest measured separation | Fencer 2 net |
|---|---|---|---|---|
| 1 | 0 | 100% | 0.65 m | -0.28 m |
| 2 | 0 | 100% | 0.38 m | +0.16 m |
| 3 | 0 | 100% | 0.26 m | -0.43 m |
| 4 | **14** | **26.9%** | **0.09 m** | **+7.08 m** |

The three clips whose slots never swap all report plausible displacements; the one that swaps
fourteen times, at a median of 2.1 seconds apart, is the one reporting seven metres. A
first-to-last displacement assumes the slot followed one fencer throughout, and on clip 4 it did
not, so the figure measures the swaps.

This also identifies the mechanism rather than only the symptom. Clip 4's two fencers come
within 0.09 m in the measured position against 0.26 to 0.65 m elsewhere, which is the range at
which the matcher has nothing left to separate them by. That is the close-range identity flicker
already named as the second failure mode, now with a number attached and a consequence traced
through to a reported statistic.

**Applying the check to every earlier output settles two more questions.** Under the previous
candidate selection clip 2 swapped 9 times and reported the +3.86 m this report flagged as not
believable; under the current one it swaps 0 times and reports -0.00 m, so the candidate fix did
not merely coincide with a better number, it removed the swapping that produced the bad one. And
the two clip 4 experiments recorded in Appendix E as simply "worse", cropping to the piste and
masking the gallery, swap 57 and 64 times against a baseline of 14: tightening the frame around
the fencers puts them closer together in the measured space, which is exactly where the matcher
runs out of ways to separate them.

The masked variant also shows why the magnitude check was never sufficient. It reports net
displacements of +1.00 and -0.59 m, which pass the implausibility test comfortably, while its slots
exchange fencers 64 times. **A believable-looking number is not evidence that identity held**, and
the two checks catch different failures.

**The system now detects this rather than reporting the figure confidently.** A bout with any
side swap at all carries a warning saying that per-slot figures are not attributable to a
particular fencer, which is a stronger and more useful statement than observing that a number
looks large. The check costs one pass over two columns, fires on clip 4 and stays silent on the
other three, and needed no threshold: any swap is evidence, because fencers do not cross.

### Assignment ties were being decided by list order

Correcting the candidate cut exposed a second defect that had been invisible
because it hid behind the first.

With two detections and two slots the matcher compares a straight assignment
against a swapped one and takes the cheaper, breaking equality with `<=`. Because
the cost is a sum of distances, it ties whenever one slot's fencer is absent: that
slot contributes a large distance to both assignments and drowns the difference.
The tie then resolves by whichever order the candidate list happened to be in.

Three behaviours were resting on that, and all three flipped when the ordering
changed. The unit test demonstrating that a re-anchor recovers a captured slot
scored 565 px for both assignments. The test demonstrating that a distant
bystander cannot steal a slot scored 1455 px for both. And clip 2's real capture
sat on the same knife edge. **The test that supposedly demonstrated the project's
central correction mechanism had been passing on an arbitrary tie-break in a
synthetic scenario**, which is the sharpest instance in this project of a defect
that satisfies its specification while establishing nothing.

Ties now prefer the assignment containing the single best-explained pairing: a
5 px match beside a 1450 px one is a fencer correctly identified beside a slot
whose fencer has left, whereas 355 px beside 1100 px is two mediocre guesses, and
the gates then reject the unmatched half. A test asserts that presenting the same
detections in the opposite order produces the same assignment, which is the
property that was missing rather than any particular outcome.

### Observed failure modes

Prototype evaluation on two independently sourced FIE-level bout clips reproducibly revealed
two specific failure modes. Both are well-known data-association problems in single-camera
multi-object tracking and both are predicted in the design chapter as the reason the system
is AI-assisted rather than fully automatic. Their presence in the prototype is therefore not
a contradiction of the design but evidence supporting it.

*Note added 29 Aug 2026: the section above supersedes part of what follows. Both
failure modes below were traced to the candidate cut, and failure mode 1 no longer
occurs on three of the four clips. The descriptions are kept because they are what
the evidence supported at the time and because the diagnosis is only legible
against them.*

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

The fourth iteration added the piste-region filter and the motion model described in the
implementation chapter. These moved coverage to 93% on clip 1 and 96% on clip 2, and are the
subject of the before-and-after evaluation reported in the sections on those two fixes.

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

### Against the original aims

The project set out to orchestrate several pre-trained models into a system that helps a
fencer analyse a bout, on the argument that unreliable computer vision becomes useful when a
person can repair each class of error cheaply. Three models were orchestrated in a genuine
chain rather than demonstrated side by side: YOLOv8 detections bound the region MediaPipe Pose
works in, pose landmarks supply the ground reference the distance series is measured from,
that series drives touch proposal, the confirmed touches scope every aggregate, and those
aggregates are what the language model is given to interpret. Each stage consumes the previous
one, and a defect in an early stage propagates, which section 5.4 demonstrates rather than
assumes.

The measurable outcomes are strongest where the evidence is strongest. Touch detection reaches
F1 0.85 across the three 720p clips against hand-labelled ground truth, and confirming its
proposals costs six corrections where labelling from scratch would cost twenty-one. Tracking
holds both fencers in 93 to 98 per cent of frames on three of four clips. Attribution of a
touch to a fencer, which the detector could never supply, now identifies whether one named
fencer was involved in all twenty-seven labelled touches. These are the claims the project can
defend.

The central claim is not among them. That assisted review is faster than manual review has
never been measured, and the corrections figures that stand in for it weight a rejection and a
manual addition equally, which is known to be false. Section 5.9 states this rather than
softening it, and it remains the largest gap between what the design argues and what the
evidence supports.

### What the project learned about its own method

The most transferable outcome is methodological, and it recurred often enough to be treated as
a finding rather than a series of accidents. **Five separate defects passed their tests while
measuring the wrong thing.** A distance-band classifier used thresholds invented rather than
derived, and inverted a clip's headline reading once corrected. A movement metric accumulated
its own smoothing and reported twenty-four metres of travel on a fourteen-metre piste. A
resolution ablation varied a quantity the detector never sees, so its flat result was
guaranteed in advance. A piste polygon placed by eye raised coverage while quadrupling the
count of physically impossible readings. And the tracker chose between three people using
confidence margins of about 0.01, which is noise, for the whole life of the project.

Each was invisible to unit testing because each was implemented exactly as specified. What
exposed them was measurement against reality: a physical constraint the sport imposes, a
held-back clip, a metric chosen because it could only move one way. The practical rules the
project ended with are that a parameter should be derived from data rather than chosen, that a
single-clip result establishes nothing until a held-back clip agrees, and that a headline
metric improving is not evidence when a second metric can move the opposite way. Clip 4
demonstrated that last point twice: coverage rose while wrong-target capture also rose.

The corollary is uncomfortable and worth stating. Wrong-target capture was presented for most
of the project as evidence for the assisted-annotation design, on the reasoning that computer
vision fails this way and so a user must be able to repair it. It was a defect in candidate
selection, and on three of four clips it no longer occurs. The design argument survives on the
failures that remain, but a substantial amount of design effort went into accommodating a bug.

### Limitations

Four qualify everything above. There is **no user evaluation**, so the effort claim is
unevidenced. **Pose does not earn its place**: it improves the distance measurement, but its
headline application to lunge detection produced an operating point that did not survive a
change of camera, and only per-bout calibration made it usable at all. **The sample is four
clips and one labeller**, three of them competition footage, so figures are plausibly
optimistic for the club setting that is the target use case. And **there is no frame-level
tracking ground truth**, so coverage measures whether the tracker committed to a detection,
not whether it committed to the right person.

### Future work

The immediate work is evaluative rather than technical. A user study measuring review time
against a manual baseline would settle the project's central claim, and weighting corrections
by their true cost would repair a metric the report currently qualifies. Both need
participants rather than code.

Technically, three directions follow from measured failures rather than from ambition. An
appearance-based re-identification embedding is the remaining lever against wrong-target
capture, since a spatial filter cannot separate a referee standing on the piste from a fencer.
Clip 4's Fencer 2 still reports an implausible seven metres of net displacement, which the
candidate-selection fix did not explain and which therefore has a cause still unidentified.
And lunge detection rests on two clips and one labeller, so more labelled footage would
establish whether per-bout calibration generalises or merely fitted twice.

Beyond that, reading the score overlay with OCR would add a fourth pre-trained model and
recover the single-versus-double distinction the lamps cannot reliably provide, and GPU
processing would remove the roughly two-times-real-time cost that currently shapes what a user
will sit through. A task-level breakdown is maintained in `TODO.md`.

---


## References

*(Placeholder - the preliminary report's reference list will be carried over and extended
here in Harvard style.)*

---


## Appendices

### A. Source code overview

Three directories: the pipeline, the application that drives it, and the interface.

**`code/prototype/` - the pipeline.** Every stage is a standalone command-line program, which
the web layer invokes rather than reimplements, so there is no second code path that could
disagree with the figures reported here.

| module | what it does |
|---|---|
| `run_detection.py` | detection, tracking, pose, distance and the annotated render. The main pipeline |
| `derive_piste.py` | measures a piste region from footage, replacing a hand-authored polygon |
| `detect_touches.py` | proposes touches from the distance series |
| `detect_scorer.py` | attributes a touch to a fencer by reading the scoring lamps |
| `detect_lunges.py` | proposes lunges, calibrated on lunges the user has confirmed on that bout |
| `fencer_profile.py` | six per-fencer axes, the scoreline and piste zones, and the refusal to compute the axes where slot identity failed |
| `in_play.py` | scopes metrics to playing time using confirmed touches |
| `tempo.py` | exchange-rate and tempo measures |
| `generate_summary.py` | builds the statistics payload and calls the language model |
| `evaluate_touches.py`, `evaluate_lunges.py` | scoring against hand-labelled ground truth |

**`code/backend/` - the application.** FastAPI. `app.py` holds the endpoints, `jobs.py` the
background runner that processes an upload without blocking a request, `store.py` the annotation
store, and `storage.py` disk accounting and cleanup of derived files.

**`code/frontend/` - the interface.** React, built with Vite. Upload and job progress, the piste
confirmation step, and the review interface implementing the four annotation actions. The
earlier no-build-step interface is retained and served at `/legacy`, deliberately frozen at those
four actions: it guarantees that a checkout with Python alone still has a working review
interface, and it is the comparison between an interface with a build step and one without.

**Tests.** 367 in the pipeline, 184 in the backend and 97 in the interface, plus two end-to-end
tests that drive the real pipeline on a synthetic video to exercise the joins between stages,
which are made by filename convention rather than by return value. A GitHub Actions workflow runs
everything that needs neither footage nor model weights.

Dependencies are declared in `code/prototype/requirements.txt` and
`code/frontend/package.json`.

### B. CSV schema

The per-frame CSV produced by the prototype has columns: `frame`, `time_s`, `distance_raw_m`,
`distance_smooth_m`, `method` (`pose` or `bbox`), `f1_advance_m`, `f1_retreat_m`,
`f2_advance_m`, `f2_retreat_m`.

### C. Footage attribution

*(Placeholder - final URLs and per-clip details to be filled in here; the drop-in attribution
paragraph for the preliminary report is in `TODO.md` Part A4.)*

### E. Measurement investigations

Moved here from the draft report's appendix, where it could not stay: the draft is bound by a
strict 9,500 word limit across six chapters and the brief exempts tables and figures but not
appendix narrative. This document has no such limit. The draft states each finding with its
figure and carries the tables; the derivations and ruled-out alternatives are below.

**E.1 Why the cumulative push and pull totals are not a measurement.** Re-measuring one clip's
position series under median smoothing windows from 1 to 121 frames:

| Window (frames) | F1 path length | F2 path length | F1 net | F2 net |
|---|---|---|---|---|
| 1 | 161.2 m | 193.1 m | +3.43 m | +0.43 m |
| 5 | 142.1 m | 163.6 m | +3.43 m | +0.43 m |
| 31 | 100.3 m | 112.7 m | +3.43 m | +0.43 m |
| 121 | 32.7 m | 51.4 m | +3.43 m | +0.43 m |

Path length falls by a factor of five with no asymptote while net displacement is unchanged to the
centimetre. Path length sums the magnitude of every frame's change, so measurement noise adds to it
and never cancels; a difference between two positions lets noise cancel. Tightening the tracker's
identity gate would not help, and this was checked rather than assumed: the largest per-frame
position change is 0.229 bounding-box heights, with p50 at 0.010 and p99 at 0.088, so there are no
discrete jumps to remove and the tail is continuous bounding-box instability.

**E.2 The per-frame movement clamp accounts for the divergence exactly.** Replaying the recorded
position series through the accumulator's own logic:

| Fencer | Endpoint difference | Accumulated advance minus retreat | Unclosed gap | Discarded by the clamp |
|---|---|---|---|---|
| 1 | +4.71 m | +13.96 m | +9.26 m | -9.26 m |
| 2 | -0.94 m | +23.01 m | +23.95 m | -23.94 m |

Residue left unbanked is 0.015 m, so nothing else contributes. The clamp fires on 63 and 65 frames
of 5,246. The mechanism is tracking gaps rather than symmetric glitches: capped frames are nearly
balanced in count, 30 retreat-side against 35 advance-side for Fencer 2, yet net to -23.94 m,
because the 35 per cent of them within a smoothing window of a lost-tracking frame carry -25.14 m
while the remainder roughly cancel. A symmetric out-and-back glitch crosses the clamp twice in
opposite directions and cancels; a fencer re-acquired at a new position after a dropout is a
one-sided step that does not. No clamp threshold repairs this, because the truncated steps reach
5.4 m in a single frame.

The unit test asserting the endpoint identity could not have caught this. Its own comment records
that its steps were chosen to stay inside the clamp threshold, so it measures the sign handling and
steps around the cause.

**E.3 Camera motion was tested and refuted as the cause.** Pan bias displaces both fencers alike,
while "toward the opponent" points in opposite directions for them, so pan makes their net figures
move oppositely. Clip 3's were both positive. Compensation was implemented as Lucas-Kanade optical
flow with the tracked boxes masked out, validated by recovering a synthetic pan exactly and by
reducing a synthetic pan bias of +8.33 and -8.30 m to +0.03 and +0.03. On real footage it improved
three clips and made clip 3 worse, 21.88 to 37.34 m, and it is off by default.

| Clip | Pan (px) | Worst net before | After |
|---|---|---|---|
| 1 | 3,172 | 2.16 m | 0.93 m |
| 2 | 20 | 1.64 m | 1.64 m |
| 3 | 3,900 | 21.88 m | 37.34 m |
| 4 | low | 3.25 m | 2.36 m |

**E.4 Source resolution is invisible to the detector.** YOLO rescales its input so the longest side
is 640 px, which undoes any rescale of the source before the network sees anything.

| Clip | Source supplied | Network input | Fencer height in input | Two fencers detected |
|---|---|---|---|---|
| 4 | 320x180 | 640x360 | 103 px | 89.0% |
| 4 | 640x360 | 640x360 | 104 px | 89.0% |
| 4 | 1280x720 | 640x360 | 103 px | 90.0% |
| 3 | 320x180 | 640x360 | 133 px | 99.0% |
| 3 | 1280x720 | 640x360 | 134 px | 99.0% |

The resolution ablation of section 5.5 was therefore varying a quantity the model never receives,
which is why its F1 was flat. What differs between clips is the fraction of the frame a fencer
occupies.

**E.5 Removing the spectators from clip 4 makes it worse.** The boundary was derived from the
detection histogram rather than placed by eye: the gallery cluster's bottom edge sits at p99 =
128.8 px and the fencer cluster's top edge at p1 = 188.5 px, a 60 px gap, so a cut at y = 129
removes 52 per cent of all detections without touching any observed fencer box. Two mechanisms were
tried, masking the region black at unchanged frame size and cropping it away.

| Condition | Coverage | Pose | Tracking gaps | Best F1 | Corrections |
|---|---|---|---|---|---|
| baseline 640x360 | 73.7% | 11.0% | 248 | 0.67 | 4 (67% of manual) |
| masked 640x360 | 66.4% | 8.3% | 328 | 0.53 | 7 (117%) |
| cropped 640x231 | 65.9% | 9.1% | 370 | 0.59 | 7 (117%) |

Raw detection of two fencer-sized boxes falls from 85.3 to 80.0 and 79.3 per cent with median fencer
height unchanged at 101.9 px, so the loss is in the detector and not in the tracker, and not because
the fencers became smaller. Why a detector should benefit from context it is not being asked about is
not established here. At 117 per cent of manual effort the tool would be slower than labelling by
hand, so the intervention is rejected.

**E.8 The abandoned audio path.** Retained behind a flag with the negative result recorded, as a
documented dead end rather than deleted. Per-clip band calibration was necessary rather than tidy:
the buzzer sits at 1200-1700 Hz on clip 3, 2700-3200 Hz on clip 2 and 1700-2200 Hz on clip 4, so any
hardcoded frequency would have worked on at most one. A second defect was found in the calibration
metric itself: candidate bands were scored by the kurtosis of their envelope standardised per band,
and standardising divides out amplitude, so a band holding only faint spectral leakage outscored the
band containing the tone. Weighting peakiness by absolute peak raised F1 from 0.73 to 0.79 by itself.
Both fixes were real improvements to a component that was then removed, which is the more useful
lesson: a component can be correct, well tuned, and still worth deleting.

  | Configuration | Clip 3 (tuned on) | Clip 2 (held back) |
  |---|---|---|
  | audio plus geometry | F1 0.79, 6 corrections | F1 0.35, 15 corrections vs 4 manual |
  | geometry alone | F1 0.86, 4 corrections | F1 0.86, 1 correction |

**E.14 Work plan.** Referenced in section 3.6.

| Task | Periods | Status |
|---|---|---|
| Requirements and background research | P1 to P2 | complete |
| Project and environment setup | P1 | complete |
| Model selection and comparative testing | P2 to P3 | complete |
| AI processing pipeline (detection, pose, metrics) | P3 to P5 | complete |
| Language-generation stage | P5 | complete |
| Backend API and data storage | P4 to P6 | not started |
| Frontend: upload and review views | P5 to P7 | not started |
| Assisted annotation interface | P6 to P8 | not started |
| Statistics, profiling and dashboard | P7 to P8 | partial (metrics only) |
| Software and user testing | P7 to P9 | partial (unit tests only) |
| Refinement and iteration | P8 to P9 | ongoing |
| Evaluation and report write-up | P9 | in progress |
| Buffer / contingency | P9 | unused |

**E.13 Evaluation footage.** Referenced in section 5.1.

| Clip | Setting | Camera | Resolution | Others in frame |
|---|---|---|---|---|
| 1 | domestic competition | low angle, close, some pan | 720p | officials, adjacent-piste fencer |
| 2 | World Cup broadcast | elevated, fixed | 720p | referee, adjacent piste, spectators |
| 3 | club training | hand-held, panning throughout | 720p | none |
| 4 | junior team competition | elevated, wide, little pan | 360p | referee in foreground, spectators |

**E.11 Movement across the four clips**, from one version of the pipeline, each clip with the piste
configuration it requires. Reported in section 5.3.

| Clip | Coverage | F1 net | F1 closing | F2 net | F2 closing |
|---|---|---|---|---|---|
| 1 | 93.5% | +1.29 m | 51.0% | +0.28 m | 48.3% |
| 2 | 98.0% | -0.00 m | 50.1% | -0.16 m | 50.0% |
| 3 | 97.4% | +3.43 m | 52.4% | +0.43 m | 52.0% |
| 4 | 79.8% | -0.05 m | 50.4% | **+7.08 m** | 52.8% |

Regenerated 29 Aug 2026 after the candidate-selection change described in 5.x. The previous
figures were 92.9% / +3.86 m / 50.5% (clip 2 F1) and 73.7% / +0.42 m / +7.54 m (clip 4).

**The change is independently corroborated by the plausibility check this table was built to
apply.** Play resets to the guard lines after every touch, so each fencer should finish within
about a metre of where they started. Clip 2's Fencer 1 previously measured +3.86 m, which the
report flagged as not believable; it now measures -0.00 m, which is exactly what the physical
constraint predicts. Clip 4's Fencer 1 moves from +0.42 m to -0.05 m on the same reasoning. This
was not the change's target, and it is stronger evidence than the coverage figures precisely
because it is a prediction the footage itself makes rather than a metric being optimised.

Clip 4's Fencer 2 remains implausible at +7.08 m, down from +7.54 m, so it is not explained by
the candidate-selection defect. It was subsequently explained by a different one, below.

**E.12 The two unsolved tracking failures.** *Wrong-target capture* is prevented by the piste polygon,
which by construction cannot distinguish a referee standing on the piste from a fencer. On clip 4 the
referee passes the polygon and survives to the slot competition, where he is rejected only by the size
gate and confidence ordering, clearing by 7 pixels against clip 2's 150. The zero implausible-sample
result on three clips therefore shows that the filter works where the piste boundary happens to
separate fencers from bystanders, not that the problem is solved. *Close-range identity flicker*
arises because YOLOv8 frequently returns a single merged box when fencers clinch, so one slot receives
no update for the duration; on separation the assignment is decided by the motion model, which prefers
the detection consistent with prior direction of travel and has no signal at all where both fencers
reversed direction inside the clinch. Appearance-based re-identification addresses both and is the
main proposal in Chapter 6.

**E.9 Three tracking defects found only on footage.** Each passed the unit suite. The *noise floor*
was intended to suppress bounding-box jitter and discarded any movement below a threshold; in fencing
an attack is explosive and clears it every frame while the recovery and walk-back are slow and clear it
on none, so discarding small movements discarded retreats and injected false advance. Measured on
synthetic input with a fencer returning to its exact starting position, the discarding version
reported +5.25 m of net advance with pull recorded as 0.00 m. It was replaced by banking sub-threshold
movement until it accumulates past the threshold. The *velocity model* estimated velocity from the last
two committed positions without checking how far apart in time they were; differencing two positions
recorded many frames apart measures displacement over the gap rather than velocity, and extrapolating
from it threw predictions outside the frame, taking clip 2's coverage from 87 to 13 per cent. It was
fixed by refusing to extrapolate across gaps longer than three frames. The *piste polygon* was placed
by eye from a single frame and admitted fencers on the adjacent piste, which raised coverage while
making the implausible-sample count five times worse: a headline number improving as the underlying
measurement degraded.

**E.10 The halt feature.** The design predicted that fencers stop after a touch, so the first version
of the feature tested whether combined movement fell to near zero afterwards. Against ground truth it
separated true from false positives by a factor of 1.04. Two causes compounded. The movement totals it
consumed are corrupted (D.1, D.2), so a defect in one metric disabled a feature two stages later. And
the premise was wrong independently of that: a referee's halt does not make fencers still, it ends the
phrase and sends them back to their guard lines, which is movement. The replacement measures
post-event separation directly, at +0.47 m median after a real touch against -0.03 m for a false
positive, and moved recall from 0.50 to 0.79.

**E.7 Tracking performance before and after the piste filter and motion model.** Reported in
section 5.3. The push and pull rows are retained to show what the fix appeared to do at the time,
not as measurements; D.1 and D.2 establish that they are not measurements at all.

| Measure | Clip 1 before | Clip 1 after | Clip 2 before | Clip 2 after |
|---|---|---|---|---|
| Coverage | 82.0% | **93.0%** | 86.8% | **98.0%** |
| Mean distance (m) | 2.26 | 2.27 | 3.33 | 3.34 |
| Max distance (m) | 6.60 | 6.44 | 8.45 | **6.82** |
| Implausible samples (>7 m) | 0 | 0 | 53 | **0** |
| Fencer 1 push / pull (m), withdrawn | 14.2 / 11.1 | 11.8 / 10.4 | 41.0 / 41.5 | 38.2 / 36.6 |
| Fencer 2 push / pull (m), withdrawn | 15.8 / 17.1 | 21.8 / 19.6 | 36.8 / 37.1 | 47.6 / 48.9 |

**E.6 Lunge detection from pose stance features.** Against 36 hand-labelled lunge peaks on clip 3,
using the dimensionless ratio of ankle separation to hip height, for the fencer with 30 of the
labels:

| Window | Past the random-window p90 | Fires per minute | Lunges per minute | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| -0.10/+0.10s | 24 of 30 | 29.3 | 10.0 | 0.80 | 0.27 | 0.41 |
| -0.20/+0.20s | 24 of 30 | 19.3 | 10.0 | 0.80 | 0.41 | 0.55 |
| -0.40/+0.20s | 18 of 30 | 12.3 | 10.0 | 0.60 | 0.49 | 0.54 |

The ratio is what carries the signal, not either feature alone: raw stance separation reaches 43 per
cent and hip drop 57, and combining them reaches 80. The ratio is also what removes a per-fencer
scale error of about 40 per cent, since a clip-wide scale cannot track how far each fencer is from
the camera; the ratio reads 1.05 against 1.02 for the two fencers where the metre figures differ by
40 per cent. Excluding the three labels the labeller flagged as doubtful raises the figure to 82 to
86 per cent rather than lowering it.

An earlier version of this test used awarded touches as a proxy for lunges and found nothing usable,
43 per cent enrichment at nine times the event rate. Most lunges miss and some touches are not
lunges, so the proxy was weak in both directions, and the null result was a property of the proxy
rather than of pose.

### D. Development log highlights

The full development trail is in the project's git history. Significant milestones include:
initial bounding-box-centre distance prototype; switch to pose-based front-foot distance with
MediaPipe; introduction of `FencerTracker` for stable identity; the regression caused by
overly strict ID locking and its reversion to spatial continuity; the inflated push / pull
values and the addition of rolling-median smoothing plus motion clamping; the addition of
spatial and size gates against bystanders and the subsequent re-calibration; the move of the
HUD from corners to a translucent bottom panel for legibility; the evaluation runs on two
independent test clips.
