# An AI-Assisted Web Application for Epee Fencing Bout Analysis and Fencer Profiling

## Abstract

Post-bout video review in fencing is manual across the whole sport. This project develops an
AI-assisted web application that chains three pre-trained models, YOLOv8 detection with
ByteTrack, MediaPipe Pose and a large language model, into a pipeline that turns an uploaded
epee bout into a distance series, proposed touches, per-fencer measurements and a written
summary, which a user then confirms or corrects through a browser interface. Touch detection
reaches F1 0.73 across six labelled clips against hand-labelled ground truth, 0.86 on the three
it was tuned on, and reading the scoring machine's lamps identifies whether one named fencer was
involved in 48 of 49 labelled touches. The project's most transferable outcome is methodological: five separate metrics
passed every test written for them while measuring something other than what they claimed, and
each was exposed by measurement against an external constraint rather than by further
reasoning.

---

## 1. Introduction (985/1000 words)

### Context

Video review is routine in sports performance analysis, but in fencing the review itself has
remained manual across the whole sport rather than only at its amateur end. Coaches at elite,
FIE and Olympic level still scrub footage by hand and tag actions themselves; no automated
pipeline is in widespread use anywhere in the sport. What differs by level is the resources
brought to the same manual task, not the task. At club level the consequence is that recorded
bouts often sit unreviewed altogether, or are reviewed by watching whole clips repeatedly for
qualitative impressions, and even dedicated fencing analysis tools require the user to label
every action by hand.

The addressable need is therefore broader than accessibility for amateurs, although amateur
accessibility is this project's primary user focus: a club fencer has no analyst and no budget,
so they are the user for whom automation makes the difference between analysis happening and
not happening.

### Motivation

Several tactically valuable quantities are measurable by hand only live, at the cost of a
coach's full attention, and then only as coarse discrete events. Inter-fencer distance over time,
engagement distance and its variance, and the frequency of advances, retreats and lunges cannot
be recorded continuously by eye at all. This layer of quantitative insight is consequently almost
never captured, and review remains limited to subjective impressions.

### Problem statement and aims

The problem is to design and build a working system that ingests an uploaded epee bout video;
uses pre-trained models to extract fencer positions, poses, derived movement metrics and likely
event segments; presents that information through an annotation interface allowing
confirmation, correction and labelling; and produces aggregated outputs that are useful to
fencers and coaches.

A central design principle is that the system is AI-assisted, not automatic. The unreliability
of computer vision on real fencing footage is treated as a constraint the workflow accommodates
rather than a problem to be solved before the system can be useful.

### What this adds that existing tools do not

Stated plainly, because the obvious reading of a project like this is that it automates work
other systems already do, and that is not the claim.

Existing fencing tools are **manual loggers**. The user watches the bout and tags what they saw;
the analytics that follow are summaries of those tags. What such a system knows is exactly what
was typed into it. This project's measurement layer knows things nobody entered: the distance
between the fencers at every frame of the recording, and every quantity derived from it.

That difference is not one of convenience. **Continuous inter-fencer distance is not merely
absent from manual tools; it is impractical to produce by hand at all**, because recording it for
one bout means pausing and measuring across thousands of frames. A coach can log who scored and
roughly where they were standing. No coach can log the distance thirty times a second, and so no
manual system, however well designed, can report how a fencer's working distance changed under
pressure or where on the strip their touches were scored from.

The event layer is the opposite case, and the report does not overclaim it. Deciding that a touch
occurred, and to whom, is a judgement a coach makes accurately and instantly, and the system's
automated attempts at it are worse. Those are proposals precisely because the human is better at
them. **The contribution is therefore a division of labour rather than a replacement**:
measurement that cannot be done by hand, paired with judgement that should not be taken from the
user.

### Contributions

**A working end-to-end system.** A bout is uploaded through a browser, processed by a
background job runner and reviewed through an annotation interface, with no terminal involved.
Three pre-trained models are chained rather than demonstrated separately: detections bound the
region pose works in, pose landmarks supply the ground reference the distance series is
measured from, that series drives touch proposal, confirmed touches scope every aggregate, and
those aggregates are what a language model interprets.

**A geometric touch detector evaluated against hand-labelled ground truth.** Forty-nine hits
were labelled across five independently sourced recordings, with clips held back from tuning so
generalisation could be tested rather than assumed. An audio detector was built first, shown
to fail on held-back footage, and replaced.

**Two capabilities the detector could not supply.** Attribution of a touch to a fencer, read
from the scoring machine's lamps, and lunge proposals calibrated per bout from lunges the user
has already confirmed. Both are reported with the limits measurement established.

**A methodological result about measuring one's own system.** Five metrics survived their own
derivation and then failed a test of what they could actually detect. Chapter 5 treats that
pattern as a finding, because the diagnosis in each case came from measurement against an
external constraint rather than from further reasoning.

### Scope

Out of scope by choice: refereeing decisions and right of way, which epee does not require;
identification of individual athletes, which the system avoids by tracking anonymous slots; and
multi-camera or three-dimensional reconstruction, which would resolve several measurement
limitations and is treated as further work.

This project follows **CM3020 Artificial Intelligence, Project Idea 1, "Orchestrating AI models
to achieve a goal"**, which expects an integrated system built around at least three pre-trained
models.

**Source code:** https://github.com/minh-bitrise/epee-fencing-analysis

### Structure of this report

Chapter 2 reviews the literature and positions the project against existing systems. Chapter 3
gives the design, including the argument that settles the system's role as an assistant
rather than an automaton. Chapter 4 is written as a development record rather than a description of a
finished artefact, because the changes of direction carry most of the evidence. Chapter 5
reports results on real footage and, equally, the results that did not survive scrutiny.
Chapter 6 judges the outcome against the aims above and states what the project failed to
establish.

---

## 2. Literature Review (1621/2500 words)

This chapter reviews work across four themes relevant to the application: deep learning in
sports video analysis; temporal action detection in untrimmed video; pose estimation for
fine-grained sports actions; and the human-in-the-loop paradigm underpinning the assisted
design. It then examines existing systems in order to position the project. Where the
implementation has since tested a claim from this literature against real footage, the outcome
is noted, because a review that informed a build should be readable alongside what the build
found.

### 2.1 Deep learning in sports video analysis

Rangasamy et al. (2020) contrast handcrafted-feature approaches with deep learning for human
activity recognition in sport, establishing that convolutional networks dominate frame-level
understanding, with recurrent and temporal models capturing motion across sequences. The value
here is methodological: it confirms that deep learning is the appropriate family of techniques
for extracting structure from sports footage. The review nonetheless treats analysis as a fully
automatic classification problem and is necessarily broad rather than fencing-specific. It does
not address how such systems should behave where reliable automation remains out of reach,
which is precisely this project's situation.

### 2.2 Temporal action detection in untrimmed video

Vahdani and Tian (2021) survey deep-learning approaches to localising when actions occur within
long untrimmed footage. This bears directly on the system, because a bout is a continuous
recording in which touches are sparse and separated by footwork, preparation and pauses. Their
taxonomy of supervision levels is instructive: the fully supervised methods that dominate
published benchmarks depend on dense frame-level boundary annotations that are expensive to
produce and that do not exist for any fencing dataset. The survey is explicit that temporal
action detection remains an open research problem rather than a solved capability.

Two implications follow. Automatic event localisation should highlight probable segments for
human review rather than serve as a reliable end in itself; and the absence of annotated
fencing data makes a fully supervised automatic system infeasible within an undergraduate
project. The realistic contribution is to reduce manual review time through assisted
highlighting.

### 2.3 Pose estimation for fine-grained sports actions

Hong et al. (2021) introduce Video Pose Distillation and demonstrate that pose is a strong cue
for fine-grained sports action recognition. Equally importantly, they show that off-the-shelf
pose estimators degrade in sports footage through motion blur, occlusion and domain shift. This
applies acutely to fencing, where two athletes overlap on a narrow piste and move explosively.
Their conclusion that pose is informative yet unreliable supports a design in which pose-derived
features inform suggestions a user confirms.

The implementation bears this out. Pose succeeded on 27 to 32 per cent of frames where both
fencers were tracked on eight of the nine clips, a figure governed mainly by a deliberate stride
of three frames, and on 12 per cent of clip 4, the clip in which the fencers stand smallest in
the frame. It failed most often during precisely the clinches and rapid exchanges Hong et al.
identify.
The pipeline falls back to bounding-box geometry on those frames rather than treating a missing
pose as missing data. Their distillation method is, however, a model-training contribution
aimed at improving accuracy on curated datasets; it does not deliver an end-user tool, and
reproducing its training pipeline lies beyond this scope. This project therefore adopts pose as
a pragmatic pre-trained component rather than attempting to advance pose recognition itself.

### 2.4 Fencing-specific computer vision

The closest domain-specific work is Mo (2022), whose "Allez Go" applies pose estimation and a
temporal convolutional network, augmented with audio, to referee bouts. It reports approximately
89 to 90 per cent accuracy in classifying which fencer scored, on a custom dataset of around
4,000 international-level clips, with audio detecting blade contact. This is evidence that
pose-based fencing analysis is feasible and that audio can complement visual features.

Two contrasts matter. Allez Go pursues automatic refereeing, a high-stakes binary decision,
whereas this system pursues assisted profiling, a lower-stakes task that tolerates imperfect
automation far better. And Allez Go trains on elite, well-filmed competition footage. It would
be wrong to infer that manual review is an amateur problem: it remains manual at FIE and
Olympic level too. What elite footage provides is better conditions for automation, namely
fixed camera positions, consistent framing and professional lighting. Club recordings are
single-camera, less standardised and noisier, weakening the case for full automation further.
Allez Go therefore simultaneously demonstrates feasibility and reinforces the rationale for an
assisted design, with the assisted approach mattering most where footage quality is lowest.

### 2.5 Human-in-the-loop machine learning

The assisted-annotation workflow is not merely a pragmatic compromise but an established
methodology. Mosqueira-Rey et al. (2023) survey human-in-the-loop machine learning, in which
humans and models collaborate so that human input corrects and guides model output. This
provides theoretical legitimacy for a system whose suggestions the user reviews, reframing
manual annotation from a weakness into a deliberate strategy that improves reliability and
yields verified data. Of the three paradigms they distinguish, the system most closely
resembles interactive machine learning, in which the user iteratively refines model outputs.

This grounding also shapes evaluation. The appropriate criteria are human-in-the-loop criteria:
reduction in manual effort relative to fully manual review, the quality of the resulting
annotations, and the number of interactions needed to repair a given class of error. That last
criterion is what makes the tracking failures reported in Chapter 5 tolerable rather than
fatal, since each is designed to be correctable by a single high-level action rather than
frame-by-frame editing.

### 2.6 Existing systems

Three commercial tools illustrate the practical landscape. Athlete Analyzer offers
fencing-specific video review in which the user uploads footage and tags actions by hand,
producing hit-zone and pattern statistics from those tags. Its domain alignment confirms demand
for structured fencing analysis, but the labelling is entirely manual and the analytics are
summaries of what the user entered rather than measurements taken from the video. Dartfish, a
mature general-purpose platform, demonstrates the proven value of video analysis across many
sports, but is not fencing-specific, depends heavily on manual operation, and assumes analyst
expertise club-level users frequently lack. AI FencingMeter is the closest commercial system to
this project's stance, applying automated analysis to smartphone footage, but it reports
blade-speed and reaction-time measures for a single fencer's technique rather than the
inter-fencer geometry and event structure that this project targets, and its methods are not
published, so its claims cannot be assessed.

Together these define the gap this project addresses: fencing-specific tools whose analytics
are only as good as what the user typed in, a general platform without fencing specificity, and
an automated tool measuring a different quantity by undisclosed means. None computes continuous
inter-fencer distance, which is not merely absent from them but impractical to produce by hand
at all. That is the substantive argument for automating the measurement layer even when event
interpretation stays with the user.

### 2.7 Synthesis

**An apparent contradiction, and its resolution.** Vahdani and Tian hold that temporal action
detection is unsolved on realistic footage, while Mo reports 89 to 90 per cent accuracy on a
harder fencing task than this project attempts. Taken at face value these conflict. They do not,
and the reconciliation determines this project's design. Mo's conditions are precisely those the
survey identifies as what supervised methods require: roughly 4,000 clips of international
competition, captured under fixed cameras with consistent framing. The survey's pessimism is
about footage without those properties. Mo therefore does not refute the survey; he satisfies its
preconditions, and in doing so demonstrates that the binding constraint is data and capture
conditions rather than technique.

**A dependency running through three of the four works.** Mo's system rests on pose, and Hong et
al. establish that pose estimators degrade under motion blur, occlusion and domain shift, which
are exactly the conditions elite capture minimises and club capture does not. Mo's result is
therefore conditioned on the circumstances that suppress the failure Hong et al. document. The
same architecture on single-camera club footage inherits a weaker signal, and the literature
offers no evidence about how much weaker. This project's own measurements supply a little:
pose succeeded on roughly a third of tracked frames, failing most in the clinches Hong et al.
name.

**Where that leaves the design.** If automation in this domain is conditioned on data and capture
this project does not have, the useful question is not how to approach Mo's accuracy but how to
be useful without his conditions. Mosqueira-Rey et al. answer it by making human correction a
designed component rather than a fallback, and by supplying the evaluation criteria that follow,
namely effort saved and interactions needed to repair an error rather than accuracy alone. The
four works therefore form an argument rather than a list: the method is available, its
performance is conditioned on data quality, the conditions do not hold here, and a
human-in-the-loop design is the response.

**Two critical reservations.** First, the fencing-specific literature is thin, resting
substantially on one system, so several arguments here transfer from adjacent sports rather than
resting on fencing evidence. Second, and more sharply, following the literature was itself a
source of error. Mo's validation of audio as a cue was adopted and the resulting detector failed
on held-back footage, because the finding was conditioned on capture quality that the paper had
no reason to foreground. **Background work supplies methods together with the circumstances that
produced them, and a review reporting the method without the circumstances invites exactly that
mistake.** Chapter 5 records the cost, which was one labelling session, and the practice adopted
in response, which is that no claim from the literature is carried into the design without being
tested on held-back footage of the kind the system will actually meet.

---

## 3. Design (1960/2000 words)

### Users and domain

The target user is a club-level epee fencer or coach with recorded bouts, no analyst and no
budget. Every automated output is a proposal, and the user's time is the scarce resource.

### Architecture

![**Figure 3.1: System architecture.** Four layers and what passes between them. The dashed
return path is the only structural element rather than a descriptive one, and it is what makes
the workflow human-in-the-loop: confirmed corrections are written back and re-scope every
aggregate, so the statistics describe the verified record rather than the system's first
guess.](figures/fig_architecture_components.png)

| Layer | Contains |
|---|---|
| Client | upload, piste confirmation, frame-accurate player and timeline, the review loop, results |
| Application and API | upload handling, job dispatch and status polling, the annotation store, analytics endpoints |
| Processing pipeline | ingestion; detection and tracking; pose; feature extraction; touch proposal; attribution and profiling; language generation |
| Data | uploaded video, per-frame CSV, job records, annotations |

Two properties matter more than the layering. **Every stage's output is persisted as editable
data rather than baked into the rendered video**, which lets the interface read it back. And
**the return path is a rescoping rather than a refresh**: confirmed touches bound each exchange,
so a correction changes which frames contribute to every downstream aggregate, not merely which
events are listed.

Pipeline stages are subprocesses rather than imports. The heavy work is native code, and YOLO,
OpenCV and MediaPipe can fail in ways that end the interpreter rather than raise something
catchable; in a thread that takes the API and every queued job with it, whereas a subprocess
failure is an exit code against one job. The models cost hundreds of megabytes of resident
memory, which a subprocess returns on exit. And the pipeline stays a working command line,
invoked by the commands that produced the figures here, so no second code path can disagree.

### Technology selection

**YOLOv8 with ByteTrack** for detection and tracking, mature, pre-trained on a generic person
class and bundled with a stable tracker; the `nano` variant runs on a consumer CPU and a larger
one substitutes without code changes. **MediaPipe Pose** (`pose_landmarker_lite`), run on each
fencer's crop so the model is given an easier problem constrained to the right person. **A hosted
large language model** for the written summary, the alternative being a template system that
could not adapt as the payload grows.

All three are used pre-trained. No model is trained or fine-tuned: no annotated fencing dataset
exists at the scale training requires, and building one is a larger project than this. The
contribution is the orchestration, not model development.

### The assisted-annotation workflow

Four user actions carry the design, each operating at the level a user already thinks about a
bout and repairing many frames of derived data in one interaction: **confirm or correct a
proposed touch**, accepting, retiming, attributing or rejecting it; **add a touch the system
missed**; **mark a segment tracking-unreliable**, excluding it from aggregates while the raw data
is kept; and **re-anchor a slot** by clicking the correct fencer.

The first carries structural weight beyond the touch record. Confirmed touches bound each
exchange, so resets and walkbacks are excluded from every aggregate, and those are precisely the
periods in which the referee enters frame. One domain-level act therefore removes most
bystander-contaminated data without the user thinking about tracking at all.

### Touch detection: why one signal cannot work

The obvious signal is the scoring machine's buzzer, and Mo (2022) validates audio as a fencing
cue. Audio alone is insufficient for reasons specific to how fencing is filmed: several bouts run
within earshot at a competition and the microphone cannot tell which machine fired; fencers test
blades against the floor, which registers identically; and parries produce metallic transients
resembling a short buzzer. Each candidate is therefore scored on several features, each included
because it discriminates a specific false positive.

Because the design only needs to *propose* candidates, recall matters more than precision, a
materially easier target than Mo (2022), whose system decides which fencer scored. Double touches
are an attribution problem rather than a detection one: both fencers score within the lockout
interval, the event is detected identically, and assignment is already the user's
responsibility.

### A capability hierarchy, and the annulment argument

Available signals differ by footage type, so the design degrades in tiers rather than failing.
Tier 1 is a score change, from a broadcast overlay or venue machine, giving timing and
attribution. Tier 2 is a buzzer with geometric corroboration. Tier 3 is geometry alone, for club
footage with no visible machine and no usable audio. Reporting per tier is more honest than a
single number averaging over categorically different information.

Tier 1 is not merely the most convenient signal but the only one reflecting the referee's
decision. A scoring machine registers a valid electrical contact; a score display registers an
awarded point. These differ whenever a referee annuls a touch, for corps-a-corps, for covering
target, or any other non-valid action.

This justifies the assisted design more strongly than the literature does. That argument rests on
models being unreliable, which invites the reply that better models would remove the need for the
user. The annulment case is not of that kind. When a referee annuls a touch, the hit occurred,
the machine fired, and no point was awarded: **the information distinguishing those outcomes is
absent from the recording entirely.** It is a refereeing judgement, not a physical event, and no
model can recover it. The detector therefore cannot be correct in principle, only useful.

### Piste regions are measured, not drawn

Detections whose feet fall outside the piste are discarded before the identity matcher runs.
This matters because the matcher's gates depend on slot history, so on the first frame they
accept whatever the detector returns; filtering first means a bystander cannot become a slot's
initial anchor and then be defended by the gates thereafter.

The region is derived by sampling frames and clustering where feet actually fall, then shown for
confirmation, rather than drawn by hand: a polygon placed by eye raised headline coverage while
quadrupling physically impossible distance readings. The user is still asked, because the
measurement assumes the two people closest together at the same depth are the fencers.

### Evaluation strategy

The template proposes evaluating such a system by accuracy against ground truth and by user
feedback. Both are adopted, but neither is sufficient, and the strategy departs from them in
three ways the results chapter justifies.

**Held-back data rather than accuracy alone.** An accuracy figure quoted on the data a system was
tuned on measures fit, not capability. Clips are therefore withheld, and the operating point
fixed on one applied unchanged to the others. This exposed the first touch detector, which scored
respectably on its own clip and collapsed elsewhere; without it the evaluation would have
reported a feature that did not work.

**Signatures of failure rather than proxies for success.** Coverage records whether the tracker
committed to a detection, not whether it committed to the right person, so a confident wrong
answer raises it. The strategy adds a measure that can only indicate failure: two fencers do not
cross on a piste, so a sign change in their relative position is a tracking error, not a fencing
event. It needs no threshold and cannot be improved by being wrong more confidently. Evaluation is
anchored where possible to a constraint the sport imposes rather than a number the system
produces.

**Resolution as well as correctness.** A metric can be correctly derived and still incapable of
distinguishing anything. Each is therefore tested for what it could detect, by block resampling
for intervals and by running the same machinery against synthetic series with known biases, so a
null is readable rather than merely absent. The template does not suggest this strand, and it is
where the methodological contribution sits.

**The effort claim** is measured by timing an assisted pass against a manual one in the same
interface. A user study would measure it better and is further work; this is what can be done
without participants.

Three limits are acknowledged in advance rather than discovered: the set is small and
single-labelled; there is no frame-level tracking ground truth, so the tracking strand measures
commitment rather than correctness; and the effort measurement, on the author alone, is an
illustration rather than a population estimate.

### Methodology and risk

Development proceeds in two-week iterations, each producing demonstrable output, with commits at
green-test states. AI components are de-risked first, so feasibility is established before the
interface is built. The principal risks are model performance on real footage, mitigated by the
assisted workflow and fallbacks; scope creep, mitigated by separating current scope from further
work; and compute cost, mitigated by frame-stride throttling on pose.

### Inclusive design

Inclusive design is broader than usability and accessibility. It asks who a system excludes and
why, and the answer is rarely that a person lacked some capacity: exclusion is produced by
processes that settled on a narrow default through convenience or habit. That radical-inclusion
position locates the barrier in the environment rather than the user, and asks that inclusivity
be present from the start.

**Who this excludes.** Para fencers, completely and not incidentally. Wheelchair fencing is fenced
from frames fixed to the floor: no footwork, no closing of distance by the feet, no lunge in the
sense the pose features model, and every measurement rests on those. Pointed at a Para bout the
system would not fail visibly, which is worse: a flat distance series, no proposed touches, a
profile at parity. **Confident nonsense on a population it was never designed for is a worse
exclusion than a refusal.**

**What was done about it.** The system now refuses: if neither fencer's position varies by more
than 0.25 m, the per-fencer axes are withheld and the reason given. The threshold comes from the
evaluation set, whose smallest spread is 0.85 m, over three times it, and is calibrated from one
side only, no wheelchair footage being available. This does not make the system inclusive; it makes it honest about the
boundary of what it measures.

**A population assumption is baked into every distance figure.** The pixel-to-metre scale assumes
a fencer height of 1.75 m, a men's senior average, applied to every bout. On a women's or junior
bout every distance is wrong by the ratio of true height to assumed, invisibly, because it scales
everything consistently. A default chosen for convenience became a structural exclusion, unnoticed
until the project was examined this way.

**Data provenance and bias.** The models were selected for availability, itself a bias. YOLOv8's
person class comes from COCO, a convenience-collected web dataset the project neither audited nor
controls; the documented failure of facial-analysis systems trained on predominantly pale-skinned
cohorts is the standing example, and nothing here establishes the detection stage is free of an
analogous skew. The evaluation set compounds it: six labelled clips from five recordings, out of
nine the pipeline was run over, mostly elite competition and all labelled by the author.

**What the design gets right, though neither was motivated by inclusion.** The review loop is
keyboard-driven, every decision one key, removing the precise pointing a scrubbing interface
demands. And colour is never the only carrier of meaning: the fencers are blue and amber, which
survives the common red-green confusions, but each is also labelled and positioned consistently,
and every state shown as a colour is also a word. Both were arrived at for legibility, which is
the point radical inclusion makes about process: getting somewhere by accident is not a method.

**What was not done.** No disabled user was consulted, no assistive technology tested, no
stakeholder outside the author's own club involved. The participatory element is absent. The
stated user, a club fencer with no budget, is an economic widening of access rather than an
inclusive one, and the two should not be conflated.

### Ethics

Evaluation footage is publicly available video used solely for non-commercial academic
evaluation, not redistributed, with rights retained by the uploaders. It shows identifiable
individuals and the system extracts position and pose data about them; no attempt is made to
identify anyone by name, no biometric identity model is used, and the system tracks anonymous
slots. If the full system stored profiles tied to named individuals, informed consent and a
retention policy would be required under UK data-protection norms. Any user testing will be
carried out with consenting healthy adults.

---

## 4. Implementation (2267/2500 words)

This chapter records what was built and where the direction changed, because on this project
the changes of direction carry most of the evidence.

### Pipeline

Per frame: YOLOv8 runs with ByteTrack filtered to the person class; detections are filtered by
the piste region; a custom `FencerTracker` assigns them to two persistent slots; MediaPipe runs
on each slot's crop; distance is measured front-foot to front-foot where pose is available and
bottom-of-box otherwise, normalised to metres by mean bounding-box height against an assumed
1.75 m fencer. Pose runs every third frame, trading sample frequency for a roughly threefold
reduction in wall-clock time.

Front foot to front foot rather than centre to centre because arm and weapon extension distort a
bounding box in a way unrelated to where the body is, and because the gap between front feet is
what coaches mean by distance. The metre scale is approximate and reported as an estimate,
consistent enough for comparison within a bout.

### Identity tracking, in four iterations

The first version locked onto the two highest-confidence ByteTrack IDs and accepted only those
thereafter: robust against bystanders, extremely fragile to ID reassignment, losing both fencers
for over 95 per cent of frames.

The second replaced strict IDs with **spatial continuity**, assignments minimising total movement
from each slot's last known position. Coverage recovered, but the matcher accepted any bystander
who happened to be the second most confident detection.

The third added two **gates**, rejecting candidates too far from a slot's last centre or too
different in height. Set strictly these dropped coverage to about 55 per cent by rejecting
legitimate motion; relaxed, they gave about 82 per cent while filtering obvious bystanders, and
cut maximum observed inter-fencer distance from 11.27 m to 6.60 m, confirming they excluded the
wrong-target outliers inflating the estimate.

The fourth added the piste filter and a **constant-velocity motion model**. The motivating case
is a referee walking into the position a fencer occupied a moment ago: under position-only gating
that is an excellent match, since the gate asks only whether the candidate is near where the slot
last was; under prediction-based gating it is poor, because the fencer was moving and the referee
is not where that motion leads. The predicted position anchors both the gate and the assignment
cost, so one change improves rejection and allocation together. Two guards proved necessary after
the first version failed on footage: velocity is estimated only from commits within three frames
of each other, since differencing positions further apart measures displacement over the gap; and
a slot unmatched for thirty frames discards its history, because one whose stale history rejects
every candidate can otherwise never recover.

The fifth change was to candidate selection, and it turned out to be the cause of both
documented failure modes. `FencerTracker.select` narrowed each frame to the two most
**confident** detections before any anchor, gate or correction was consulted. On competition
footage that is measurably false, because the referee stands on the strip and survives the piste
filter by construction. In clip 2's six-second bystander capture the fencer a user would click
is detected in every frame yet discarded by the confidence cut in 9 of 16, the three competing
confidences being 0.899, 0.896 and 0.886. The tracker was arbitrating between three people on
margins that are noise, which is also why a momentary re-anchor cannot repair a cause sustained
over six seconds. It now selects instead the two candidates nearest each slot's predicted
position, which costs nothing since those predictions already exist. Chapter 5 reports what that
changed.

### Attribution and lunge proposals

`detect_scorer.py` reads the scoring machine's lamps. Lamps rather than a score overlay, because
broadcasts carry both but club footage carries only the machine. Whole-frame saturated-colour
counts rather than a region around the machine, because club footage is hand-held and follows the
action, so the machine drifts out of frame and a fixed region is empty much of the time. A local
baseline two to three seconds earlier removes what is permanently red in shot: an exit sign,
sponsor banners, the piste. It is never permitted to propose touches, only to attribute ones
already found, since the lamps fire whenever the circuit closes, including when fencers test
weapons just after a touch. Taking touch times as given removes that failure class rather than
filtering it. This is classical colour thresholding and adds no fourth model.

`detect_lunges.py` proposes lunges from the ratio of ankle separation to hip height, calibrated
on the first five the user confirms on the bout being watched. It does not transfer a threshold
between bouts, because Chapter 5 shows transfer is what fails. Calibrating from confirmed labels
reuses the correction mechanism the design already depends on rather than adding a demand on the
user.

### The language model, and the constraints placed on it

The third pre-trained model turns the collected statistics into a written summary.
`generate_summary.py` is a separate command-line stage, decoupled from the video pipeline: it
reads the per-frame CSV, aggregates it into bout-level statistics, embeds that payload as JSON
in a structured prompt, and calls a hosted large-language-model API. Output is constrained to a
fixed structure of a summary paragraph, observed-tendency bullets each tied to a concrete
number, suggestions to explore, and a data-caveats paragraph.

Two design points matter more than the integration. The prompt carries explicit **honesty
constraints**: the model is told that distances are estimates, that aggregates are scoped by
confirmed touches, that cumulative movement totals are withdrawn, and that it must never invent
touches, scores or events absent from the data. This addresses the known risk of fabricated
plausible specifics. And generation is **never automatic and is cached**: a hash of the model
identity and both prompts is stored alongside the output and the call skipped when nothing has
changed, so re-running does not re-bill.

The summaries are constrained by what the pipeline can honestly supply rather than by the
model. They grow as the payload does, since the architecture is unchanged as confirmed touches,
in-play scoping, tempo and the per-fencer profile arrive. Seventeen unit tests cover the
statistics computation, prompt construction and cache behaviour with the API call mocked, so
the suite needs neither a network connection nor a key.

### The application layer

Every job is a JSON file on disk, so a restart loses nothing: jobs caught mid-run are marked
interrupted rather than left claiming to run, and are not retried automatically, since a video
that crashes the pipeline would otherwise crash it on every start. Progress is parsed from a
machine-readable line the pipeline prints rather than through a callback, which would have
required importing the pipeline and undone the subprocess separation.

Two interface decisions are worth recording. Summary generation costs a paid API call, so it is
never automatic; what the application layer changed is that the user presses a button rather
than being handed a command to type. And re-anchoring, the one action that changes tracking
rather than interpretation, can only take effect on a reprocess, so the interface queues that
reprocess and writes the result to a **new** bout, leaving the original intact for comparison.

### The review interface

The interface is a working instrument, and three decisions in it were reversals of what the
first version did.

Colour carries meaning and the chrome does not use it. The original palette took the Fencer 1
blue as its general accent, so the primary buttons, the progress bars and one of the two
fencers were the same hue. Since the two fencers are identified by colour throughout, that made
the most information-carrying signal in the interface indistinguishable from decoration.

The screen stopped presenting everything at once. Six bordered panels of equal visual weight
meant nothing was more important than anything else, and the single act the interface exists
for competed with five things the user was not doing. The working column now holds only the
video, the timeline, the decision loop and the record it produces; everything else moved into a
tabbed column grouped by the question it answers, with the corrective tools separated out
because they are used rarely and never during a review pass. Explanatory notes are folded
behind single-line disclosures rather than deleted: nearly every caveat here was paid for by a
measurement that went wrong, but printed simultaneously they produced a screen that was mostly
prose, and a caveat inside a wall of text is read no more carefully than an absent one.

A **review queue** holds the playhead at each proposal in turn and takes one keystroke as the
answer. This is less a new capability than the removal of a cost the earlier interface imposed
by accident: a table is the right shape for checking work already done and the wrong shape for
doing it, since finding the next undecided row and waiting for the seek repeats once per
proposal. That is precisely the cost the design claims assisted annotation removes, so leaving
it in place would have understated the system in its own evaluation.

### What the pipeline produces

![**Figure 4.1: Annotated output frame, clip 1 at 50.1 s.** Both fencers are boxed with pose
landmarks overlaid; the display shows elapsed time, smoothed distance with its derivation method,
and per-fencer net displacement and closing share.](figures/fig_annotated_clip1.png)

Figure 4.1 is the artefact the review interface plays, and it is chosen over the source video
deliberately: a user adjudicating a proposed touch can see what the tracker saw at that moment
rather than having to trust it. It also documents the piste filter working under conditions the
evaluation set otherwise has little of. A fencer on the adjacent strip, the officials behind the
desk and a spectator at the right are all present in this frame and all correctly excluded. That
this is visible at all is the argument for rendering the overlay rather than only logging the
numbers: the same frame that carries the measurement carries the evidence that it was taken from
the right two people.

![**Figure 4.2: Inter-fencer distance over time, clip 1.** Pose-derived and bounding-box fallback
samples are distinguished, with the tactical bands marked.](figures/fig_distance_clip1.png)

Figure 4.2 is the series every downstream stage consumes, and the two sample colours are the
reason it is plotted this way. The pose-derived and fallback samples are visibly interleaved
rather than segregated into stretches, which is what showed that pose availability varies frame
to frame rather than clip to clip, and is what made the paired comparison in Section 5 possible
at all. Section 5 reports the outcome of that comparison, which is that the two estimators are
close enough that the refinement does not reach the decision. Read alongside Figure 5.1, the
plot also shows why the touch detector works on this signal: the minima are sharp and the
recoveries sustained, so the pattern the detector looks for is present without smoothing chosen
to produce it.

The pipeline also writes a per-frame CSV (Appendix B), the proposed-touch CSV and the summary.

### What the pipeline produces on different footage

![**Figure 4.3: The same pipeline on tight and wide framing.** Clip 3 above, clip 4 below,
printed at equal width. The fencers are the same physical size; what differs is how much of the
frame they occupy, and clip 4's spectator gallery is visible along the
top.](figures/fig_framing_comparison.png)

![**Figure 4.4: Fencer size against two measures of tracking loss, all nine clips.** Left, the
coverage figure the chapter reports. Right, only the frames lost to gaps shorter than a second,
which is the detector losing the fencers rather than the fencers leaving the
frame.](figures/fig_framing_vs_coverage.png)

Coverage, meaning both fencers detected and held in stable slots, runs between 92.1 and 98.0 per
cent on seven of the nine clips. The two exceptions are clip 8 at 88.2 and clip 4 at 79.8.

The obvious reading of clip 4, that it is 360p and the others 720p, is wrong: **source
resolution is invisible to the detector**, which resizes so the longest side is 640. An earlier
version of this section offered the replacement explanation, that what predicts coverage is how
tall a fencer stands in that input, and on the four clips then available it fitted. On nine it
does not. The correlation across the set is r = 0.02 (Figure 4.4, left), and clip 8 has the
largest fencers of any clip alongside the second-worst coverage.

The mistake was in the measure, not the explanation. Coverage counts two unrelated things as one
loss: frames where the detector lost fencers who were there, and frames where a fencer had
walked out of shot after a touch while the referee conferred. Gap length separates them, because
a break in play lasts seconds and a detection failure lasts a few frames. Clip 4 loses 11.2 per
cent of its frames to gaps under a second, against at most 2.6 per cent anywhere else. Clip 8's
missing frames sit in 33 gaps of which the longest runs twelve seconds, so clip 8 is not tracked
badly; its fencers are repeatedly absent, which is a property of the bout rather than a defect.
Measured against short-gap loss alone the framing explanation points the right way and still
does not reach significance (r = -0.51, p = 0.16, Figure 4.4, right). Clip 4 is the one clip
with a real tracking problem and it does have the smallest fencers, but one point is an anecdote.
Nine clips cannot settle this, and Figure 4.3 is offered as the mechanism it would be, not as
evidence that it is.

Cropping to the piste and masking the gallery, the obvious interventions, both **reduced**
coverage (Appendix D), so removing image content on the assumption that it can only help is
wrong.

### Testing

The system is supported by 799 automated tests: 451 over the pipeline, 198 over the application
layer and 150 over the interface, plus two end-to-end tests driving the real pipeline on a
synthetic video. The end-to-end pair exists for what unit tests structurally cannot reach: the
stages are joined by filename conventions rather than return values, and a bout identifier is
assembled in one module and taken apart in another. It found two defects on its first run.

Tests are written against pure data behaviours where possible, so the suite runs in seconds and
adds no friction; regressions get a test before the fix, and several were confirmed by
reintroducing the defect afterwards. The limits of this are recorded honestly in Chapter 5:
every defect that mattered on this project passed its tests.

---

## 5. Evaluation (2499/2500 words)

Six bouts from five independently sourced recordings, each three minutes, hand-labelled by the
author with the scoring fencer recorded: 49 hits, 48 awarded and one annulled. Both counts are
used deliberately. Detection is scored against the 48, since a hit awarding no point cannot
appear in a scoreline; attribution against all 49, since the lamp fires on an annulled hit as
readily and reading it is the task being measured. Two clips were labelled late, after the
results below had first been computed on four.


### The audio detector failed to generalise, and geometry replaced it

The first touch detector scored candidates from a band-passed audio envelope. Tuned on clip 3 it
reached F1 0.79 and collapsed on a held-back clip:

| | clip 3 (tuned on) | clip 2 (held back) |
|---|---|---|
| Precision | 0.79 | 0.21 |
| Recall | 0.79 | 1.00 |
| F1 | 0.79 | 0.35 |
| Corrections vs manual | 6 vs 14 | 15 vs 4 |

On the held-back clip the assistance cost four times the work of labelling by hand, and
confidence lost its ranking power: real touches scored 0.74 to 0.79, false positives 0.91.

Three diagnostics established a method that cannot be retuned rather than a poorly chosen
constant (Appendix D). The decisive one: on the club clip the loudest tonal component at each touch sat at
a different frequency every time. A scoring machine has one pitch, so that scatter says the
buzzer is not reliably in the recording and the detector was finding blade contact that merely
correlates with touches.

Geometry alone, using features the pipeline already produced, scored F1 0.86 on both clips with
identical settings; the audio was not merely unhelpful but harmful, generating candidates
geometry then had to filter. The signature follows from how fencing works rather than from a fit: a
touch requires closing to scoring distance, and the halt sends both fencers back to their guard
lines. It is robust to panning, since a pan shifts both fencers together.

**Across all six labelled clips the detector reaches F1 0.73**, and 0.86 across the three it was
tuned on. The gap is the result: per-clip F1 runs 0.89, 0.86, 0.80, 0.77, 0.59 and 0.57, the two
clips added last at 0.77 and 0.57. Quoting the higher figure alone would select the best half of
the evidence, the error this chapter diagnoses elsewhere.

In the terms the design argues in, those 48 awarded touches need **25 corrections, 52 per cent of
the work of logging them by hand**, ranging from 25 per cent on clip 2 to 117 on clip 4. Clip 4
is the point of the range rather than an outlier to discount: there the assistance costs more
than the job itself. It is a saving that varies by bout, not a rate to promise.

![**Figure 5.1: Inter-fencer distance across clip 3, with the fourteen hand-labelled touches
marked.** The touches sit at local minima followed by sustained separation, which is the pattern
the detector looks for.](figures/fig_touch_signature.png)

Figure 5.1 evidences that claim and shows its limits. Every labelled touch coincides with a
minimum and not every minimum is a touch, which is why the detector is tuned for recall. The last
touch, at 179 s of 180, fails only because its separation window runs past the end.

Two points follow. **A single-clip evaluation establishes nothing**: the audio F1 of 0.79 was a
tuning artefact. And **the more sophisticated approach was the worse one**: audio
followed the literature, needed band calibration and an external dependency, and lost to a local
minimum in a signal already computed.

### A learned proposer, and what it could not beat

The detector accepts a candidate when the distance minimum has at least 0.4 m of prominence and
the fencers then separate by at least 0.8 m, both constants set against clip 3. Whether that
works because the sport has this structure or because two numbers suit these recordings is not a
question the rule can answer, so a trained classifier was put against it. Every local minimum
becomes a candidate described by fifteen features; two models are trained leave one recording
out; the rule is scored on the same candidates by the same function, or the comparison would be
between protocols; and the operating point is chosen inside each training fold. Two protocol
defects surfaced while building it, both flattering and neither visible in the output
(Appendix D).

![**Figure 5.2: A learned touch proposer against the hand rule.** Left, F1 under leave one clip
out, with the six per-clip folds drawn over the pooled bar. Right, the change in the boosted
model's F1 when each feature group is removed.](figures/fig_touch_model.png)

**The rule still leads, by less than the folds vary**: F1 0.73 against 0.70 for gradient boosted
trees and 0.46 for logistic regression, over 48 touches. The rule's own folds span 0.57 to 0.89,
so a gap of 0.03 is not a result, and the two fail on different clips: the rule scores 0.57 on 7b
against the model's 0.77, and 0.86 on clip 3 against its 0.70.

**This was first measured on four clips, and adding two changed its conclusions.** The earlier run
gave the rule 0.78 against 0.68, with an ablation putting the model's performance on separation
at 0.19 F1 and nothing else above 0.05, which supported a tidy claim: a model free to choose
differently had independently chosen the quantity the rule thresholds. **On six clips it is
gone.** Separation now costs 0.03 when removed and the group that matters is context, at 0.20:
where in the bout a candidate falls, where on the strip, how long since the last. That is a model
learning the rhythm of these recordings rather than what a touch looks like, and none of it shows
in the headline F1, so no claim about which feature carries the model survives at this sample
size.

**The learning curve is now readable, and it rises**: 0.58, 0.65 and 0.71 as the training set
grows from two recordings to four, monotone where four clips gave two contradictory points. Three
points justify no extrapolation, but the direction is at least consistent. Candidate generation
remains a domain rule: the model ranks minima rather than finding touches in video.

### The candidate cut: one cause behind both tracking failure modes

Wrong-target capture and close-range identity flicker were treated for most of this project as
limits inherent to single-camera tracking. Asking why a correctly aimed correction repaired
neither found one cause behind both: the confidence cut described in Chapter 4, which discarded
the right fencer on margins of three thousandths. Replacing it with selection by predicted
position gives:

| Clip | Coverage before / after | Mix-ups before / after | Touch F1 before / after |
|---|---|---|---|
| 1 | 92.9% / 93.5% | 2 / **0** | 0.80 / 0.80 |
| 2 | 98.0% / 98.0% | 15 / **0** | 0.86 / 0.86 |
| 3 | 97.4% / 97.4% | 0 / 0 | 0.86 / 0.86 |
| 4 | 73.7% / **79.8%** | 114 / **54** | 0.60 / 0.60 |

Mix-ups are single-frame position jumps exceeding 1.5 m. An earlier version of this table
credited the change with a rise in clip 4's F1, read at that clip's own best threshold, the error
the audio detector was diagnosed with (Appendix D).

**What this costs the argument.** Wrong-target capture was presented as evidence for the assisted
design: vision fails this way, therefore the user must repair it. On three of four clips it now
does not occur. It was a defect in candidate selection, not a limit of single-camera tracking,
and the honest reading is that the project spent a long time designing around a bug.

### Slot identity, and a check that needed no threshold

Clip 4's Fencer 2 reported +7.08 m of net displacement, which is impossible: play resets to the
guard lines after every touch, so slot identity had not held. Two fencers do not cross on a
piste, so a sign change in the difference between tracked positions is the tracker exchanging
which fencer each slot follows. Counting those split the first four clips cleanly, clips 1 to 3
recording zero swaps against clip 4's 14. Over all nine it is less tidy and more useful: four
clips record none, 7a one and 7b three, and clips 5, 8 and 4 record 19, 15 and 14, so identity
degrades by degrees rather than holding or failing. Clip 4's masked variant meanwhile
passes the implausibility test while exchanging fencers 64 times: **a believable-looking number
is not evidence that identity held.** The system now warns on any swap,
needing no threshold, because fencers do not cross.

### Attributing a touch to a fencer

Evaluated leave-one-out against the scorer column in the ground truth:

| clip | three-way (left/right/double) | green lamp alone |
|---|---|---|
| 1 | 3/3 | 3/3 |
| 2 | 2/4 | 4/4 |
| 3 | 9/14 | 14/14 |
| 4 | 4/6 | 6/6 |
| 7a | 5/9 | 9/9 |
| 7b | 10/13 | 12/13 |
| **all** | **33/49 (67%)** | **48/49 (98%)** |

Nearly every three-way error is a red-lamp error, red being contaminated by everything
permanently red in frame. The claim is narrower than "the system says who scored": it identifies
whether one named fencer was involved in 48 of 49 touches. The user confirms the colour-to-side
mapping once per bout.

**That figure was 27 of 27 on the first four clips.** It was not wrong; calling it "always" would
have been. The same two clips weakened three headline claims in one afternoon of labelling, this
one, the pose comparison's significance and the proposer's ablation.

### Lunge detection works per bout and does not transfer

An operating point fitted on clip 3 reaches F1 0.22 on clip 2 while firing on a quarter of all
windows: clip 3 calibrates to a stance ratio of 1.902 and clip 2 to 2.706, a 42 per cent
difference in what counts as a lunge-like posture. Calibrating instead on a bout's first five
confirmed lunges and testing on those that follow, clip 3 reaches F1 0.75 on 25 held-out lunges,
and a second labelling session by the same labeller reached 0.80. Precision is a lower bound,
since firing on a real but unlabelled lunge counts against it. The claim is not that lunge
detection works, but that per-bout calibration does.

### What closing share can resolve, which is nothing

Closing share, the proportion of moving frames spent reducing distance, survived scrutiny on the
reasoning that counting direction is robust where summing magnitudes is not, which says nothing
about resolution. Across all nine clips the eighteen fencer-clip values fall between 45.9 and
52.8 per cent, and resampling contiguous blocks of a third of a second, to respect the
autocorrelation between frames, **one of the eighteen separates from a coin flip.** The same
procedure fires on 11 per cent of series with no tendency at all, so one hit in eighteen is
chance rather than a finding. A null is only readable alongside what the method could detect: on
three minutes it finds a 60/40 tendency every time and 52/48 in 30 per cent of trials, so the
measurements are consistent both with balance and with tendencies too small to resolve.

This is the fifth measurement here to survive its own reasoning and fail a test of what it can
resolve, specific enough to state as a rule: **a metric's derivation establishes what it means,
never what it can detect.**

### Characterising a fencer without accumulating error

Cumulative push and pull totals were withdrawn because they accumulated (Appendix D): clip 3's
Fencer 2 reached +23.01 m against an endpoint difference of -0.94 m, because a fencer re-acquired
after a dropout contributes a one-sided step that never cancels. The operation failed, not the
series.

![**Figure 5.3: Path length and net displacement across median smoothing windows, clip 3.** Both
are computed from the same smoothed series at each window.](figures/fig_smoothing_sweep.png)

Figure 5.3 is why they were withdrawn rather than qualified. Path length falls monotonically as
the smoothing window widens with no asymptote, so any figure quoted describes the smoother rather
than the fencer, while net displacement from the same series moves once and settles.

A six-axis per-fencer profile was therefore built in which no axis accumulates: three a mean or
quantile of position, three counts over confirmed touches. On clip 3 the fencers separate on a
reading no earlier number expressed, Fencer 2 occupying nearly twice the ground while Fencer 1
scores at longer distance. Whether that describes the bout is unverified.

The profile introduces the system's first outright refusal, because all six axes assume slot
identity held and where it did not a profile would describe the tracker while looking just as
convincing. A caveat beside a number is sound, beside a shape it is not: a radar with a footnote
is still read as a radar.

### Measuring the effort the design claims to save

The central claim is that confirming proposals costs less effort than labelling from scratch, and
nothing measured it. A review queue and a manual mode are the two conditions, the manual mode
withholding proposals rather than dimming them, since a visible suggestion has been read by the
time it is ignored. Both time themselves per decision.

**Measured on two bouts neither condition had seen before.** Manual logging took 181 s for 6
touches, 30.2 s per decision; assisted review 26 s for 7, at 3.7 s. The assisted figure keeps the
four seconds the reviewer spent realising the run had ended.

**The ratio is a property of the clip, so the useful figure is the crossover.** Manual review is
bounded below by the recording's length, because finding touches means watching the bout, while
assisted pays only per decision. The two meet at about **16 touches per minute**, a touch every
3.7 seconds and not a rate epee produces: pool bouts and direct eliminations run near 1.7 and the
densest clip here reaches 4.7. The advantage holds across the range the sport generates and
widens with duration.

That assisted would win was close to guaranteed by that structure, so the direction is not the
discovery. The size of the assisted cost was: seek latency or rejecting bad proposals could have
eaten it, and an earlier interface did exactly that. The limitations are severe. **One run per
condition, one participant, who built the system, and the two conditions on different bouts**, so
bout difficulty is confounded with mode. A second person would be worth more than a hundred
further runs by this one.

### Evaluating the models as components

Taking each model separately exposes what the pipeline-level figures hide. **Detection is
measured by commitment, not correctness**: coverage records whether the tracker committed, so a
confidently wrong detection raises it.

**Pose was reported as a configuration, not a capability.** The share of frames quoted, between a
quarter and a third, is governed mainly by the frame stride. Pose and the bounding box estimate
the same quantity from different landmarks, so wherever pose succeeded both exist and can be
compared, which is why both are recorded every frame. Judged on the decision distance exists
to serve, separating the two-second windows holding a labelled touch from the rest, the bounding
box scores 0.821 against pose's 0.807 over 535 windows, an interval of -0.040 to +0.011 spanning
zero. **An earlier run on four clips put that interval at -0.091 to -0.006, which excludes
zero**, and that firmer conclusion did not survive two more recordings: pose leads on three clips
of six. Neither estimate is checked against a measured distance, so what survives is that the
refinement does not reach the decision.

**The language model was not evaluated at all**, the weakest point in the original strategy. Its
documented failure is fabrication: fluent prose containing figures never in the input, the
fluency being what stops a reader noticing. `audit_summary.py` checks every number against the
payload, finding **58 figures, none unsupported**, across two summaries.

That is only readable alongside what the audit could catch, so it was run against fabricated
figures. **The first version detected none**, because treating any two payload values' difference
or sum as derivation let sixty values cover the small numbers densely. Restricting derivation to
direct matches raised detection to 90 to 98 per cent for any figure with a decimal or above 20.
The honest result is narrower than "the summaries are faithful": **no fabricated decimal figure
appears in either**, on an instrument measured to catch them nine times in ten.

---

## 6. Conclusion (900/1000 words)

### Against the original aims

The project set out to orchestrate several pre-trained models into a system that helps a fencer
analyse a bout, on the argument that unreliable computer vision becomes useful when a person
can repair each class of error cheaply. Three models were orchestrated in a genuine chain
rather than demonstrated side by side: detections bound the region pose works in, pose
landmarks supply the ground reference the distance series is measured from, that series drives
touch proposal, confirmed touches scope every aggregate, and those aggregates are what the
language model interprets. Each stage consumes the previous one, and a defect in an early stage
propagates, which Chapter 5 demonstrates rather than assumes.

The measurable outcomes are weaker than the first four clips suggested. Touch detection reaches
F1 0.73 across six labelled clips, against 0.86 on the three it was tuned on, and both clips added
last fall below that. Tracking holds both fencers in 92.1 to 98.0 per cent of frames on seven of
the nine clips the pipeline was run over. Attribution, which the detector could never supply,
identifies whether one named fencer was involved in 48 of 49 labelled touches. These are the
claims the project can defend.

The central claim is now measured, late and thinly. Assisted review cost 3.7 s per decision
against 30.2 s for manual logging, and the two costs cross at a touch density the sport does not
produce. It rests on one run per condition by the system's author on two different bouts, which
is an instrument demonstrated on itself rather than a user study, and the honest description of
the gap has changed rather than closed: from a claim with no measurement behind it to a claim
with one participant behind it.

### What the project learned about its own method

The most transferable outcome is methodological, and it recurred often enough to be a finding
rather than a series of accidents. **Five separate defects passed their tests while measuring
the wrong thing.** A distance-band classifier used thresholds invented rather than derived, and
inverted a clip's headline reading once corrected. A movement metric accumulated its own
smoothing and reported twenty-four metres of travel on a fourteen-metre piste. A resolution
ablation varied a quantity the detector never sees, so its flat result was guaranteed in
advance. A piste polygon placed by eye raised coverage while quadrupling the count of
physically impossible readings. And the tracker chose between three people using confidence
margins of about 0.01, which is noise, for the whole life of the project.

Each was invisible to unit testing because each was implemented exactly as specified. What
exposed them was measurement against reality: a physical constraint the sport imposes, a
held-back clip, a metric chosen because it could only move one way. The rules the project ended
with are that a parameter should be derived from data rather than chosen; that a single-clip
result establishes nothing until a held-back clip agrees; and that a headline metric improving
is not evidence when a second metric can move the opposite way. Clip 4 demonstrated that last
point twice, with coverage rising while wrong-target capture rose too.

The corollary is uncomfortable and worth stating. Wrong-target capture was presented for most
of the project as evidence for the assisted design. It was a defect in candidate selection, and
on three of four clips it no longer occurs. The design argument survives on the failures that
remain, but a substantial amount of design effort went into accommodating a bug.

The same pattern held for presentation. The interface was rebuilt four times, and not one of
the intermediate failures showed up in a test, because what the tests assert is that the right
information is reachable and in every version it was.

### Limitations

Four qualify everything above. There is **no user evaluation**, so the effort claim is
unevidenced. **Pose does not fully earn its place**: it improves the distance measurement, but
its headline application to lunge detection produced an operating point that did not survive a
change of camera, and only per-bout calibration made it usable. **The sample is four clips and
one labeller**, three of them competition footage, so figures are plausibly optimistic for the
club setting that is the target use case. And **there is no frame-level tracking ground truth**,
so coverage measures whether the tracker committed to a detection, not whether it committed to
the right person.

### Future work

The immediate work is evaluative rather than technical. A user study measuring review time
against a manual baseline would settle the central claim, and weighting corrections by their
true cost would repair a metric this report currently qualifies. Both need participants rather
than code.

Technically, three directions follow from measured failures rather than from ambition. An
appearance-based re-identification embedding is the remaining lever against wrong-target
capture, since a spatial filter cannot separate a referee standing on the piste from a fencer.
Clip 4's Fencer 2 still reports an implausible seven metres of net displacement, which the
candidate-selection fix did not explain and which therefore has a cause still unidentified. And
lunge detection rests on two clips and one labeller, so more labelled footage would establish
whether per-bout calibration generalises or merely fitted twice. Beyond that, reading the score
overlay with OCR would add a fourth pre-trained model and recover the single-versus-double
distinction the lamps cannot reliably provide.

---

## References

Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F. and Grundmann, M. (2020)
'BlazePose: On-device Real-time Body Pose Tracking', arXiv:2006.10204.

Bradski, G. (2000) 'The OpenCV Library', *Dr. Dobb's Journal of Software Tools*, 25(11).

Dartfish (no date) *Dartfish*. Available at: https://www.dartfish.com/ (Accessed: 17 August
2026).

Hong, J., Fisher, M., Gharbi, M. and Fatahalian, K. (2021) 'Video Pose Distillation for
Few-Shot, Fine-Grained Sports Action Recognition', *Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV)*. arXiv:2109.01305.

Jocher, G., Chaurasia, A. and Qiu, J. (2023) *Ultralytics YOLOv8* (Version 8.0.0) [Computer
software]. Available at: https://github.com/ultralytics/ultralytics (Accessed: 5 September
2026).

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang,
C.-L., Yong, M.G., Lee, J., Chang, W.-T., Hua, W., Georg, M. and Grundmann, M. (2019)
'MediaPipe: A Framework for Building Perception Pipelines', arXiv:1906.08172.

Mo, J. (2022) 'Allez Go: Computer Vision and Audio Analysis for AI Fencing Referees', *Journal
of Student Research*, 11(4).

Mosqueira-Rey, E., Hernández-Pereira, E., Alonso-Ríos, D., Bobes-Bascarán, J. and
Fernández-Leal, Á. (2023) 'Human-in-the-loop machine learning: a state of the art', *Artificial
Intelligence Review*, 56(4), pp. 3005-3054.

Rangasamy, K., As'ari, M.A., Rahmad, N.A., Ghazali, N.F. and Ismail, S. (2020) 'Deep learning
in sport video analysis: a review', *TELKOMNIKA (Telecommunication Computing Electronics and
Control)*, 18(4), pp. 1926-1933.

Vahdani, E. and Tian, Y. (2021) 'Deep Learning-based Action Detection in Untrimmed Videos: A
Survey', arXiv:2110.00111.

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W. and Wang, X. (2022)
'ByteTrack: Multi-Object Tracking by Associating Every Detection Box', *Computer Vision - ECCV
2022*. Cham: Springer, pp. 1-21.

### Systems referred to in Chapter 2

Athlete Analyzer (no date) *Fencing Video Analysis*. Available at:
https://www.athleteanalyzer.com/video-analysis-fencing (Accessed: 5 September 2026).

AI FencingMeter (no date) *AI FencingMeter*. Available at: https://aifencingmeter.com/
(Accessed: 5 September 2026).

---

## Appendices

These carry the evidence behind Chapter 5, kept short. `final/final_report.md` in the repository
holds the full derivations; what follows is the subset a reader needs in order to check the
claims made above.

### A. Source code overview

Three directories. `code/prototype/` is the pipeline, every stage a standalone command-line
program the web layer invokes rather than reimplements, so no second code path can disagree with
the figures in Chapter 5: detection and tracking, piste measurement, touch, scorer and lunge
detection, the fencer profile, summary generation and its faithfulness audit, the learned
proposer, and the `evaluate_*.py` scorers against ground truth. `code/backend/` is the FastAPI
application; `code/frontend/` the React interface. The README lists every command.

### B. CSV schema

The per-frame CSV has columns `frame`, `time_s`, `distance_raw_m`, `distance_smooth_m`,
`distance_bbox_m`, `method` (`pose` or `bbox`), `f1_advance_m`, `f1_retreat_m`, `f2_advance_m`,
`f2_retreat_m`, `f1_pos_m`, `f2_pos_m`, `f1_stance_m`, `f2_stance_m`, `f1_hip_height_m`,
`f2_hip_height_m`.

`distance_bbox_m` holds the bounding-box distance on every frame, including those where pose
succeeded, and it is the column that makes the pose comparison in Chapter 5 a paired one. The
four advance and retreat columns are **retained but not reported**: they are the withdrawn
cumulative totals, kept because the analysis of why they fail is reproduced from them.

### C. Footage attribution

All footage is publicly available video, used solely for non-commercial academic evaluation and
not redistributed with this report. Each clip was trimmed to three minutes; no source is
reproduced in full. Clips 1 to 4 were accessed on 17 September 2026 and the rest on 18 September.

| Clip | Title | Uploader | URL |
|---|---|---|---|
| 1 | *Men's Div1 T16 Bida vs Lawson* (2023 January NAC) | HJH Fencing | https://www.youtube.com/watch?v=YNxuyeNL40M |
| 2 | *Berne Men's Epee World Cup 2026, Mencarelli vs Kanok* | Fencing Database | https://www.youtube.com/watch?v=Heim7zIE6ME |
| 3 | *15 Touch Epee Bout* | Paul Sise | https://www.youtube.com/watch?v=m3_DFZtGAyc |
| 4 | *Junior Epee Team, Basel 2024, Final, Hungary vs Kazakhstan* | Zoltan Somody | https://www.youtube.com/watch?v=KDJ65dPtlwQ |
| 5 | *2013 Magyar Bajnoksag, Ferfi Parbajtor Donto* | (Hungarian national final) | https://www.youtube.com/watch?v=f8HW1wtZWgM |
| 6 | *Epee Tymoshenko UKR vs Reizlin UKR, Averbach, Kharkov 2021* | | https://www.youtube.com/watch?v=Om50o0ZnJso |
| 7a, 7b | *War of the Roses Final, Crotchett vs Fabrice Jeannet* | | https://www.youtube.com/watch?v=KrW4hGbQmiA |
| 8 | *Samuel G-Pelletier (CAN) vs Strokatov (UKR), Vancouver World Cup 2022 pool bout* | GP Fencing | https://www.youtube.com/watch?v=EfqI5bblF4c |

**Clips 7a and 7b are two windows of one recording**, 0:05 to 3:05 and 3:05 to 6:05. They are
never held out separately in any cross-validation here: two windows of one bout share a camera,
a venue and both fencers, so treating them as independent would be testing a model on the
recording it was trained on. Clip 8 is processed and reported for coverage and tracking but is
not hand-labelled, its referee obscuring the strip often enough that labelling it would have
measured the framing rather than the detector.

Fencer 1 and Fencer 2 throughout this report are tracking slots. Nothing in the project
identifies anyone by name.

### D. Measurement investigations

Each of these is held in full in `code/prototype/RESULTS.md` and `final/final_report.md` in the
repository, with the commands that reproduce it. Summarised here only far enough to say what was
done and what it settled.

- **Why the cumulative movement totals were withdrawn.** Replaying the recorded position series
  through the accumulator accounts for the divergence exactly: clip 3's Fencer 2 shows -0.94 m
  endpoint difference against +23.01 m accumulated, and the per-frame clamp fires on 65 frames
  that net to -23.94 m because those near a lost-tracking frame are one-sided.
- **Camera motion was tested and refuted as the cause.** Optical-flow compensation recovered a
  synthetic pan exactly and did not repair the totals on real footage, which is what identified
  accumulation rather than panning.
- **The resolution ablation was invalid by construction**: it varied output encoding rather than
  the resolution the models see, so a flat result was guaranteed in advance.
- **The audio diagnostics in full**: noise floors an order of magnitude apart, five threshold
  rules that each flood one clip or find nothing in the other, and a best separation ratio below
  1.0 in every frequency band on both clips.
- **Reading a threshold off the held-back clip.** The candidate-selection change was first
  reported as raising clip 4's F1 from 0.67 to 0.77, read at clip 4's own best threshold rather
  than the operating point tuned on clip 3. At the stated point it scores 0.60 before and after.
- **Two protocol defects in the learned proposer**, both flattering, neither visible in the
  output: every candidate within the match tolerance was labelled positive, giving 1,326
  positives for 27 touches; and the learning curve's one-clip point used an untuned threshold, so
  it scored higher than two clips and read as a model learning less from more data.

### E. Development log highlights

The full record, including every measurement that did not survive and the reasoning at the time,
is in `final/final_report.md` and `code/prototype/RESULTS.md` in the repository.
