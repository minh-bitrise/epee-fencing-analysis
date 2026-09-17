# What each results directory is

Twelve output directories accumulated during development, none of them named for the
experiment it belongs to. This is the index. Everything here is reproducible from the
committed CSVs; the annotated MP4s are gitignored for size.

Directories are kept rather than deleted because the report's before/after comparisons
cite them, and several exist only to make a negative result reproducible.

| directory | what produced it | why it exists |
|---|---|---|
| `results/` | earliest runs, clips 1 and 2 | first working pipeline, superseded |
| `results_before/` | clips 1 and 2, pre-fix | the "before" half of an early comparison |
| `results_after/` | all four clips | the long-standing reference set. Touch detection, the LLM summaries and the evaluation figures are all quoted from here |
| `results_banked/` | all four clips | push/pull with the banking buffer, when sub-threshold movement stopped being discarded |
| `results_bank_nostab/` | clip 3 | banking with camera stabilisation off, isolating the two changes |
| `results_stabilised/` | all four clips | camera-motion compensation ON. Kept because it is the evidence that stabilisation made clip 3 worse, 21.88 to 37.34 m |
| `results_fixedscale/` | clip 3 | clip-wide fixed scale instead of the per-frame one |
| `results_ablation/` | clip 3 at 540p, 360p, 270p, 180p | the resolution ablation in TODO B1c. B1i later showed this varied a quantity the model never sees, so the flat result was inevitable |
| `results_fixed/` | clip 3 | first run after the sign-bug and cap fixes, and the first with the raw `f1_pos_m` / `f2_pos_m` columns |
| `results_pose/` | clip 3 at `--pose-stride 1` | the stance-feature experiment in B1h, and the bout to label lunges on. Only directory with the stance columns |
| `results_clip4_masked/` | clip 4, gallery masked | control for B1i. Removing the spectators made coverage worse |
| `results_clip4_crop/` | clip 4, cropped to the piste | the crop B1c proposed. Also worse |
| `results_confidence_candidates/` | the four clips under the OLD confidence-based candidate selection, kept so the draft report's figures stay reproducible. No videos |
| `results_current/` | all four clips, one pipeline version | **the set to quote from.** First set where every clip has the raw position columns, so movement figures are comparable across clips |

## Candidate selection changed on 29 Aug 2026, and `results_current` was regenerated

`FencerTracker.select` used to keep the two most CONFIDENT detections each frame and discard the
rest before any anchor was consulted. It now keeps the two NEAREST to where each slot is expected.
`--confidence-candidates` restores the old behaviour, and `results_confidence_candidates/` holds
the old outputs (CSVs, touch files and plots; the annotated videos were not kept, at about 1 GB).

Why: across clip 2's six-second bystander capture there are always exactly three detections inside
the piste, both fencers and the referee, who stands ON the strip so the region filter cannot remove
him. The left fencer is detected in every sampled frame and the confidence cut discarded them in 9
of 16, flickering between first and third place on margins around 0.01.

Two changes landed close together and are easy to conflate, so they are reported separately. Every
touch figure below is at **min confidence 0.00, clip 3's tuned operating point**, which is the only
protocol-compliant way to read a held-back clip.

**Change 1, candidate selection.** Tracking only. Measured with the frame-based window that was
still in force at the time, so this isolates the tracker.

| clip | coverage | mix-ups | touch F1 |
|------|----------|---------|----------|
| 1 | 92.9% -> **93.5%** | 2 -> **0** | 0.80, no change |
| 2 | 98.0%, no change | 15 -> **0** | 0.86, no change |
| 3 | 97.4%, no change | 0, no change | 0.86, no change |
| 4 | 73.7% -> **79.8%** | 114 -> **54** | 0.60, no change |

Mix-ups are single-frame position jumps over 1.5 m, the direct signature of a slot switching
person. Coverage and mix-ups improve or hold everywhere; touch F1 moves on no clip.

An earlier version of this table gave clip 4's touch F1 as 0.67 rising to 0.77. Both were read at
clip 4's OWN best threshold of 0.80, which is choosing an operating point by looking at the
held-back clip, and the gain disappears at the tuned point. **The candidate change improves
tracking and not touch detection.**

**Change 2, the local-minimum window** from 25 frames to 0.85 seconds. Touch detection only.

| clip | before | after |
|------|--------|-------|
| 1 | P1.00 R0.67 F1 0.80, 1 corr | no change |
| 2 | P1.00 R0.75 F1 0.86, 1 corr | P0.67 **R1.00** F1 0.80, 2 corr |
| 3 | P0.86 R0.86 F1 0.86, 4 corr | no change |
| 4 | P0.43 R1.00 F1 0.60, 8 corr | no change |

Clip 2 now finds all four touches where it missed one, and pays two false positives for it. F1
falls because it weights precision and recall equally; corrections rise because that metric treats
a rejection and a manual addition as equally expensive, which the report states is false.

**Current state of `results_current`:** both changes applied, touch files regenerated 31 Aug 2026.
Clips 1-3 need 6 corrections against 21 manual entries, unchanged from the draft. Clip 4 needs 8
against 6.

Independent corroboration of change 1: play resets to the guard lines after every touch, so net
displacement should be near zero. Clip 2's Fencer 1 went from an implausible +3.86 m to -0.00 m,
and clip 4's Fencer 1 from +0.42 m to -0.05 m. That was not the target of the change.

## Slot identity: check side swaps, not just the size of a number

`count_side_swaps` in `generate_summary.py` counts sign changes in
`f1_pos_m - f2_pos_m`. Fencers do not cross on a piste, so any swap is the tracker
exchanging which fencer a slot follows, and a per-slot figure on such a bout is not
attributable to a fencer at all. Applied across every output carrying the raw position
columns:

| directory | clip | swaps | F1 net | F2 net |
|---|---|---|---|---|
| results_current | clip 1, 2, 3 | **0** | plausible | plausible |
| results_current | clip 4 | 14 | +0.05 | +7.08 |
| results_confidence_candidates | clip 2 | 9 | **+3.86** | +0.16 |
| results_confidence_candidates | clip 4 | 17 | -0.42 | +7.54 |
| results_pose | clip 2 | 9 | +3.86 | +0.16 |
| results_clip4_crop | clip 4 cropped | **57** | +1.99 | -7.27 |
| results_clip4_masked | clip 4 masked | **64** | +1.00 | -0.59 |

Three things this settles.

**It explains clip 2's old +3.86 m exactly.** Under the previous candidate selection clip 2 swapped
9 times; under the current one it swaps 0 and the figure is -0.00 m. The candidate fix did not
merely coincide with a more plausible number, it removed the swapping that produced the
implausible one.

**It explains why the clip 4 crop and mask experiments were worse.** Both were recorded as "worse"
on coverage. The real effect is on identity: 57 and 64 swaps against a baseline of 14. Tightening
the frame around the fencers puts them closer together in the measured space, which is precisely
where the matcher runs out of ways to tell them apart.

**A plausible-looking net displacement is NOT evidence that identity held.** `results_clip4_masked`
reports +1.00 and -0.59 m, which pass the implausibility check comfortably, while swapping 64
times. The magnitude check and the identity check catch different failures, and only the second
one asks the question that matters for attributing anything to a named fencer.

## Flags matter, and are easy to forget

Clips 1, 2 and 4 need their piste configuration; clip 3 does not, because it contains no other
people. Omitting it is not a small difference: rebuilding `results_current/` without the configs
gave clip 2 **80.1 per cent** coverage against **98.0** with them, an 18 point gap that looked
exactly like a code regression until the flags were checked. The tracker had not changed. Record
the command with the output.

```
python3 run_detection.py --video fencing_clip.mp4  --output results_current --piste-config piste_clip1.json
python3 run_detection.py --video fencing_clip2.mp4 --output results_current --piste-config piste_clip2.json
python3 run_detection.py --video fencing_clip3.mp4 --output results_current
python3 run_detection.py --video fencing_clip4.mp4 --output results_current --piste-config piste_clip4.json
```

Everything else is default: pose stride 3, fixed scale on, stabilisation off.

## Piste configs can now be measured instead of hand-authored

`derive_piste.py` reproduces the procedure that produced the hand-authored polygons: sample every
tenth frame, take the pair of detections at the same apparent depth (NOT the two tallest - on
broadcast footage the nearest person is the referee), cluster their feet-y and keep the largest
group, then set the edges just outside it.

It reproduces the hand-authored polygon exactly on clip 2, the hardest one, and is worse on clips
1 and 4, where the strip recedes from the camera so other people stand at the same apparent depth
as its far end and no horizontal band can separate them.

Quote the TELEPORT column, not coverage. A slot that switches onto another person jumps metres of
piste between consecutive frames and a fencer cannot, so it is a direct signature of wrong-target
capture. Coverage is not: on clip 4 it moves the opposite way to the truth.

| clip | coverage hand / derived | teleports per 1k tracked, hand / derived |
|------|-------------------------|------------------------------------------|
| 1 | 92.9% / 92.5% | 0.20 / 5.21 |
| 2 | 98.0% / 98.0% | 1.70 / 1.70 |
| 4 | 73.7% / **75.9%** | 27.15 / **42.97** |

**Do not use "readings over 6 m" as a defect count**, which an earlier version of this section did.
Checked against frames: clip 4 at 86.0 s reads 7.44 m and shows two real fencers genuinely far
apart during a reset, because the shot is wide and they were walking back to their guard lines. A
raised rate there can mean a bystander was captured OR that resets are tracked more completely, and
those are opposite verdicts.

**Keep using the hand-authored configs for anything the report quotes.** The derived ones exist so
that an uploaded video, which has no config at all, gets something measured rather than nothing,
with the user confirming it in the interface.

## Per-fencer figures need the swap check first (4 Sep 2026)

`fencer_profile.py` computes six per-fencer axes from a results directory. All six assume slot
identity held for the bout, so it runs `count_side_swaps` first and REFUSES rather than qualifies:

```
python3 fencer_profile.py --metrics results_current/fencing_clip3_distance.csv \
                          --touches ground_truth/fencing_clip3_touches.csv
```

| clip | swaps | profile |
|---|---|---|
| 1, 2, 3 | 0 | computed |
| 4 | 14 | refused |

Measured on clip 3 with all 12 matched ground-truth touches confirmed:

| axis | Fencer 1 | Fencer 2 |
|---|---|---|
| Territory, m up the strip from own end | 2.40 | 3.19 |
| Ground used, interquartile range of position, m | 0.89 | 1.66 |
| Scoring share, doubles counted half | 58.3% | 41.7% |
| Scoring range, mean distance when they scored, m | 1.67 | 1.42 |
| Lunges per minute | not measured | not measured |
| Best run, consecutive touches | 2 | 1 |

**None of these are accumulations, and that is the point.** The withdrawn push / pull totals summed
per-frame deltas, where a re-acquisition after a dropout banks a one-sided step permanently
(appendix E.2: +23.01 m accumulated against -0.94 m of endpoint difference). Every axis above is
either an instantaneous reading averaged over frames or a count of confirmed touches, so a dropout
displaces a few samples out of thousands. **Do not add an axis that sums anything per-frame.**

The interquartile range, not the full range, for "ground used": one dropout frame at the far end of
the piste sets a full-range figure by itself, and it is the least trustworthy frame in the bout.

The lunge axis reads "not measured" on every clip because nothing supplies confirmed lunges until
someone labels them in the app. That is absence, not a measured zero, and it is reported as such.

## The scoreline is computed from touches, not tracking (5 Sep 2026)

`score_progression` and `piste_zones` in `fencer_profile.py` derive the final score, lead changes,
time each fencer spent ahead, and where along the strip the touches were scored. All of it comes
from the confirmed touch list, so it is valid on clip 4 where the per-fencer axes are refused.

Clip 3, all 12 matched ground-truth touches confirmed:

| figure | value |
|---|---|
| final | 9-9 |
| lead changes | 3 |
| Fencer 1 ahead | 69.6% of the bout |
| Fencer 2 ahead | 30.4% |
| level | 55.0 s |
| touches in the far third | 0 for either fencer |

**A defect this found.** Lead changes first read 0 on a bout where one fencer led for 87 seconds
and the other for 38. A lead almost always changes hands by passing THROUGH level, so comparing
each new leader against the current one sees the sequence 1, None, 2 and counts no swap at either
step. Tracking the last fencer to have HELD the lead gives 3. Anything comparing consecutive
leaders in this codebase should be checked for the same mistake.

Unattributed touches advance neither score and are counted separately. They are unknown events,
not nil-nil ones.

## Pose is the weaker distance estimate (17 Sep 2026)

`results_posecmp` is the four evaluation clips re-run with the `distance_bbox_m` column, which
records the bounding-box distance on EVERY frame rather than only where pose failed. That column
is the whole point: without it pose and the box are compared on disjoint sets of frames, which
measures which frames pose copes with, not what pose adds.

```bash
python3 evaluate_pose.py --results results_posecmp
```

| clip | paired frames | median diff | IQR | jitter pose | jitter box | AUC pose | AUC box |
|---|---|---|---|---|---|---|---|
| fencing_clip  | 3236 | -0.561 | 0.259 | 0.038 | 0.033 | 0.887 | 0.967 |
| fencing_clip2 | 2767 | -0.608 | 0.435 | 0.084 | 0.066 | 0.986 | 0.972 |
| fencing_clip3 | 1372 | -0.548 | 0.313 | 0.071 | 0.073 | 0.827 | 0.833 |
| fencing_clip4 |  522 | -0.091 | 0.769 | 0.431 | 0.139 | 0.586 | 0.708 |

Pooled over 361 two-second windows holding 39 labelled touches: **pose 0.759, box 0.805, a
difference of -0.046 with a 95 per cent interval of -0.091 to -0.006, which excludes zero.**

**Quote the pooled figures, not a clip.** Pose leads on exactly one clip of four, and quoting
clip 2 alone would be the single-clip error the audio detector was diagnosed with.

Pose reads about 0.55 m closer on the three 720p clips, which is expected and not itself a fault:
the front foot is ahead of the box bottom-centre. Clip 4 is where it comes apart. At 360p the
median difference collapses to -0.09 m with an IQR of 0.77, and pose's frame-to-frame jitter is
three times the box's, so the landmark is not being found in a consistent place at all.

**What this does NOT say.** Neither estimate is checked against a measured distance, and no
ground truth in this project could provide one. Both could be wrong in the same direction and
this would not see it. The claim is only that the refinement does not reach the decision distance
exists to serve.

## A learned touch proposer loses to the hand rule (17 Sep 2026)

```bash
python3 touch_features.py --results results_current
python3 train_touch.py --results results_current --curve --ablation \
        --json results_current/touch_model_eval.json
```

Leave one clip out over four clips and 27 labelled touches. The rule is applied
to the same candidate set, merged by the same step and scored by the same
function, so this is a model comparison and not a protocol comparison.

| model | precision | recall | F1 |
|---|---|---|---|
| hand rule (prominence 0.4 m, separation 0.8 m) | 0.72 | 0.85 | **0.78** |
| gradient boosted trees | 0.74 | 0.63 | 0.68 |
| logistic regression | 0.38 | 0.67 | 0.48 |

Figures are MICRO-averaged: counts summed, then rates computed. Clip 3 holds
fourteen touches and clip 1 holds three, so a mean of per-clip F1 would let a
good result on three offset a bad one on fourteen.

**Feature ablation, boosted model, F1 with each group removed**

| removed | F1 | delta |
|---|---|---|
| nothing | 0.68 | |
| separation | 0.49 | **-0.19** |
| pose | 0.64 | -0.04 |
| rates, dwell, context | 0.68 to 0.69 | about 0 |
| distance | 0.73 | +0.05 |
| prominence | 0.72 | +0.04 |

**This is the result worth quoting.** Separation carries the model and nothing
else moves it by more than 0.05, so the model puts its weight on the same
quantity the rule thresholds. Removing raw distance and prominence HELPS, which
at 27 positives is what redundant correlated features do. The rule's two
constants are therefore not arbitrary: they are where a learned model
independently ends up.

**Learning curve, mean F1 by training-set size**

| model | 2 clips | 3 clips | fold spread (sd) |
|---|---|---|---|
| logistic | 0.55 | 0.49 | 0.23 |
| boosted | 0.54 | 0.73 | 0.22 |

**Do not read a trend into this.** Two points, and the fold-to-fold standard
deviation is as large as the gap between them. The curve cannot currently tell a
data-starved model from a flat one, which is a statement about the evidence base
and the quantitative case for labelling more clips. The curve starts at two
because choosing an operating point needs an inner leave-one-clip-out inside the
training set and one clip has no inner fold.

**Two protocol defects found while building this, both of which flattered the
result.** Labelling every candidate inside the two-second tolerance as positive
gave 1,326 positives for 27 touches, about fifty per touch, turning "is this the
touch" into "is this near a touch". And the learning curve's one-clip point
silently used an untuned threshold, so it scored HIGHER than two clips: the
protocol changing, which reads exactly like the model learning less from more
data. Both are fixed and both have tests.

**What this does NOT say.** Candidate generation is still a domain rule; the
model ranks local minima and does not find touches in video. And 27 positives
is small enough that the per-clip numbers are noisy, which is why only the
pooled figures are quoted.

## Reproducing any of them

```
python3 run_detection.py --video <clip>.mp4 --output <dir>
python3 detect_touches.py --csv <dir>/<clip>_distance.csv
python3 evaluate_touches.py --candidates <dir>/<clip>_distance_touches.csv \
                            --truth ground_truth/<clip>_touches.csv
```

`--stabilise` and `--no-fixed-scale` reproduce the two directories named for them.
`--pose-stride 1` reproduces `results_pose/`. The clip 4 variants were built by the
script recorded in B1i, which derives its mask boundary from the detection histogram
rather than from a chosen pixel row.

## Which numbers to quote

`results_after/` for anything about touch detection or the summaries, because that is
the set the evaluation was run on. `results_fixed/` or `results_pose/` for anything about
movement, because only those have the raw position columns the reliable metrics are
derived from. Do not quote movement figures from `results_after/`: they come from the
cumulative-difference fallback, which TODO B1g measured as 24 m adrift.

`results_posecmp/` for the pose-against-box comparison only. It is the same pipeline as
`results_current/` plus one extra column, so its other figures should agree; quote them from
`results_current/` anyway, so there is one reference set and not two.
