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

| clip | coverage old / new | mix-ups old / new | touch F1 old / new | corrections old / new |
|------|--------------------|-------------------|--------------------|-----------------------|
| 1 | 92.9% / **93.5%** | 2 / **0** | 0.80 / 0.80 | 1 of 3 / 1 of 3 |
| 2 | 98.0% / 98.0% | 15 / **0** | 0.86 / 0.86 | 1 of 4 / 1 of 4 |
| 3 | 97.4% / 97.4% | 0 / 0 | 0.86 / 0.86 | 4 of 14 / 4 of 14 |
| 4 | 73.7% / **79.8%** | 114 / **54** | 0.60 / 0.60 | 8 of 6 / 8 of 6 |

**Touch figures here are all at min confidence 0.00, which is clip 3's tuned operating point, and
that is the only protocol-compliant way to read a held-back clip.** An earlier version of this
table gave clip 4 as 0.67 rising to 0.77 with 4 then 3 corrections. Those were read at clip 4's
OWN best-scoring threshold of 0.80, which is choosing an operating point by looking at the
held-back clip. At the tuned point clip 4 is 0.60 either way: the candidate change improves
tracking, not touch detection, on that clip.

Mix-ups are single-frame position jumps over 1.5 m, the direct signature of a slot switching person.
**The 720p headline is unchanged**: clips 1-3 still give 6 corrections against 21 manual entries.
Only clip 4 moves. Nothing regressed on any clip on any measure.

Independent corroboration: play resets to the guard lines after every touch, so net displacement
should be near zero. Clip 2's Fencer 1 went from an implausible +3.86 m to -0.00 m, and clip 4's
Fencer 1 from +0.42 m to -0.05 m. That was not the target of the change.

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
