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
| `results_current/` | all four clips, one pipeline version | **the set to quote from.** First set where every clip has the raw position columns, so movement figures are comparable across clips |

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

It matches the hand-authored polygon where a horizontal band can separate the groups, and is worse
where it cannot. Coverage alone does not show this, so both are quoted. Implausible = inter-fencer
distance over 6 m, same threshold both sides.

| clip | coverage hand / derived | >6 m hand / derived | verdict |
|------|-------------------------|---------------------|---------|
| 1 | 92.9% / 92.5% | 1 / 37 | worse: officials sit at y 430-460, inside a fencer band of 421-695 |
| 2 | 98.0% / 98.0% | 245 / 245 | identical |
| 4 | 73.7% / 75.9% | 208 / 301 | worse, while coverage improved |

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
