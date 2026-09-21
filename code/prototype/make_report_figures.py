"""
Generate the analysis figures used in the draft report's Chapter 5.
=================================================================
Figures are exempt from the report's word count, so a finding that reads as a table
of numbers is usually better shown. Each figure here corresponds to a claim the
chapter makes in text, and nothing is plotted that the chapter does not discuss.

    python3 make_report_figures.py

Writes into ../../final/figures/. Regenerate after any pipeline change: a figure that
disagrees with its caption is worse than no figure, which happened once already when
the video overlay changed and the committed frame did not.
"""

import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "final", "figures")
BLUE, ORANGE, GREY, RED = "#2196F3", "#FF9800", "#607D8B", "#E53935"

# Print resolution. The report text column is about 170 mm, so 240 dpi here lands
# near 300 dpi on the page. The first pass used 120 to 140 and looked grainy.
DPI = 240


def load_positions(path, fencer):
    """Raw per-frame position series in metres, gaps carried forward."""
    rows = list(csv.DictReader(open(path)))
    col = f"{fencer}_pos_m"
    pos = np.array([float(r[col]) if r[col] else np.nan for r in rows])
    for i in range(1, len(pos)):
        if np.isnan(pos[i]):
            pos[i] = pos[i - 1]
    first = np.flatnonzero(~np.isnan(pos))
    if len(first):
        pos[: first[0]] = pos[first[0]]
    return pos


def median_smooth(x, w):
    if w <= 1:
        return x
    out = np.empty_like(x)
    half = w // 2
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - half): i + half + 1])
    return out


def fig_smoothing_sweep():
    """
    The figure that retires the cumulative totals.

    Two panels rather than one, and the reason matters. On a single axis scaled to path
    length, net displacement is a flat line at the bottom, which reads as exact
    invariance. It is not exactly invariant: it moves about 30 per cent and then
    settles. An earlier version of this figure and of the report claimed invariance,
    which was an artefact of comparing a smoothed path length against an unsmoothed
    net. Giving net its own axis shows both the collapse and the residual movement,
    which is what the evidence actually supports.
    """
    csv_path = "results_pose/fencing_clip3_distance.csv"
    windows = [1, 3, 5, 9, 15, 21, 31, 45, 61, 81, 101, 121]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for fencer, colour, label in (("f1", BLUE, "Fencer 1"), ("f2", ORANGE, "Fencer 2")):
        pos = load_positions(csv_path, fencer)
        paths, nets = [], []
        for w in windows:
            sm = median_smooth(pos, w)
            paths.append(np.abs(np.diff(sm)).sum())
            nets.append(sm[-1] - sm[0])
        top.plot(windows, paths, "o-", color=colour, label=label)
        bot.plot(windows, nets, "s-", color=colour, label=label)

    top.set_ylabel("Cumulative path length (m)")
    top.set_title("Path length collapses with the smoothing window; net displacement does not")
    top.grid(True, alpha=0.3); top.legend(fontsize=9)
    bot.set_ylabel("Net displacement (m)")
    bot.set_xlabel("Median smoothing window (frames)")
    bot.axhline(0, color=GREY, linewidth=0.8)
    bot.grid(True, alpha=0.3)
    bot.set_ylim(-2, 7)
    bot.annotate("note the scale: this axis spans 9 m, the one above spans 200 m",
                 xy=(0.99, 0.06), xycoords="axes fraction", ha="right",
                 fontsize=8, color=GREY)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_smoothing_sweep.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("wrote", os.path.basename(out))


def _place_labels(fig, ax, xs, ys, names, fontsize=8):
    """
    Annotate each point with the first candidate offset that collides with nothing.

    Five of the nine clips sit inside a 25 px band, so a single offset stacks their
    labels illegibly. Cycling through fixed offsets by rank in x was the first fix and
    it was not enough: an offset chosen from x alone cannot know that a label pushed
    right lands on top of a marker that happens to share its height, which is how the
    label for clip 2 ended up underneath clip 7b. Measuring the drawn extents is the
    only version that holds when the numbers move.
    """
    candidates = [(0, -16), (0, 12), (34, 0), (-34, 0),
                  (30, -14), (-30, -14), (30, 12), (-30, 12), (0, -28), (0, 24)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Markers are obstacles too, not just other labels.
    taken = []
    for xi, yi in zip(xs, ys):
        cx, cy = ax.transData.transform((xi, yi))
        taken.append((cx - 9, cy - 9, cx + 9, cy + 9))

    for xi, yi, nm in zip(xs, ys, names):
        placed = None
        for off in candidates:
            ann = ax.annotate(nm, (xi, yi), textcoords="offset points", xytext=off,
                              ha="center", va="center", fontsize=fontsize)
            bb = ann.get_window_extent(renderer=renderer)
            box = (bb.x0 - 2, bb.y0 - 2, bb.x1 + 2, bb.y1 + 2)
            if not any(box[0] < t[2] and t[0] < box[2] and
                       box[1] < t[3] and t[1] < box[3] for t in taken):
                placed = box
                break
            ann.remove()
        if placed is None:
            # Nothing fits; keep the last candidate rather than dropping the label.
            ax.annotate(nm, (xi, yi), textcoords="offset points", xytext=candidates[-1],
                        ha="center", va="center", fontsize=fontsize)
            continue
        taken.append(placed)


def fig_framing_vs_coverage():
    """
    The figure that retires the framing explanation.

    Its first version plotted four clips and showed a clean trend: the smallest
    fencer had the worst coverage, the rest lined up behind it. With nine clips the
    trend is gone, and the two panels are how the figure says so rather than hiding
    it. The left panel is the original claim tested on the full set. The right panel
    is the same x axis against the part of the loss that is actually a detector
    failure, which is the version of the claim that survives at all.

    Numbers come from measure_framing.py rather than from constants typed here. The
    constants are what let the first version go stale unnoticed while the chapter
    around it moved on.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_current", "framing.csv")
    rows = [r for r in csv.DictReader(open(path)) if r["fencer_px"]]
    px = np.array([float(r["fencer_px"]) for r in rows])
    cov = np.array([float(r["coverage_pct"]) for r in rows])
    short = np.array([float(r["short_gap_pct"]) for r in rows])
    names = [(r["clip"].replace("fencing_clip", "Clip ").strip() or "Clip") for r in rows]
    names = ["Clip 1" if n == "Clip" else n for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, y, label, title in (
            (axes[0], cov, "Tracking coverage (per cent of frames)",
             "Coverage does not follow fencer size\n(r = %.2f, p = %.2f, n = %d)"),
            (axes[1], short, "Frames lost to gaps under 1 s (per cent)",
             "Short-gap loss, the detector's own failures\n(r = %.2f, p = %.2f, n = %d)")):
        r, p = stats.pearsonr(px, y)
        n = len(px)
        for xi, yi, nm in zip(px, y, names):
            colour = RED if nm == "Clip 4" else BLUE
            ax.scatter(xi, yi, s=110, color=colour, zorder=3,
                       edgecolor="white", linewidth=1.2)
        ax.set_xlabel("Fencer height in the network's input (pixels)")
        ax.set_ylabel(label)
        ax.set_title(title % (r, p, n), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.12, y=0.18)

    # Labels go on after tight_layout, not before. Placement is measured in display
    # coordinates, and tight_layout resizes the axes, so labels placed first are
    # checked for collisions against a geometry that no longer exists by the time the
    # figure is saved. Two of them drifted back on top of each other that way.
    fig.tight_layout()
    _place_labels(fig, axes[0], px, cov, names)
    _place_labels(fig, axes[1], px, short, names)
    out = os.path.join(FIG, "fig_framing_vs_coverage.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("wrote", os.path.basename(out))


def fig_touch_signature():
    """
    Distance over time around the confirmed touches on clip 3.

    The detector's entire premise is a geometric signature: closing to scoring
    distance, then a sustained separation as the referee's halt sends both fencers
    back to their guard lines. The chapter asserts that signature; this shows it.
    """
    rows = list(csv.DictReader(open("results_current/fencing_clip3_distance.csv")))
    t = np.array([float(r["time_s"]) for r in rows])
    d = np.array([float(r["distance_raw_m"]) if r["distance_raw_m"] else np.nan
                  for r in rows])
    with open("ground_truth/fencing_clip3_touches.csv") as f:
        touches = [float(r["time_s"]) for r in
                   csv.DictReader(l for l in f if not l.startswith("#"))]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, d, linewidth=0.8, color=GREY, alpha=0.85)
    for i, tt in enumerate(touches):
        ax.axvline(tt, color=RED, linestyle="--", linewidth=1.0, alpha=0.8,
                   label="Awarded touch (hand-labelled)" if i == 0 else None)
    ax.axhline(2.6, color=ORANGE, linestyle=":", linewidth=1.0,
               label="Lunge distance (2.6 m)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Inter-fencer distance (metres)")
    ax.set_title("Awarded touches sit at distance minima followed by sustained separation")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_touch_signature.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("wrote", os.path.basename(out))


def fig_framing_frames():
    """
    One frame from clip 3 beside one from clip 4, at the same printed width.

    Deliberately the SOURCE frames, not the annotated ones. The first version used
    annotated output and the burnt-in overlay panel, which is sized as a fraction of
    the frame, covered the fencers on the 360p clip entirely: the figure obscured the
    exact thing it existed to show. The framing difference is the subject, so the
    overlay is noise here.
    """
    m = {r["clip"]: r for r in csv.DictReader(open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results_current", "framing.csv")))}
    picks = [("fencing_clip3", 40.0, "Clip 3: tight framing"),
             ("fencing_clip4", 60.0, "Clip 4: wide framing")]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.4))
    for ax, (clip, at, label) in zip(axes, picks):
        path = f"{clip}.mp4"
        # Subtitles are read from the measurements rather than typed, because the
        # typed pair outlived the run they came from and understated clip 4 by six points.
        title = (f"{label}. Fencer height {float(m[clip]['fencer_px']):.0f} px in the "
                 f"network's input, {m[clip]['coverage_pct']} per cent coverage")
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(at * cap.get(cv2.CAP_PROP_FPS)))
        ok, frame = cap.read(); cap.release()
        if not ok:
            ax.text(0.5, 0.5, f"could not read {path}", ha="center")
        else:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_framing_comparison.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("wrote", os.path.basename(out))


def fig_architecture():
    """
    The system architecture, drawn rather than typed.

    It was ASCII art inside a code block, which reads as source rather than as a
    diagram, split across page breaks in Word, and is weak against the marking
    criterion on whether diagrams are appropriate and clear. The content is
    unchanged from the design; only the presentation differs.

    Heights are computed from the row counts and the axis is sized to the total, since
    a hand-picked ylim silently pushed the last layer's text outside its box.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    FILL = {"client": "#E3F2FD", "api": "#FFF3E0", "pipe": "#F1F8E9", "data": "#F3E5F5"}
    EDGE = {"client": "#1565C0", "api": "#E65100", "pipe": "#33691E", "data": "#6A1B9A"}
    ROW, HEAD, PAD, GAP = 3.4, 7.5, 2.6, 5.4

    layers = [
        ("client", "CLIENT LAYER   React single-page application",
         ["Video upload",
          "Frame-accurate player and timeline scrubber",
          "Assisted-annotation panel: confirm, correct, add events",
          "Results dashboard: statistics, distance and tempo charts, summary"]),
        ("api", "APPLICATION AND API LAYER   FastAPI",
         ["Upload and file handling",
          "Asynchronous job dispatch and status polling",
          "Annotation create, read and update",
          "Analytics and results"]),
        ("pipe", "PROCESSING AND AI PIPELINE   background workers",
         ["0  Ingestion and preprocessing: decode, sample, normalise",
          "1  Detection and tracking: YOLOv8 + ByteTrack, giving boxes and stable IDs",
          "2  Pose estimation: MediaPipe Pose, giving body keypoints",
          "3  Feature extraction: distance, velocity, footwork, arm extension",
          "4  Event-segment proposal: probable touches and exchanges",
          "5  Analytics and profiling: statistics, tempo, tactical classification",
          "6  Language generation by LLM: written tactical summary"]),
        ("data", "DATA LAYER",
         ["Object storage: raw uploaded video",
          "Database: metadata, positions, keypoints, events, statistics, summaries"]),
    ]
    arrows = ["HTTPS, REST / JSON", "dispatch processing job",
              "persist features, events, statistics, summary   /   read for review"]

    heights = [HEAD + ROW * len(rows) + PAD for _, _, rows in layers]
    total = sum(heights) + GAP * (len(layers) - 1)

    fig, ax = plt.subplots(figsize=(9.5, 0.096 * total + 1.2))
    ax.set_xlim(0, 11.4); ax.set_ylim(-1.5, total + 1.5); ax.axis("off")

    y = total
    spans = []
    for i, ((key, title, rows), h) in enumerate(zip(layers, heights)):
        ax.add_patch(FancyBboxPatch((0.3, y - h), 9.4, h, boxstyle="round,pad=0.2",
                                    facecolor=FILL[key], edgecolor=EDGE[key],
                                    linewidth=1.6))
        ax.text(0.75, y - 4.0, title, fontsize=10.5, fontweight="bold", color=EDGE[key])
        for j, r in enumerate(rows):
            ax.text(1.15, y - HEAD - ROW * j, r, fontsize=9, va="center", color="#212121")
        spans.append((y, y - h))
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((5, y - h), (5, y - h - GAP + 0.4),
                                         arrowstyle="-|>", mutation_scale=16,
                                         linewidth=1.4, color="#455A64"))
            ax.text(5.3, y - h - GAP / 2, arrows[i], fontsize=8.5, style="italic",
                    color="#455A64", va="center")
        y -= h + GAP

    # The feedback loop, which is what makes the workflow human-in-the-loop rather
    # than a one-directional pipeline, so it is drawn rather than left to the caption.
    top = spans[0][1] - 1.2                      # just under the client layer
    stage5 = spans[2][0] - HEAD - ROW * 5        # row for stage 5
    ax.add_patch(FancyArrowPatch((9.7, top), (9.7, stage5),
                                 connectionstyle="arc3,rad=-0.32",
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.5, color="#C62828", linestyle=(0, (5, 3))))
    ax.text(11.25, (top + stage5) / 2,
            "user corrections re-feed the analytics engine at stage 5",
            fontsize=8.5, color="#C62828", rotation=270, va="center", ha="center")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_architecture.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("wrote", os.path.basename(out))


def fig_architecture_components():
    """
    The architecture as COMPONENTS ONLY.

    Draft feedback: "the system architecture diagram here should be refined to
    show just the components, while the descriptions of what each component
    comprises of is provided separately." The earlier version listed every
    stage's contents inside its box, so the diagram carried its own
    documentation and the structure it exists to show was buried in the text
    inside it. The contents move into a table in the body; the diagram keeps the
    four layers, what flows between them, and the feedback loop.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    FILL = {"client": "#E3F2FD", "api": "#FFF3E0",
            "pipe": "#F1F8E9", "data": "#F3E5F5"}
    EDGE = {"client": "#1565C0", "api": "#E65100",
            "pipe": "#33691E", "data": "#6A1B9A"}

    layers = [
        ("client", "CLIENT", "React single-page application"),
        ("api", "APPLICATION AND API", "FastAPI"),
        ("pipe", "PROCESSING AND AI PIPELINE", "background workers, 7 stages"),
        ("data", "DATA", "object storage and annotation store"),
    ]
    arrows = ["HTTPS, REST / JSON",
              "dispatch processing job",
              "persist features and events  /  read for review"]

    H, GAP = 9.0, 7.0
    total = len(layers) * H + (len(layers) - 1) * GAP

    fig, ax = plt.subplots(figsize=(8.6, 0.105 * total + 0.9))
    ax.set_xlim(0, 11.6); ax.set_ylim(-1.0, total + 1.0); ax.axis("off")

    y = total
    spans = []
    for i, (key, title, sub) in enumerate(layers):
        ax.add_patch(FancyBboxPatch((0.4, y - H), 8.8, H,
                                    boxstyle="round,pad=0.25",
                                    facecolor=FILL[key], edgecolor=EDGE[key],
                                    linewidth=1.8))
        ax.text(4.8, y - H / 2 + 1.5, title, fontsize=12, fontweight="bold",
                color=EDGE[key], ha="center", va="center")
        ax.text(4.8, y - H / 2 - 1.8, sub, fontsize=9.5, color="#424242",
                ha="center", va="center", style="italic")
        spans.append((y, y - H))
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((4.8, y - H), (4.8, y - H - GAP + 0.5),
                                         arrowstyle="-|>", mutation_scale=18,
                                         linewidth=1.5, color="#455A64"))
            ax.text(5.15, y - H - GAP / 2, arrows[i], fontsize=8.5,
                    style="italic", color="#455A64", va="center")
        y -= H + GAP

    # The feedback loop stays in the diagram. It is the one thing here that is
    # structural rather than descriptive: it is what makes the workflow
    # human-in-the-loop rather than a one-directional pipeline.
    top = spans[0][1] - 1.5
    bottom = spans[2][0] - H / 2
    ax.add_patch(FancyArrowPatch((9.2, top), (9.2, bottom),
                                 connectionstyle="arc3,rad=-0.30",
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=1.8, color="#C62828",
                                 linestyle=(0, (5, 3))))
    ax.text(11.0, (top + bottom) / 2,
            "confirmed corrections re-scope every aggregate",
            fontsize=9, color="#C62828", rotation=270, va="center", ha="center")

    fig.tight_layout()
    out = os.path.join(FIG, "fig_architecture_components.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("wrote", os.path.basename(out))


def fig_touch_model(results_json="results_current/touch_model_eval.json"):
    """
    The learned proposer against the hand rule, and what the model leans on.

    TWO PANELS BECAUSE THERE ARE TWO CLAIMS AND ONLY ONE IS THE HEADLINE. The
    left panel says the rule wins; on its own that is a bar chart of three
    numbers and invites the reader to take a 0.10 gap at face value across four
    recordings, so the per-clip folds are drawn on top of the pooled bar. Their
    spread is most of the story: clip 4 sits far below the others for every
    method, and no difference between methods is large against that.

    The right panel is the finding. Removing the separation features costs the
    model 0.19 and nothing else costs it more than 0.05, so the model puts its
    weight on the same quantity the rule thresholds. Bars run both ways because
    two groups have POSITIVE deltas, meaning the model scores better without
    them, which is what redundant correlated features do at this many positives.
    Hiding that by plotting magnitudes would turn an honest oddity into a clean story.

    EVERY LABEL HERE IS DERIVED FROM THE JSON. The touch count, the fold count, the
    legend and the right panel's title were all typed once and all went stale together
    when the set grew, and the right panel's title ended up asserting the opposite of
    the bars under it.
    """
    import json
    with open(results_json) as f:
        d = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    names = [("rule", "Hand rule"), ("boosted", "Boosted trees"),
             ("logistic", "Logistic")]
    xs = np.arange(len(names))
    pooled = [d[k]["pooled"]["f1"] for k, _ in names]
    bars = ax1.bar(xs, pooled, width=0.55,
                   color=[GREY, BLUE, BLUE], zorder=2)
    # Clip 4 is marked because it is the hardest recording in the set and the
    # panel's point is how much of the spread is one clip. It is the lowest fold
    # for the rule and for the boosted model; for logistic it is SECOND lowest,
    # behind clip 1 at 0.22, and the mark is what lets a reader see that rather
    # than take "clip 4 is always worst" on trust. An earlier version of this
    # comment asserted the stronger claim and the plot disproves it.
    for x, (k, _) in zip(xs, names):
        for r in d[k]["folds"]:
            is4 = r["clip"].endswith("clip4")
            ax1.scatter(x + 0.22, r["f1"], s=44 if is4 else 34,
                        color=RED if is4 else "#263238", marker="D" if is4 else "o",
                        zorder=5 if is4 else 4, edgecolor="white", linewidth=0.8)
    ax1.scatter([], [], color=RED, marker="D", s=44, label="clip 4 (360p)")
    ax1.scatter([], [], color="#263238", s=34,
                label=f"the other {len(d['rule']['folds']) - 1} clips (720p)")
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    for b, v in zip(bars, pooled):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.022, f"{v:.2f}",
                 ha="center", fontsize=10, fontweight="bold")
    ax1.set_xticks(xs); ax1.set_xticklabels([n for _, n in names])
    # Both the touch count and the fold count are read from the evaluation rather
    # than typed. They were typed once, and stayed at 27 touches over four folds
    # while the evaluation moved to 48 over six.
    n_touches = sum(f["touches"] for f in d["rule"]["folds"])
    ax1.set_ylabel(f"F1, micro-averaged over {n_touches} touches")
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Leave one clip out: the rule leads by less than the folds vary\n"
                  f"(dots are the {len(d['rule']['folds'])} per-clip folds)", fontsize=11)
    ax1.grid(True, axis="y", alpha=0.3, zorder=0)

    abl = d["ablation"]
    base = abl["base"]
    groups = [(g, abl[g] - base) for g in abl if g != "base"]
    groups.sort(key=lambda kv: kv[1])
    ys = np.arange(len(groups))
    ax2.barh(ys, [v for _, v in groups],
             color=[RED if v < -0.05 else (ORANGE if v < 0 else GREY)
                    for _, v in groups], zorder=2)
    for y, (g, v) in zip(ys, groups):
        ax2.text(v + (-0.008 if v < 0 else 0.008), y, f"{v:+.2f}",
                 va="center", ha="right" if v < 0 else "left", fontsize=9)
    ax2.set_yticks(ys); ax2.set_yticklabels([g for g, _ in groups])
    ax2.axvline(0, color="#263238", linewidth=1)
    ax2.set_xlabel("change in F1 when the group is removed")
    ax2.set_xlim(-0.24, 0.12)
    # The title names whichever group actually dominates. It read "Only separation
    # carries the model, which is what the rule already thresholds" for weeks after
    # the six-clip run put separation at -0.03 and context at -0.20, so the figure
    # asserted the opposite of the bars directly beneath it.
    worst, drop = groups[0]
    ax2.set_title(f"Removing {worst} costs the model most, at {drop:+.2f} F1,\n"
                  "and the four-clip run said separation", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.3, zorder=0)

    fig.tight_layout()
    out = os.path.join(FIG, "fig_touch_model.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    fig_architecture_components()
    fig_touch_model()
    fig_smoothing_sweep()
    fig_framing_vs_coverage()
    fig_touch_signature()
    fig_framing_frames()
    fig_architecture()
