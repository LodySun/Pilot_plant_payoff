"""Poster figures 1-4, drawn at true size for the A0 portrait poster.

Each figure is drawn on a canvas exactly the size of its frame on the poster
(FRAMES, in pt; A0 is 2383 x 3370 pt), saved as a vector PDF with a
transparent background and WITHOUT bbox_inches="tight", so it is placed at
100% and every font size below is the printed size:

    ticks 22 pt, axis titles 26 pt, panel titles (A/B/C) 30 pt,
    in-figure annotations 20 pt (key labels 22 pt).

Figure titles are not drawn inside the figures (the poster carries them as
text). Figure 4 is a 2x2 grid with the bottom-right cell left empty for its
caption. A 2x PNG preview is written next to each PDF for checking only.

Inputs are the result tables written by the analysis steps (see README);
nothing is recomputed here except the participant bootstrap CI in Figure 3.

Run:  python -m src.make_poster_figures [--frame NAME=WxH ...]
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/microstructure-mpl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from src.build_controls import load_pairs  # noqa: E402
from src.common import REPO_ROOT, load_config, load_movie_timeline, resolve  # noqa: E402

OUT = REPO_ROOT / "figures"
RES = REPO_ROOT / "results"

PALETTE = {
    "blue_light": "#BCCBD8", "blue_dark": "#657E8D", "rose_dark": "#8F6067",
    "sage": "#9DADA4", "lavender": "#AAA0B8", "lavender_light": "#D9D0E0",
    "grid": "#CDD5DE", "muted": "#77828A", "paper": "#EEF3F9",
}
STRIPE_A, STRIPE_B = "#FFFFFF", "#D6DFEA"     # alternating Chen scenes
C_PLANT, C_PAYOFF = "#1f4e79", "#b5651d"

# Frames on the poster, in pt (width, height). Override with --frame NAME=WxH.
FRAMES = {
    "Figure1_design": (977, 375),
    "Figure2_viewing_null": (702, 625),
    "Figure3_recall_main": (1280, 960),
    "Figure4_robustness": (950, 978),
}

TICK, LABEL, PANEL, NOTE, KEY = 22, 26, 30, 20, 22



def bootstrap_ci(values: np.ndarray, reps: int = 10_000, seed: int = 260929) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean over participants."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(reps, values.size), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def stars(q: float) -> str:
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."


def load_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "pseudo_draws": "viewing_pseudo_null/pseudo_draws.csv.gz",
        "pseudo_summary": "viewing_pseudo_null/roi_summary.csv",
        "label_permutation": "chen_timing_corrected/order_valid_permutation.csv",
        "timecourse": "chen_recall_timecourse/timing_corrected/curve.csv",
        "timecourse_summary": "chen_recall_timecourse/timing_corrected/curve_summary.csv",
        "content": "recall_content/association_tests.csv",
    }
    return {k: pd.read_csv(RES / v) for k, v in paths.items()}

def setup() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": NOTE,
        "axes.labelsize": LABEL, "axes.titlesize": PANEL, "axes.titleweight": "bold",
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": NOTE,
        "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "axes.edgecolor": PALETTE["grid"], "grid.color": PALETTE["grid"],
        "axes.linewidth": 1.4, "grid.linewidth": 1.2,
        "xtick.major.width": 1.4, "ytick.major.width": 1.4,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "axes.facecolor": "none", "figure.facecolor": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.h_pad": 4 / 72, "figure.constrained_layout.w_pad": 4 / 72,
    })


def canvas(name: str, **kw):
    w, h = FRAMES[name]
    return plt.figure(figsize=(w / 72, h / 72), layout="constrained", **kw)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", transparent=True,           # true size, no tight bbox
                metadata={"CreationDate": None})                # reproducible bytes
    fig.savefig(OUT / f"{name}_preview.png", dpi=144, facecolor="#EEF3F9")
    plt.close(fig)


def spines(ax, keep=("left", "bottom")) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


# --------------------------------------------------------------------------- 1
def figure1() -> None:
    cfg = load_config(None)
    pairs = load_pairs(cfg, False)
    timeline = load_movie_timeline(cfg)
    duration = float(timeline["movie_duration_seconds"])
    part1 = float(timeline["part1_seconds"])
    scenes = pd.read_csv(resolve(cfg, "derived") / "scenes.csv")
    blocks = (scenes.groupby("scene_id").agg(start=("start_s", "min"), end=("end_s", "max"))
              .sort_values("start"))
    plant_band, payoff_band = (0.00, 0.20), (0.72, 0.92)     # plant below, payoff above

    fig = canvas("Figure1_design")
    ax = fig.add_subplot()
    for y0, y1 in (plant_band, payoff_band):
        for i, (_, b) in enumerate(blocks.iterrows()):
            ax.add_patch(Rectangle((b["start"], y0), b["end"] - b["start"], y1 - y0,
                                     facecolor=STRIPE_A if i % 2 else STRIPE_B,
                                     edgecolor="white", linewidth=0.6, zorder=1))
    mark_w = max(duration * 0.004, 6)
    for _, r in pairs.iterrows():
        p_mid = (float(r["plant_start_s"]) + float(r["plant_end_s"])) / 2
        o_mid = (float(r["payoff_start_s"]) + float(r["payoff_end_s"])) / 2
        ax.add_patch(FancyArrowPatch((p_mid, plant_band[1]), (o_mid, payoff_band[0]),
                                       arrowstyle="-|>", mutation_scale=18, shrinkA=0, shrinkB=0,
                                       color=C_PLANT, alpha=0.65, linewidth=2.2, zorder=3))
    for _, r in pairs.iterrows():
        for (t0, t1), (y0, y1), colour in (
            ((float(r["plant_start_s"]), float(r["plant_end_s"])), plant_band, C_PLANT),
            ((float(r["payoff_start_s"]), float(r["payoff_end_s"])), payoff_band, C_PAYOFF),
        ):
            ax.add_patch(Rectangle((t0, y0), max(t1 - t0, mark_w), y1 - y0,
                                     facecolor=colour, edgecolor="none", zorder=5))
    for (y0, y1), label in ((payoff_band, "payoff"), (plant_band, "plant")):
        ax.text(-duration * 0.015, (y0 + y1) / 2, label, ha="right", va="center",
                fontsize=LABEL, fontweight="bold")
    ax.plot([part1, part1], [plant_band[0], payoff_band[1] + 0.04], color="#7c8791",
            ls=(0, (4, 3)), lw=1.6, zorder=2)
    ax.text(part1, payoff_band[1] + 0.05, "Run 1 | Run 2", ha="center", va="bottom", fontsize=NOTE)
    ax.text(duration, payoff_band[1] + 0.05, f"{len(blocks)} scenes", ha="right", va="bottom",
            fontsize=NOTE)
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.02, payoff_band[1] + 0.17)
    ticks = np.arange(0, duration + 1, 600)
    ax.set_xticks(ticks, [f"{int(t // 60)}" for t in ticks])
    ax.set_yticks([])
    ax.set_xlabel("Time in movie (min)")
    spines(ax, ("bottom",))
    save(fig, "Figure1_design")


# --------------------------------------------------------------------------- 2
def figure2(data) -> None:
    summary = data["pseudo_summary"]
    main = summary[(summary.roi == "pmc_chen") & (summary.null_type == "family_a_segment")].iloc[0]
    draws = data["pseudo_draws"]
    x = draws[(draws.roi == "pmc_chen") & (draws.null_type == "family_a_segment")].dataset_score

    fig = canvas("Figure2_viewing_null")
    ax = fig.add_subplot()
    ax.hist(x, bins=34, color=PALETTE["blue_light"], edgecolor="#EEF3F9", linewidth=1.2, zorder=2)
    ax.axvline(main.pseudo_mean, color=PALETTE["blue_dark"], ls=(0, (4, 3)), lw=3)
    ax.axvline(main.observed, color=PALETTE["rose_dark"], lw=5)
    # Horizontal keys in the empty top corners, each with a line sample matching
    # its vertical line: locked pairs top-left, pseudo mean (and p) top-right.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)               # headroom for the keys
    for x0, ha, colour, style, width, text in (
        (.03, "left", PALETTE["rose_dark"], "-", 5, f"locked pairs {main.observed:.3f}"),
        (.97, "right", PALETTE["blue_dark"], (0, (4, 3)), 3,
         f"pseudo mean {main.pseudo_mean:.3f}\np = {main.pseudo_p_greater:.3f}"),
    ):
        sample = (x0, x0 + .05) if ha == "left" else (x0 - .05, x0)
        ax.plot(sample, [.955, .955], transform=ax.transAxes, color=colour, ls=style,
                lw=width, solid_capstyle="butt", clip_on=False)
        tx = x0 + .065 if ha == "left" else x0 - .065
        ax.text(tx, .985, text.replace("0.", "."), transform=ax.transAxes, ha=ha, va="top",
                fontsize=NOTE, linespacing=1.15)
    ax.set_xlim(.37, .63)
    ax.set_xticks([.40, .45, .50, .55, .60], [".40", ".45", ".50", ".55", ".60"])
    ax.set_yticks([0, 400, 800])
    ax.set_xlabel("Matched-null percentile")
    ax.set_ylabel("Pseudo-narratives")
    ax.grid(axis="y", zorder=0)
    spines(ax)
    save(fig, "Figure2_viewing_null")


# --------------------------------------------------------------------------- 3
def figure3() -> None:
    res = RES / "chen_timing_corrected"
    scores = pd.read_csv(res / "subject_scores.csv")
    scores = scores[(scores.shift_trs == 0) & (scores.metric == "identity_percentile")]
    perm = pd.read_csv(res / "order_valid_permutation.csv")
    perm = perm[perm.metric == "identity_percentile"].set_index("roi")
    ceil = pd.read_csv(RES / "identity_ceiling/identity_noise_ceiling.csv")
    ceil = ceil[ceil.metric == "identity_percentile"].set_index("roi")
    draws = pd.read_csv(res / "order_valid_null_draws.csv")
    rois = ("aud_early_chen", "pmc_chen")
    labels = {"aud_early_chen": "Early auditory\ncortex (control)", "pmc_chen": "Posterior medial\ncortex"}
    dot = {"aud_early_chen": PALETTE["sage"], "pmc_chen": PALETTE["lavender"]}
    summ = {"aud_early_chen": PALETTE["blue_dark"], "pmc_chen": PALETTE["rose_dark"]}
    k = 2.4                                              # marker / line scale for this canvas

    fig = canvas("Figure3_recall_main")
    ax = fig.add_subplot()
    rng = np.random.default_rng(260929)
    band, half = .36, .24
    for x, roi in enumerate(rois):
        vals = scores[scores.roi == roi].value.to_numpy()
        mean, (lo, hi) = float(vals.mean()), bootstrap_ci(vals)
        null = draws[draws.roi == roi].value.to_numpy()
        n_lo, n_hi = np.percentile(null, [2.5, 97.5])
        n_mean = float(perm.loc[roi, "null_mean"])
        ax.fill_between([x - band, x + band], n_lo, n_hi, color="#C3CCD6", alpha=.55, lw=0, zorder=1)
        ax.plot([x - band, x + band], [n_mean] * 2, color="#7D8894", lw=1.6 * k, zorder=2)
        if roi == "pmc_chen":
            c_lo = float(ceil.loc[roi, "ceiling_ordervalid_lower_loso"])
            c_hi = float(ceil.loc[roi, "ceiling_ordervalid_upper_all"])
            ax.fill_between([x - band, x + band], c_lo, c_hi, color="#E4CFD6", alpha=.85, lw=0, zorder=1)
        ax.scatter(x + rng.uniform(-.16, .16, vals.size), vals, s=150 * k ** 2,
                   color=dot[roi], alpha=.60, edgecolor="none", zorder=3)
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=summ[roi],
                    elinewidth=3 * k, capsize=9 * k, capthick=3 * k, zorder=5)
        ax.plot([x - half, x + half], [mean] * 2, color=summ[roi], lw=6 * k,
                solid_capstyle="butt", zorder=6)
        q = float(perm.loc[roi, "p_fdr_rois"])
        ax.text(x, .85, stars(q), ha="center", va="center",
                fontsize=56 if q < .05 else LABEL, fontweight="bold")
        if roi == "pmc_chen":
            for y, nm, val in (((c_lo + c_hi) / 2, "reliability ceiling",
                                f"{c_lo:.3f}–{c_hi:.3f}".replace("0.", ".")),
                               (mean, "observed", f"{mean:.3f}".lstrip("0")),
                               (n_mean, "null", f"{n_mean:.3f}".lstrip("0"))):
                ax.text(x + band + .05, y, f"{nm}\n{val}", ha="left", va="center",
                        fontsize=KEY, linespacing=1.1)
    ax.set_xlim(-.5, 1.95)
    ax.set_ylim(.15, .90)
    ax.set_yticks(np.arange(.2, .81, .1), [f"{v:.1f}" for v in np.arange(.2, .81, .1)])
    ax.set_xticks(range(2), [labels[r] for r in rois])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylabel("Correct-plant identity percentile")
    ax.grid(axis="y", zorder=0)
    spines(ax)
    save(fig, "Figure3_recall_main")


# --------------------------------------------------------------------------- 4
def figure4(data) -> None:
    fig = canvas("Figure4_robustness")
    axd = fig.subplot_mosaic([["A", "B"], ["C", "."]])

    # A: correct relation, grouped Raw / Payoff-controlled, rows A1 / PMC
    ax = axd["A"]
    perm = data["label_permutation"]
    rows = [(0, "identity_percentile", "aud_early_chen"), (1, "identity_percentile", "pmc_chen"),
            (2.7, "partial_identity_percentile", "aud_early_chen"),
            (3.7, "partial_identity_percentile", "pmc_chen")]
    for y, metric, roi in rows:
        r = perm[(perm.metric == metric) & (perm.roi == roi)].iloc[0]
        col = PALETTE["sage"] if "aud" in roi else PALETTE["lavender"]
        ax.plot([r.null_mean - r.null_sd, r.null_mean + r.null_sd], [y, y],
                color=PALETTE["grid"], lw=10, solid_capstyle="round", zorder=1)
        ax.scatter(r.null_mean, y, s=110, color=PALETTE["muted"], zorder=2)
        ax.scatter(r.observed, y, marker="D", s=200, color=col, edgecolor="white",
                   linewidth=1.5, zorder=4)
        if roi == "pmc_chen":
            ax.text(r.observed + .012, y, f"q = {r.p_fdr_rois:.3f}".replace("0.", "."),
                    va="center", fontsize=NOTE)
    for y, head in ((-.75, "Raw"), (1.95, "Payoff-controlled")):
        ax.text(.385, y, head, ha="left", va="center", fontsize=NOTE, fontweight="bold")
    ax.axvline(.5, color=PALETTE["muted"], ls=(0, (4, 3)), lw=1.6)
    ax.set_yticks([r[0] for r in rows], ["A1", "PMC", "A1", "PMC"])
    ax.set_ylim(4.3, -1.25)
    ax.set_xlim(.38, .68)
    ax.set_xticks([.4, .5, .6], [".4", ".5", ".6"])
    ax.set_xlabel("Identity percentile")
    ax.set_title("A  Correct relation", loc="left")
    ax.grid(axis="x", zorder=0)
    spines(ax)

    # B: onset-centred time course (payoff-controlled)
    ax = axd["B"]
    curve = data["timecourse"]
    for roi, col, lab, lw in (("pmc_chen", PALETTE["lavender"], "PMC", 4),
                              ("aud_early_chen", PALETTE["sage"], "A1", 3)):
        d = curve[(curve.roi == roi) & (curve.metric == "partial_identity_delta_r")] \
            .sort_values("relative_seconds")
        xx, yy = d.relative_seconds.to_numpy(), d.excess.to_numpy()
        ax.plot(xx, yy, color=col, lw=lw, label=lab, zorder=4)
        if roi == "pmc_chen":
            ax.fill_between(xx, d.null_p05 - d.null_mean, d.null_p95 - d.null_mean,
                            color=PALETTE["lavender_light"], alpha=.72, lw=0, zorder=1)
    ax.axvline(0, color=PALETTE["rose_dark"], lw=2, ls=(0, (3, 3)))
    ax.axhline(0, color=PALETTE["muted"], lw=1.4)
    pmc = curve[(curve.roi == "pmc_chen") & (curve.metric == "partial_identity_delta_r")]
    peak = pmc.loc[pmc.excess.idxmax()]
    ts = data["timecourse_summary"]
    ts = ts[(ts.roi == "pmc_chen") & (ts.metric == "partial_identity_delta_r")].iloc[0]
    ax.scatter([peak.relative_seconds], [peak.excess], s=160, color=PALETTE["rose_dark"],
               edgecolor="white", linewidth=1.5, zorder=6)
    ax.text(peak.relative_seconds + 1.5, peak.excess,
            f"peak +{peak.relative_seconds:.1f} s\nmaxT p = {ts.peak_p_fwer_lags:.3f}".replace("0.0", ".0"),
            va="center", fontsize=NOTE)
    ax.set_xticks([-15, 0, 15])
    ax.set_yticks([-.05, 0, .05], ["−.05", "0", ".05"])
    ax.set_xlabel("Seconds from recall onset")
    ax.set_ylabel("Evidence above shuffle")
    ax.set_title("B  Onset-centred", loc="left")
    ax.legend(frameon=False, loc="lower left", handlelength=1.2, borderaxespad=.2)
    ax.grid(axis="y", zorder=0)
    spines(ax)

    # C: spoken content
    ax = axd["C"]
    content = data["content"]
    content = content[content.roi == "pmc_chen"]
    mapping = [("keyword_hit", "Spoken\nkeyword"), ("semantic_percentile", "Semantic\ncontent"),
               ("binding_percentile", "Recall\nbinding")]
    for y, (measure, _) in enumerate(mapping):
        r = content[content.content_measure == measure].iloc[0]
        ax.plot([r.ci_low, r.ci_high], [y, y], color=PALETTE["grid"], lw=10,
                solid_capstyle="round", zorder=1)
        ax.scatter([r.partial_r], [y], s=160, color=PALETTE["sage"], edgecolor="white",
                   linewidth=1.5, zorder=3)
        if measure == "keyword_hit":
            ax.text(r.partial_r, y + .2, f"p = {r.p_cluster_signflip:.3f}".replace("0.", "."),
                    ha="center", va="top", fontsize=NOTE)
    ax.axvline(0, color=PALETTE["muted"], ls=(0, (4, 3)), lw=1.6)
    ax.set_yticks(range(3), [m[1] for m in mapping])
    ax.set_ylim(2.5, -.5)
    ax.set_xlim(-.48, .32)
    ax.set_xticks([-.4, -.2, 0, .2], ["−.4", "−.2", "0", ".2"])
    ax.set_xlabel("Partial r with identity")
    ax.set_title("C  Spoken content", loc="left")
    ax.grid(axis="x", zorder=0)
    spines(ax)
    save(fig, "Figure4_robustness")


def main(argv: list[str] | None = None) -> int:
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="output folder relative to the repository root")
    ap.add_argument("--frame", action="append", default=[],
                    help="override a frame: NAME=WIDTHxHEIGHT (pt)")
    args = ap.parse_args(argv)
    if args.out:
        OUT = REPO_ROOT / args.out
    for item in args.frame:
        name, size = item.split("=")
        w, h = size.lower().split("x")
        FRAMES[name] = (float(w), float(h))
    setup()
    data = load_inputs()
    figure1()
    figure2(data)
    figure3()
    figure4(data)
    for name, (w, h) in FRAMES.items():
        print(f"{name:22s} {w} x {h} pt  ->  {OUT / (name + '.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
