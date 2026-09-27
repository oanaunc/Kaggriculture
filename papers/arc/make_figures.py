"""Figures for the ARC-AGI-3 paper-track writeup.

usage: python make_figures.py <run_dir>=<label> ...   (each run_dir has benchmark.json)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"

runs = []
for arg in sys.argv[1:]:
    path, label = arg.split("=", 1)
    bm = json.load(open(Path(path) / "benchmark.json"))
    runs.append((label, {r["game_id"][:4]: r for r in bm["game_runs"]}))
games = sorted(runs[0][1])

# Figure 1: levels completed per game, one dot per run (dot plot, shared axis).
fig, ax = plt.subplots(figsize=(7.5, 8.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
n = len(runs)
for i, (label, data) in enumerate(runs):
    ys = [len(games) - 1 - g + (i - (n - 1) / 2) * 0.22 for g in range(len(games))]
    xs = [data[g]["levels_completed"] for g in games]
    ax.scatter(xs, ys, s=46, color=COLORS[i], edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
ax.set_yticks(range(len(games)))
ax.set_yticklabels(list(reversed(games)), color=INK, fontsize=9)
totals = [f"{label}: {sum(d[g]['levels_completed'] for g in games)} levels" for label, d in runs]
ax.set_xlabel("Levels completed", color=MUTED)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.set_title("ARC-AGI-3 public games: levels completed per game\n" + " · ".join(totals), color=INK, fontsize=10, loc="left")
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.tick_params(colors=MUTED, length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(OUT / "levels_per_game.png", dpi=160)

# Cover image (2:1).
fig = plt.figure(figsize=(8, 4), facecolor="#0d366b")
fig.text(0.06, 0.66, "Fixing the Harness, Not the Model", color="white", fontsize=23, weight="bold")
fig.text(0.06, 0.53, "Failure analysis and fixes for a small-LLM ARC-AGI-3 agent", color="#cde2fb", fontsize=13)
fig.text(0.06, 0.24, "stale game-over state  ·  hidden animation frames", color="#9ec5f4", fontsize=11)
fig.text(0.06, 0.15, "dropped memory notes  ·  context budget  ·  stuck detection", color="#9ec5f4", fontsize=11)
fig.savefig(OUT / "cover.png", dpi=160, facecolor=fig.get_facecolor())

# Table (markdown) for the writeup.
lines = ["| game | " + " | ".join(label for label, _ in runs) + " |", "|---|" + "---|" * n]
for g in games:
    lines.append(f"| {g} | " + " | ".join(f"{d[g]['levels_completed']} ({d[g]['final_score']:.1f})" for _, d in runs) + " |")
(OUT / "levels_table.md").write_text("\n".join(lines) + "\n")
print("wrote", list(OUT.iterdir()))
