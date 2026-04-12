import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────
projectRoot: Path = Path(__file__).resolve().parents[2]
dataRoot: Path = projectRoot / "capstone" / "data"

outPerFunction: Path = dataRoot / "bbo_progress_per_function.png"
outNormalised: Path = dataRoot / "bbo_progress_normalised.png"

# ── style ───────────────────────────────────────────────────────────────────
BG_DARK = "#0f1117"
BG_PANEL = "#1a1d27"
GRID_COL = "#252535"
SPINE_COL = "#333344"
TEXT_COL = "#cccccc"
LABEL_COL = "#999999"

COLOURS = [
    "#e63946",  # F1
    "#2196f3",  # F2
    "#4caf50",  # F3
    "#ff9800",  # F4
    "#9c27b0",  # F5
    "#00bcd4",  # F6
    "#f06292",  # F7
    "#78909c",  # F8
]

DIMS = [2, 2, 3, 4, 4, 5, 6, 8]


# ── data loading ─────────────────────────────────────────────────────────────
def extractTopLevelBracketBlocks(text: str) -> list[str]:
    blocks: list[str] = []
    depth: int = 0
    startIndex: int = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                startIndex = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and startIndex != -1:
                    blocks.append(text[startIndex:i + 1])
                    startIndex = -1
    return blocks


def loadAllWeeks() -> list[list[float]]:
    weekFolders: list[Path] = sorted(
        [d for d in dataRoot.iterdir() if d.is_dir() and d.name.startswith("week_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    if not weekFolders:
        raise FileNotFoundError(f"No week folders found in {dataRoot}")

    latestFolder: Path = weekFolders[-1]
    print(f"Reading from: {latestFolder}")

    outputsText: str = (latestFolder / "outputs.txt").read_text(encoding="utf-8")
    safeGlobals = {"__builtins__": {}, "array": np.array, "np": np}
    outputBlocks: list[str] = extractTopLevelBracketBlocks(outputsText)

    allOutputs: list[list[float]] = []
    for block in outputBlocks:
        weekOutputsRaw = eval(block, safeGlobals, {})
        allOutputs.append([float(v) for v in weekOutputsRaw])

    return allOutputs


def computeCumulativeBest(allOutputs: list[list[float]]) -> list[list[float]]:
    numFunctions: int = len(allOutputs[0])
    result: list[list[float]] = []
    for f in range(numFunctions):
        best: float = -math.inf
        row: list[float] = []
        for week in allOutputs:
            best = max(best, week[f])
            row.append(best)
        result.append(row)
    return result


def styleAx(ax) -> None:
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=LABEL_COL, labelsize=7)
    ax.xaxis.label.set_color(LABEL_COL)
    ax.yaxis.label.set_color(LABEL_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.6, alpha=0.8)


# ── plot 1: per-function ─────────────────────────────────────────────────────
def plotPerFunction(allOutputs: list[list[float]], cumulativeBest: list[list[float]]) -> None:
    numWeeks: int = len(allOutputs)
    weeks: list[int] = list(range(1, numWeeks + 1))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        "BBO Capstone — Per-Function Optimisation Progress",
        fontsize=14, fontweight="bold", color=TEXT_COL, y=1.02
    )

    for f in range(8):
        ax = axes[f // 4][f % 4]
        styleAx(ax)

        weeklyOutputs: list[float] = [allOutputs[w][f] for w in range(numWeeks)]
        colour: str = COLOURS[f]
        finalBest: float = cumulativeBest[f][-1]

        # shaded area under cumulative best
        ax.fill_between(weeks, cumulativeBest[f],
                        alpha=0.08, color=colour, zorder=1)

        # weekly output scatter
        ax.scatter(weeks, weeklyOutputs, color=colour,
                   alpha=0.5, s=35, zorder=3, label="Weekly output")

        # cumulative best line
        ax.plot(weeks, cumulativeBest[f], color=colour,
                linewidth=2.5, zorder=4, label="Cumulative best")

        # final best dashed reference
        ax.axhline(finalBest, color=colour, linestyle="--",
                   linewidth=0.8, alpha=0.35, zorder=2)

        # annotate final best value
        ax.annotate(
            f"{finalBest:.4g}",
            xy=(numWeeks, finalBest),
            xytext=(-6, 8), textcoords="offset points",
            fontsize=8, color=colour, fontweight="bold"
        )

        ax.set_title(
            f"F{f+1}  (dim={DIMS[f]})",
            fontsize=10, fontweight="bold", color=TEXT_COL
        )
        ax.set_xlabel("Week", fontsize=8)
        ax.set_ylabel("Output value", fontsize=8)
        ax.set_xticks(weeks)
        ax.legend(fontsize=7, loc="upper left",
                  facecolor=BG_PANEL, edgecolor=SPINE_COL,
                  labelcolor=TEXT_COL)

    plt.tight_layout(pad=1.8)
    plt.savefig(outPerFunction, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {outPerFunction}")
    plt.show()


# ── plot 2: normalised cumulative best ───────────────────────────────────────
def plotNormalised(cumulativeBest: list[list[float]]) -> None:
    numWeeks: int = len(cumulativeBest[0])
    weeks: list[int] = list(range(1, numWeeks + 1))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_DARK)
    styleAx(ax)

    for f in range(8):
        baseline: float = abs(cumulativeBest[f][0]) if cumulativeBest[f][0] != 0.0 else 1.0
        normalised: list[float] = [v / baseline for v in cumulativeBest[f]]

        ax.plot(weeks, normalised, "o-", color=COLOURS[f],
                linewidth=2.2, markersize=7,
                label=f"F{f+1}  (dim={DIMS[f]})")

        # label at end of each line
        ax.annotate(
            f"F{f+1}",
            xy=(numWeeks, normalised[-1]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=8, color=COLOURS[f], fontweight="bold",
            va="center"
        )

    ax.axhline(1.0, color="#555566", linestyle="--",
               linewidth=0.8, alpha=0.5, label="Week 1 baseline")

    ax.set_title(
        "BBO Capstone — Normalised Cumulative Best (relative to Week 1)",
        fontsize=13, fontweight="bold", color=TEXT_COL
    )
    ax.set_xlabel("Week", fontsize=10)
    ax.set_ylabel("Normalised cumulative best", fontsize=10)
    ax.set_xticks(weeks)
    ax.tick_params(colors=LABEL_COL)
    ax.xaxis.label.set_color(LABEL_COL)
    ax.yaxis.label.set_color(LABEL_COL)
    ax.legend(ncol=4, fontsize=9, facecolor=BG_PANEL,
              edgecolor=SPINE_COL, labelcolor=TEXT_COL,
              loc="upper left")

    plt.tight_layout()
    plt.savefig(outNormalised, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {outNormalised}")
    plt.show()


# ── terminal summary ─────────────────────────────────────────────────────────
def printSummary(allOutputs: list[list[float]], cumulativeBest: list[list[float]]) -> None:
    numWeeks: int = len(allOutputs)
    weeks: list[int] = list(range(1, numWeeks + 1))
    print("\n=== Final Cumulative Best Summary ===")
    header: str = f"{'Fn':<5} {'Dim':<5} " + " ".join(f"W{w:<6}" for w in weeks)
    print(header)
    print("-" * len(header))
    for f in range(8):
        row: str = f"F{f+1:<4} {DIMS[f]:<5} "
        row += " ".join(f"{cumulativeBest[f][w-1]:<8.4g}" for w in weeks)
        print(row)


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    allOutputs: list[list[float]] = loadAllWeeks()
    print(f"Loaded {len(allOutputs)} weeks, {len(allOutputs[0])} functions\n")

    cumulativeBest: list[list[float]] = computeCumulativeBest(allOutputs)

    plotPerFunction(allOutputs, cumulativeBest)
    plotNormalised(cumulativeBest)
    printSummary(allOutputs, cumulativeBest)


if __name__ == "__main__":
    main()
