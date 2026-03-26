import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


projectRoot: Path = Path(__file__).resolve().parents[2]
dataRoot: Path = projectRoot / "capstone" / "data"


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


def loadAllWeeks() -> tuple[list[list[np.ndarray]], list[list[float]]]:
    """Find the latest week folder and load all blocks from it."""

    weekFolders: list[Path] = sorted(
        [d for d in dataRoot.iterdir() if d.is_dir() and d.name.startswith("week_")],
        key=lambda d: int(d.name.split("_")[1])
    )

    if not weekFolders:
        raise FileNotFoundError(f"No week folders found in {dataRoot}")

    latestFolder: Path = weekFolders[-1]
    print(f"Reading from: {latestFolder}")

    inputsText: str = (latestFolder / "inputs.txt").read_text(encoding="utf-8")
    outputsText: str = (latestFolder / "outputs.txt").read_text(encoding="utf-8")

    safeGlobals = {"__builtins__": {}, "array": np.array, "np": np}

    inputBlocks: list[str] = extractTopLevelBracketBlocks(inputsText)
    outputBlocks: list[str] = extractTopLevelBracketBlocks(outputsText)

    allInputs: list[list[np.ndarray]] = []
    allOutputs: list[list[float]] = []

    for inputBlock, outputBlock in zip(inputBlocks, outputBlocks):
        weekInputsRaw = eval(inputBlock, safeGlobals, {})
        weekOutputsRaw = eval(outputBlock, safeGlobals, {})

        weekInputs: list[np.ndarray] = [
            np.asarray(v, dtype=np.float64).reshape(-1) for v in weekInputsRaw
        ]
        weekOutputs: list[float] = [float(v) for v in weekOutputsRaw]

        allInputs.append(weekInputs)
        allOutputs.append(weekOutputs)

    return allInputs, allOutputs


def plotProgress(allInputs: list[list[np.ndarray]], allOutputs: list[list[float]]) -> None:

    numWeeks: int = len(allOutputs)
    numFunctions: int = 8
    weeks: list[int] = list(range(1, numWeeks + 1))

    # Per-function output over weeks
    outputsByFunction: list[list[float]] = [
        [allOutputs[w][f] for w in range(numWeeks)]
        for f in range(numFunctions)
    ]

    # Cumulative best per function
    cumulativeBest: list[list[float]] = []
    for f in range(numFunctions):
        best: float = -math.inf
        bestPerWeek: list[float] = []
        for val in outputsByFunction[f]:
            best = max(best, val)
            bestPerWeek.append(best)
        cumulativeBest.append(bestPerWeek)

    # --- Plot 1: 8 subplots, weekly output + cumulative best ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("BBO Capstone — Per-Function Progress", fontsize=14, fontweight="bold")

    colours = plt.cm.tab10.colors

    for f in range(numFunctions):
        ax = axes[f // 4][f % 4]
        ax.plot(weeks, outputsByFunction[f], "o--", color=colours[f],
                alpha=0.6, linewidth=1.2, markersize=5, label="Weekly output")
        ax.plot(weeks, cumulativeBest[f], "-", color=colours[f],
                linewidth=2.5, label="Cumulative best")
        ax.axhline(y=cumulativeBest[f][-1], color=colours[f],
                   linestyle=":", linewidth=1, alpha=0.4)
        ax.set_title(f"Function {f + 1}  (dim={len(allInputs[0][f])})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Week")
        ax.set_ylabel("Output")
        ax.set_xticks(weeks)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Annotate best value
        bestVal: float = cumulativeBest[f][-1]
        ax.annotate(f"best={bestVal:.4g}",
                    xy=(weeks[-1], bestVal),
                    xytext=(weeks[-1] - 0.5, bestVal),
                    fontsize=7, color=colours[f])

    plt.tight_layout()
    plt.savefig(dataRoot / "bbo_progress_per_function.png", dpi=150, bbox_inches="tight")
    print("Saved: bbo_progress_per_function.png")
    plt.show()

    # --- Plot 2: Cumulative best all functions on one chart (normalised) ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig2.suptitle("BBO Capstone — Cumulative Best (Normalised to Week 1)", fontsize=13, fontweight="bold")

    for f in range(numFunctions):
        baseline: float = abs(cumulativeBest[f][0]) if cumulativeBest[f][0] != 0 else 1.0
        normalised: list[float] = [v / baseline for v in cumulativeBest[f]]
        ax2.plot(weeks, normalised, "o-", color=colours[f],
                 linewidth=2, markersize=6, label=f"F{f + 1}")

    ax2.set_xlabel("Week")
    ax2.set_ylabel("Normalised cumulative best")
    ax2.set_xticks(weeks)
    ax2.legend(ncol=4, fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dataRoot / "bbo_progress_normalised.png", dpi=150, bbox_inches="tight")
    print("Saved: bbo_progress_normalised.png")
    plt.show()

    # --- Summary table print ---
    print("\n=== Cumulative Best Summary ===")
    print(f"{'Fn':<5} {'Dim':<5} " + " ".join(f"W{w:<5}" for w in weeks))
    print("-" * (12 + 7 * numWeeks))
    for f in range(numFunctions):
        dim: int = len(allInputs[0][f])
        row: str = f"F{f+1:<4} {dim:<5} "
        row += " ".join(f"{cumulativeBest[f][w-1]:<7.4g}" for w in weeks)
        print(row)


def main() -> None:
    allInputs, allOutputs = loadAllWeeks()
    print(f"Loaded {len(allOutputs)} weeks, {len(allOutputs[0])} functions\n")
    plotProgress(allInputs, allOutputs)


if __name__ == "__main__":
    main()