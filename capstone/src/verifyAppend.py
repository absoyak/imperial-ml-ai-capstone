import numpy as np
from pathlib import Path
from typing import List, Tuple

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


def loadWeeklyTxtFiles(scriptFolder: Path) -> Tuple[List[np.ndarray], List[float]]:
    inputsPath: Path = scriptFolder / "inputs.txt"
    outputsPath: Path = scriptFolder / "outputs.txt"

    if not inputsPath.is_file():
        raise FileNotFoundError(str(inputsPath))
    if not outputsPath.is_file():
        raise FileNotFoundError(str(outputsPath))

    inputsText: str = inputsPath.read_text(encoding="utf-8")
    outputsText: str = outputsPath.read_text(encoding="utf-8")

    safeGlobals = {
        "__builtins__": {},
        "array": np.array,
        "np": np
    }

    inputBlocks = extractTopLevelBracketBlocks(inputsText)
    outputBlocks = extractTopLevelBracketBlocks(outputsText)

    if len(inputBlocks) == 0 or len(outputBlocks) == 0:
        raise ValueError("No valid list blocks found")

    # Son blok = son hafta
    weeklyInputsRaw = eval(inputBlocks[-1], safeGlobals, {})
    weeklyOutputsRaw = eval(outputBlocks[-1], safeGlobals, {})

    weeklyInputs: List[np.ndarray] = [
        np.asarray(v, dtype=np.float64).reshape(-1)
        for v in weeklyInputsRaw
    ]

    weeklyOutputs: List[float] = [
        float(v) for v in weeklyOutputsRaw
    ]

    if len(weeklyInputs) != 8 or len(weeklyOutputs) != 8:
        raise ValueError("Expected exactly 8 inputs and 8 outputs")

    return weeklyInputs, weeklyOutputs



def fmtVector(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def main() -> None:
    scriptFolder: Path = Path(__file__).resolve().parent
    projectRoot: Path = Path(__file__).resolve().parents[2]
    dataRoot: Path = projectRoot / "capstone" / "data"

    weeklyInputs, weeklyOutputs = loadWeeklyTxtFiles(scriptFolder)

    print("Weekly values parsed from txt files:")
    for i in range(1, 9):
        print(f"Function {i}: X={fmtVector(weeklyInputs[i-1])} | Y={weeklyOutputs[i-1]:.12g}")
    print()

    allOk: bool = True

    for functionIndex in range(1, 9):
        functionFolder: Path = dataRoot / f"function_{functionIndex}"
        inputsPath: Path = functionFolder / "initial_inputs.npy"
        outputsPath: Path = functionFolder / "initial_outputs.npy"

        xData: np.ndarray = np.load(inputsPath).astype(np.float64)
        yData: np.ndarray = np.load(outputsPath).astype(np.float64).reshape(-1)

        lastX: np.ndarray = xData[-1].reshape(-1)
        lastY: float = float(yData[-1])

        expectedX: np.ndarray = weeklyInputs[functionIndex - 1].reshape(-1)
        expectedY: float = float(weeklyOutputs[functionIndex - 1])

        sameDim: bool = lastX.shape == expectedX.shape
        sameX: bool = sameDim and np.allclose(lastX, expectedX, rtol=0.0, atol=1e-12)
        sameY: bool = abs(lastY - expectedY) <= 1e-12

        print(f"Function {functionIndex}")
        print(f"  Dataset shapes: X={xData.shape}, Y={yData.shape}")
        print(f"  Last row in .npy : X={fmtVector(lastX)} | Y={lastY:.12g}")
        print(f"  Expected (txt)   : X={fmtVector(expectedX)} | Y={expectedY:.12g}")
        print(f"  Match            : X={'OK' if sameX else 'FAIL'} | Y={'OK' if sameY else 'FAIL'}\n")

        if not (sameX and sameY):
            allOk = False

    print("FINAL RESULT:", "ALL OK ✅" if allOk else "MISMATCH ❌")


if __name__ == "__main__":
    main()
