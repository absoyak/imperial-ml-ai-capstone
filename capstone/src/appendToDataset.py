import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple

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

    inputBlocks: list[str] = extractTopLevelBracketBlocks(inputsText)
    outputBlocks: list[str] = extractTopLevelBracketBlocks(outputsText)

    if len(inputBlocks) == 0 or len(outputBlocks) == 0:
        raise ValueError("No parseable list blocks found in txt files")

    # Latest week = last block in the file
    weeklyInputsRaw = eval(inputBlocks[-1], safeGlobals, {})
    weeklyOutputsRaw = eval(outputBlocks[-1], safeGlobals, {})

    weeklyInputs: List[np.ndarray] = [np.asarray(v, dtype=np.float64).reshape(-1) for v in weeklyInputsRaw]
    weeklyOutputs: List[float] = [float(v) for v in weeklyOutputsRaw]

    if len(weeklyInputs) != 8 or len(weeklyOutputs) != 8:
        raise ValueError("Expected exactly 8 inputs and 8 outputs in weekly txt files")

    return weeklyInputs, weeklyOutputs


def appendToDataset(functionFolder: Path, newInput: Sequence[float], newOutput: float) -> None:
    inputsPath: Path = functionFolder / "initial_inputs.npy"
    outputsPath: Path = functionFolder / "initial_outputs.npy"

    if not inputsPath.is_file():
        raise FileNotFoundError(str(inputsPath))
    if not outputsPath.is_file():
        raise FileNotFoundError(str(outputsPath))

    xData: np.ndarray = np.load(inputsPath).astype(np.float64)
    yData: np.ndarray = np.load(outputsPath).astype(np.float64).reshape(-1)

    newX: np.ndarray = np.asarray(newInput, dtype=np.float64).reshape(1, -1)
    newY: np.ndarray = np.asarray([newOutput], dtype=np.float64)

    if xData.shape[1] != newX.shape[1]:
        raise ValueError(f"Input dimension mismatch in {functionFolder.name}: {xData.shape[1]} vs {newX.shape[1]}")

    np.save(inputsPath, np.vstack((xData, newX)))
    np.save(outputsPath, np.concatenate((yData, newY)))


def main() -> None:
    scriptFolder: Path = Path(__file__).resolve().parent
    projectRoot: Path = Path(__file__).resolve().parents[2]
    dataRoot: Path = projectRoot / "capstone" / "data"

    weeklyInputs, weeklyOutputs = loadWeeklyTxtFiles(scriptFolder)

    for functionIndex in range(1, 9):
        functionFolder: Path = dataRoot / f"function_{functionIndex}"
        appendToDataset(
            functionFolder=functionFolder,
            newInput=weeklyInputs[functionIndex - 1],
            newOutput=weeklyOutputs[functionIndex - 1]
        )

    print("Done")


if __name__ == "__main__":
    main()
