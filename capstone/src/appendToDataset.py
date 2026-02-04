import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple


def loadWeeklyTxtFiles(scriptFolder: Path) -> Tuple[List[np.ndarray], List[float]]:
    inputsPath: Path = scriptFolder / "inputs.txt"
    outputsPath: Path = scriptFolder / "outputs.txt"

    if not inputsPath.is_file():
        raise FileNotFoundError(str(inputsPath))
    if not outputsPath.is_file():
        raise FileNotFoundError(str(outputsPath))

    inputsText: str = inputsPath.read_text(encoding="utf-8").strip()
    outputsText: str = outputsPath.read_text(encoding="utf-8").strip()

    safeGlobals = {
        "__builtins__": {},
        "array": np.array,
        "np": np
    }

    weeklyInputsRaw = eval(inputsText, safeGlobals, {})
    weeklyOutputsRaw = eval(outputsText, safeGlobals, {})

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
