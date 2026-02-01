import os
import math
import numpy as np
from typing import Tuple, Dict


function1Folder: str = r"C:\Users\absoy\OneDrive\Dev\ML\Capstone\initial_data\function_1"

candidateCount: int = 50_000
noiseLevel: float = 1e-8
lengthScale: float = 0.2

ucbKappa: float = 2.0
piXi: float = 1e-6


def rbfKernel(xA: np.ndarray, xB: np.ndarray, lengthScale: float) -> np.ndarray:
    a2: np.ndarray = np.sum(xA * xA, axis=1).reshape(-1, 1)
    b2: np.ndarray = np.sum(xB * xB, axis=1).reshape(1, -1)
    squaredDistance: np.ndarray = a2 + b2 - 2.0 * (xA @ xB.T)
    return np.exp(-squaredDistance / (2.0 * lengthScale * lengthScale))


def normalCdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def gpPosterior(
    xTrain: np.ndarray,
    yTrain: np.ndarray,
    xTest: np.ndarray,
    lengthScale: float,
    noiseLevel: float
) -> Tuple[np.ndarray, np.ndarray]:

    kTrainTrain: np.ndarray = rbfKernel(xTrain, xTrain, lengthScale)
    kTrainTest: np.ndarray = rbfKernel(xTrain, xTest, lengthScale)

    kTrainTrain += noiseLevel * np.eye(kTrainTrain.shape[0], dtype=np.float64)

    try:
        lMatrix: np.ndarray = np.linalg.cholesky(kTrainTrain)
    except np.linalg.LinAlgError:
        kTrainTrain += 1e-6 * np.eye(kTrainTrain.shape[0], dtype=np.float64)
        lMatrix = np.linalg.cholesky(kTrainTrain)

    yTrainCol: np.ndarray = yTrain.reshape(-1, 1)

    vMatrix: np.ndarray = np.linalg.solve(lMatrix, yTrainCol)
    alpha: np.ndarray = np.linalg.solve(lMatrix.T, vMatrix)

    mean: np.ndarray = (kTrainTest.T @ alpha).reshape(-1)

    wMatrix: np.ndarray = np.linalg.solve(lMatrix, kTrainTest)
    variance: np.ndarray = 1.0 - np.sum(wMatrix * wMatrix, axis=0)
    variance = np.maximum(variance, 1e-12)

    return mean, variance


def formatPortalInput(x: np.ndarray) -> str:
    clipped: np.ndarray = np.clip(x, 0.0, 0.999999)
    return "-".join(f"{value:.6f}" for value in clipped.tolist())


def inferRootFolder(function1Folder: str) -> str:
    return os.path.dirname(function1Folder.rstrip("\\/"))


def loadInitialData(functionFolder: str) -> Tuple[np.ndarray, np.ndarray]:
    inputsPath: str = os.path.join(functionFolder, "initial_inputs.npy")
    outputsPath: str = os.path.join(functionFolder, "initial_outputs.npy")

    if not os.path.isfile(inputsPath):
        raise FileNotFoundError(inputsPath)
    if not os.path.isfile(outputsPath):
        raise FileNotFoundError(outputsPath)

    xData: np.ndarray = np.load(inputsPath).astype(np.float64)
    yData: np.ndarray = np.load(outputsPath).astype(np.float64).reshape(-1)

    if xData.shape[0] != yData.shape[0]:
        raise ValueError("Input/output row count mismatch")

    return xData, yData


def suggestNextPoint(
    xTrain: np.ndarray,
    yTrain: np.ndarray,
    acquisitionType: str
) -> Tuple[np.ndarray, Dict[str, float]]:

    dimension: int = xTrain.shape[1]

    candidates: np.ndarray = np.random.rand(candidateCount, dimension)
    candidates = np.minimum(candidates, 0.999999)

    mean, variance = gpPosterior(
        xTrain=xTrain,
        yTrain=yTrain,
        xTest=candidates,
        lengthScale=lengthScale,
        noiseLevel=noiseLevel
    )

    sigma: np.ndarray = np.sqrt(variance)
    bestObserved: float = float(np.max(yTrain))

    acquisition: str = acquisitionType.lower()

    if acquisition == "ucb":
        score: np.ndarray = mean + ucbKappa * sigma
    elif acquisition == "variance":
        score = variance
    elif acquisition == "pi":
        z: np.ndarray = (mean - bestObserved - piXi) / (sigma + 1e-12)
        score = normalCdf(z)
    else:
        raise ValueError("Invalid acquisition type")

    bestIndex: int = int(np.argmax(score))
    suggestion: np.ndarray = candidates[bestIndex]

    info: Dict[str, float] = {
        "bestSoFar": bestObserved,
        "mean": float(mean[bestIndex]),
        "sigma": float(sigma[bestIndex])
    }

    return suggestion, info


def main() -> None:
    rootFolder: str = inferRootFolder(function1Folder)
    acquisitionType: str = "ucb"  # ucb | variance | pi

    print(f"Root folder: {rootFolder}")
    print(f"Acquisition: {acquisitionType}\n")

    for functionIndex in range(1, 9):
        functionFolder: str = os.path.join(rootFolder, f"function_{functionIndex}")
        xTrain, yTrain = loadInitialData(functionFolder)

        suggestion, info = suggestNextPoint(
            xTrain=xTrain,
            yTrain=yTrain,
            acquisitionType=acquisitionType
        )

        portalString: str = formatPortalInput(suggestion)

        print(f"Function {functionIndex}")
        print(f"  Best Y so far : {info['bestSoFar']}")
        print(f"  Suggestion   : {portalString}")
        print(f"  mu / sigma   : {info['mean']} / {info['sigma']}\n")

    print("Done.")


if __name__ == "__main__":
    main()
