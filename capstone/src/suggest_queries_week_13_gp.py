import math
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


projectRoot: Path = Path(__file__).resolve().parents[2]
dataRootFolder: Path = projectRoot / "capstone" / "data"

candidateCount: int = 200_000
baseNoiseLevel: float = 1e-8
baseLengthScale: float = 0.2

_sqrt2: float = math.sqrt(2.0)
_sqrt2pi: float = math.sqrt(2.0 * math.pi)


def rbfKernel(xA: np.ndarray, xB: np.ndarray, lengthScale: float) -> np.ndarray:
    a2: np.ndarray = np.sum(xA * xA, axis=1, dtype=np.float64).reshape(-1, 1)
    b2: np.ndarray = np.sum(xB * xB, axis=1, dtype=np.float64).reshape(1, -1)
    squaredDistance: np.ndarray = a2 + b2 - 2.0 * (xA @ xB.T)
    return np.exp(-squaredDistance / (2.0 * lengthScale * lengthScale))


def normalCdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / _sqrt2))


def normalPdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / _sqrt2pi


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


def loadInitialData(functionFolder: Path) -> Tuple[np.ndarray, np.ndarray]:
    inputsPath: Path = functionFolder / "initial_inputs.npy"
    outputsPath: Path = functionFolder / "initial_outputs.npy"

    if not inputsPath.is_file():
        raise FileNotFoundError(inputsPath)
    if not outputsPath.is_file():
        raise FileNotFoundError(outputsPath)

    xData: np.ndarray = np.load(inputsPath).astype(np.float64)
    yData: np.ndarray = np.load(outputsPath).astype(np.float64).reshape(-1)

    if xData.shape[0] != yData.shape[0]:
        raise ValueError("Input/output row count mismatch")

    return xData, yData


def filterTopK(xTrain: np.ndarray, yTrain: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if xTrain.shape[0] <= k:
        return xTrain, yTrain
    topIndices: np.ndarray = np.argsort(yTrain)[-k:]
    return xTrain[topIndices], yTrain[topIndices]


def pickConfigForFunction(functionIndex: int) -> Dict[str, float | str | int]:
    # Week 13: final round, maximum safety, final exploitation
    configMap: Dict[int, Dict[str, float | str | int]] = {
        1: {"acq": "ei",    "kappa": 0.0, "xi": 1e-6, "localFrac": 0.75, "localStd": 0.060, "topK": 3},
        2: {"acq": "ei",    "kappa": 0.0, "xi": 1e-6, "localFrac": 1.00, "localStd": 0.003, "topK": 1},
        3: {"acq": "ei",    "kappa": 0.0, "xi": 1e-6, "localFrac": 1.00, "localStd": 0.004, "topK": 1},
        4: {"acq": "ei", "kappa": 0.0, "xi": 1e-6, "localFrac": 1.00, "localStd": 0.008, "topK": 1},
        5: {"acq": "fixed", "kappa": 0.0, "xi": 0.0,  "localFrac": 0.0,  "localStd": 0.0,   "topK": 1},
        6: {"acq": "ucb",   "kappa": 0.8, "xi": 0.0,  "localFrac": 0.75, "localStd": 0.050, "topK": 3},
        7: {"acq": "ei",    "kappa": 0.0, "xi": 1e-6, "localFrac": 0.97, "localStd": 0.010, "topK": 1},
        8: {"acq": "ucb",   "kappa": 0.0, "xi": 1e-6, "localFrac": 0.30, "localStd": 0.025, "topK": 1},
    }

    defaultConfig: Dict[str, float | str | int] = {
        "acq": "ei", "kappa": 0.0, "xi": 1e-6, "localFrac": 0.80, "localStd": 0.040, "topK": 3
    }

    return configMap.get(functionIndex, defaultConfig)


def buildCandidates(
    dimension: int,
    xCenters: np.ndarray,
    totalCount: int,
    localFraction: float,
    localStd: float
) -> np.ndarray:
    localCount: int = int(totalCount * localFraction)
    globalCount: int = totalCount - localCount

    globalCandidates: np.ndarray = np.random.rand(globalCount, dimension)

    if localCount <= 0 or xCenters.size == 0:
        return np.clip(globalCandidates, 0.0, 0.999999)

    centerIndices: np.ndarray = np.random.randint(0, xCenters.shape[0], size=localCount)
    chosenCenters: np.ndarray = xCenters[centerIndices]

    noise: np.ndarray = np.random.normal(loc=0.0, scale=localStd, size=(localCount, dimension))
    localCandidates: np.ndarray = chosenCenters + noise

    candidates: np.ndarray = np.vstack((localCandidates, globalCandidates))
    return np.clip(candidates, 0.0, 0.999999)


def buildCandidatesF1(
    xTrain: np.ndarray,
    yTrain: np.ndarray,
    totalCount: int,
    localFraction: float,
    localStd: float
) -> np.ndarray:
    """
    F1 special: dual-centre around best observation AND the mirror region.
    W6 signal at (0.420, 0.464) was the only non-zero value across 12 weeks.
    Try the mirror point (0.580, 0.536) as a third centre — completely unexplored.
    """
    dimension: int = xTrain.shape[1]
    localCount: int = int(totalCount * localFraction)
    globalCount: int = totalCount - localCount

    globalCandidates: np.ndarray = np.random.rand(globalCount, dimension)

    bestIndex: int = int(np.argmax(yTrain))
    signalIndex: int = int(np.argmin(yTrain))

    # Mirror point — attempts the opposite corner of the W6 signal
    mirrorPoint: np.ndarray = np.array([0.580, 0.536], dtype=np.float64)

    centers: np.ndarray = np.vstack((
        xTrain[bestIndex].reshape(1, -1),
        xTrain[signalIndex].reshape(1, -1),
        mirrorPoint.reshape(1, -1)
    ))

    if localCount <= 0:
        return np.clip(globalCandidates, 0.0, 0.999999)

    centerIndices: np.ndarray = np.random.randint(0, centers.shape[0], size=localCount)
    chosenCenters: np.ndarray = centers[centerIndices]

    noise: np.ndarray = np.random.normal(loc=0.0, scale=localStd, size=(localCount, dimension))
    localCandidates: np.ndarray = chosenCenters + noise

    candidates: np.ndarray = np.vstack((localCandidates, globalCandidates))
    return np.clip(candidates, 0.0, 0.999999)


def getModelParams(functionIndex: int) -> Tuple[float, float]:
    if functionIndex == 2:
        return 0.60, 1e-6
    if functionIndex == 3:
        return 0.35, 1e-8
    if functionIndex == 4:
        return 0.30, 1e-7
    if functionIndex == 6:
        return 0.28, 1e-7
    if functionIndex in (7, 8):
        return 0.22, 1e-8
    return baseLengthScale, baseNoiseLevel


def suggestNextPoint(
    functionIndex: int,
    xTrain: np.ndarray,
    yTrain: np.ndarray,
    acquisitionType: str,
    kappa: float,
    xi: float,
    localFraction: float,
    localStd: float,
    topKCount: int
) -> Tuple[np.ndarray, Dict[str, float | str | int]]:

    if acquisitionType.lower() == "fixed" and functionIndex == 5:
        suggestion: np.ndarray = np.array([0.999999, 0.999999, 0.999999, 0.999999], dtype=np.float64)
        info: Dict[str, float | str | int] = {
            "acquisition": "fixed",
            "bestSoFar": float(np.max(yTrain)),
            "fitSize": int(xTrain.shape[0]),
            "mean": float(np.max(yTrain)),
            "sigma": 0.0
        }
        return suggestion, info

    # Reduced filterTopK limits for tighter local GP calibration in week 13
    if functionIndex == 2:
        xFit, yFit = filterTopK(xTrain, yTrain, 10)
    elif functionIndex == 6:
        xFit, yFit = filterTopK(xTrain, yTrain, 10)
    elif functionIndex == 7:
        xFit, yFit = filterTopK(xTrain, yTrain, 10)
    else:
        xFit, yFit = xTrain, yTrain

    topK: int = min(topKCount, xFit.shape[0])
    topIndices: np.ndarray = np.argsort(yFit)[-topK:]
    xCenters: np.ndarray = xFit[topIndices]

    if functionIndex == 1:
        candidates: np.ndarray = buildCandidatesF1(
            xTrain=xFit,
            yTrain=yFit,
            totalCount=candidateCount,
            localFraction=localFraction,
            localStd=localStd
        )
    else:
        candidates = buildCandidates(
            dimension=xFit.shape[1],
            xCenters=xCenters,
            totalCount=candidateCount,
            localFraction=localFraction,
            localStd=localStd
        )

    yMean: float = float(np.mean(yFit))
    yStd: float = float(np.std(yFit))
    if yStd < 1e-12:
        yStd = 1.0
    yFitNorm: np.ndarray = (yFit - yMean) / yStd

    lengthScale, noiseLevel = getModelParams(functionIndex)

    meanNorm, varianceNorm = gpPosterior(
        xTrain=xFit,
        yTrain=yFitNorm,
        xTest=candidates,
        lengthScale=lengthScale,
        noiseLevel=noiseLevel
    )

    sigmaNorm: np.ndarray = np.sqrt(varianceNorm)
    bestObservedNorm: float = float(np.max(yFitNorm))
    acq: str = acquisitionType.lower()

    if acq == "variance":
        score: np.ndarray = varianceNorm
    elif acq == "ucb":
        score = meanNorm + float(kappa) * sigmaNorm
    elif acq == "pi":
        z: np.ndarray = (meanNorm - bestObservedNorm - float(xi)) / (sigmaNorm + 1e-12)
        score = normalCdf(z)
    elif acq == "ei":
        improvement: np.ndarray = meanNorm - bestObservedNorm - float(xi)
        z = improvement / (sigmaNorm + 1e-12)
        score = improvement * normalCdf(z) + sigmaNorm * normalPdf(z)
        score = np.maximum(score, 0.0)
    else:
        raise ValueError(f"Invalid acquisition type: {acquisitionType!r}")

    bestIndex: int = int(np.argmax(score))
    suggestion: np.ndarray = candidates[bestIndex]

    meanOrig: float = float(meanNorm[bestIndex] * yStd + yMean)
    sigmaOrig: float = float(sigmaNorm[bestIndex] * yStd)

    info = {
        "acquisition": acq,
        "bestSoFar": float(np.max(yTrain)),
        "fitSize": int(xFit.shape[0]),
        "mean": meanOrig,
        "sigma": sigmaOrig
    }

    return suggestion, info


def main() -> None:
    print(f"Data root folder: {dataRootFolder}\n")

    for functionIndex in range(1, 9):
        functionFolder: Path = dataRootFolder / f"function_{functionIndex}"
        xTrain, yTrain = loadInitialData(functionFolder)

        cfg: Dict[str, float | str | int] = pickConfigForFunction(functionIndex)

        suggestion, info = suggestNextPoint(
            functionIndex=functionIndex,
            xTrain=xTrain,
            yTrain=yTrain,
            acquisitionType=str(cfg["acq"]),
            kappa=float(cfg["kappa"]),
            xi=float(cfg["xi"]),
            localFraction=float(cfg["localFrac"]),
            localStd=float(cfg["localStd"]),
            topKCount=int(cfg["topK"])
        )

        portalString: str = formatPortalInput(suggestion)

        print(f"Function {functionIndex}")
        print(f"  Acquisition  : {info['acquisition']}")
        print(f"  Fit size     : {info['fitSize']} obs")
        print(f"  Best Y so far : {info['bestSoFar']}")
        print(f"  Suggestion    : {portalString}")
        print(f"  mu / sigma    : {info['mean']} / {info['sigma']}\n")

    print("Done.")
    print("Copy only the 'Suggestion' lines into the portal (no spaces).")


if __name__ == "__main__":
    main()