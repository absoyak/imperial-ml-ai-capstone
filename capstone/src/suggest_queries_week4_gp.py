import math
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


projectRoot: Path = Path(__file__).resolve().parents[2]
dataRootFolder: Path = projectRoot / "capstone" / "data"

candidateCount: int = 120_000
baseNoiseLevel: float = 1e-8
baseLengthScale: float = 0.2


def rbfKernel(xA: np.ndarray, xB: np.ndarray, lengthScale: float) -> np.ndarray:
    a2: np.ndarray = np.sum(xA * xA, axis=1).reshape(-1, 1)
    b2: np.ndarray = np.sum(xB * xB, axis=1).reshape(1, -1)
    squaredDistance: np.ndarray = a2 + b2 - 2.0 * (xA @ xB.T)
    return np.exp(-squaredDistance / (2.0 * lengthScale * lengthScale))


def normalCdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def normalPdf(z: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z * z)


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


def pickConfigForFunction(functionIndex: int) -> Dict[str, float | str]:
    configMap: Dict[int, Dict[str, float | str]] = {
        1: {"acq": "variance", "kappa": 0.0, "xi": 0.0, "localFrac": 0.30, "localStd": 0.12},
        2: {"acq": "ucb", "kappa": 0.4, "xi": 1e-4, "localFrac": 0.98, "localStd": 0.01},
        3: {"acq": "ei", "kappa": 0.0, "xi": 1e-3, "localFrac": 0.65, "localStd": 0.06},
        4: {"acq": "ucb", "kappa": 1.6, "xi": 0.0, "localFrac": 0.75, "localStd": 0.05},
        5: {"acq": "ei", "kappa": 0.0, "xi": 1e-6, "localFrac": 0.85, "localStd": 0.03},
        6: {"acq": "ei", "kappa": 0.0, "xi": 1e-3, "localFrac": 0.85, "localStd": 0.04},
        7: {"acq": "ei", "kappa": 0.0, "xi": 1e-6, "localFrac": 0.80, "localStd": 0.04},
        8: {"acq": "ucb", "kappa": 2.0, "xi": 0.0, "localFrac": 0.70, "localStd": 0.04},
    }

    defaultConfig: Dict[str, float | str] = {
        "acq": "ei",
        "kappa": 0.0,
        "xi": 1e-6,
        "localFrac": 0.70,
        "localStd": 0.06
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
        candidates: np.ndarray = globalCandidates
        return np.minimum(np.clip(candidates, 0.0, 0.999999), 0.999999)

    centerIndices: np.ndarray = np.random.randint(0, xCenters.shape[0], size=localCount)
    chosenCenters: np.ndarray = xCenters[centerIndices]

    noise: np.ndarray = np.random.normal(loc=0.0, scale=localStd, size=(localCount, dimension))
    localCandidates: np.ndarray = chosenCenters + noise

    candidates: np.ndarray = np.vstack((localCandidates, globalCandidates))
    candidates = np.clip(candidates, 0.0, 0.999999)
    return np.minimum(candidates, 0.999999)


def suggestNextPoint(
    functionIndex: int,
    xTrain: np.ndarray,
    yTrain: np.ndarray,
    acquisitionType: str,
    kappa: float,
    xi: float,
    localFraction: float,
    localStd: float
) -> Tuple[np.ndarray, Dict[str, float | str]]:

    dimension: int = xTrain.shape[1]

    bestIndexTrain: int = int(np.argmax(yTrain))
    xBest: np.ndarray = xTrain[bestIndexTrain].reshape(1, -1)

    topK: int = min(5, xTrain.shape[0])
    topIndices: np.ndarray = np.argsort(yTrain)[-topK:]
    xTop: np.ndarray = xTrain[topIndices]

    centers = np.vstack((xBest, xTop))

    candidates: np.ndarray = buildCandidates(
        dimension=dimension,
        xCenters=centers,
        totalCount=candidateCount,
        localFraction=localFraction,
        localStd=localStd
    )

    yMean: float = float(np.mean(yTrain))
    yStd: float = float(np.std(yTrain))
    if yStd < 1e-12:
        yStd = 1.0
    yTrainForGp: np.ndarray = (yTrain - yMean) / yStd

    lengthScale: float = 0.18 if functionIndex == 2 else baseLengthScale
    noiseLevel: float = 1e-6 if functionIndex == 2 else baseNoiseLevel

    meanNorm, varianceNorm = gpPosterior(
        xTrain=xTrain,
        yTrain=yTrainForGp,
        xTest=candidates,
        lengthScale=lengthScale,
        noiseLevel=noiseLevel
    )

    sigmaNorm: np.ndarray = np.sqrt(varianceNorm)
    bestObservedNorm: float = float(np.max(yTrainForGp))

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
        raise ValueError("Invalid acquisition type")

    bestIndex: int = int(np.argmax(score))
    suggestion: np.ndarray = candidates[bestIndex]

    meanOrig: float = float(meanNorm[bestIndex] * yStd + yMean)
    sigmaOrig: float = float(sigmaNorm[bestIndex] * yStd)

    info: Dict[str, float | str] = {
        "acquisition": acq,
        "bestSoFar": float(np.max(yTrain)),
        "mean": meanOrig,
        "sigma": sigmaOrig
    }

    return suggestion, info


def main() -> None:
    print(f"Data root folder: {dataRootFolder}\n")

    for functionIndex in range(1, 9):
        functionFolder: Path = dataRootFolder / f"function_{functionIndex}"
        xTrain, yTrain = loadInitialData(functionFolder)

        cfg: Dict[str, float | str] = pickConfigForFunction(functionIndex)

        suggestion, info = suggestNextPoint(
            functionIndex=functionIndex,
            xTrain=xTrain,
            yTrain=yTrain,
            acquisitionType=str(cfg["acq"]),
            kappa=float(cfg["kappa"]),
            xi=float(cfg["xi"]),
            localFraction=float(cfg["localFrac"]),
            localStd=float(cfg["localStd"])
        )

        portalString: str = formatPortalInput(suggestion)

        print(f"Function {functionIndex}")
        print(f"  Acquisition  : {info['acquisition']}")
        print(f"  Best Y so far : {info['bestSoFar']}")
        print(f"  Suggestion    : {portalString}")
        print(f"  mu / sigma    : {info['mean']} / {info['sigma']}\n")

    print("Done.")
    print("Copy only the 'Suggestion' lines into the portal (no spaces).")


if __name__ == "__main__":
    main()
