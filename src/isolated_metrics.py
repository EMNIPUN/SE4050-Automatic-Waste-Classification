"""Run sklearn metrics without importing sklearn into the notebook kernel.

Useful when a live notebook has cached modules from a replaced installation.
Only small label/probability arrays are sent to a fresh Python process; the
notebook's TensorFlow models, training history, and variables stay in memory.
"""

from functools import partial
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory


_METRICS = (
    "accuracy_score", "precision_score", "recall_score", "f1_score",
    "roc_auc_score", "confusion_matrix", "classification_report",
)


def _call_metric(name, *args, **kwargs):
    with TemporaryDirectory(prefix="waste-metrics-") as directory:
        request = Path(directory) / "request.pkl"
        response = Path(directory) / "response.pkl"
        # These private temporary files contain only data from this invocation.
        with request.open("wb") as stream:
            pickle.dump((name, args, kwargs), stream)
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()),
             str(request), str(response)],
            capture_output=True, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise RuntimeError(
                f"Metric {name} failed in the separate Python process:\n"
                f"{result.stderr or result.stdout}"
            )
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        with response.open("rb") as stream:
            return pickle.load(stream)


for _name in _METRICS:
    globals()[_name] = partial(_call_metric, _name)


if __name__ == "__main__":
    from sklearn import metrics

    with open(sys.argv[1], "rb") as stream:
        name, args, kwargs = pickle.load(stream)
    if name not in _METRICS:
        raise ValueError(f"Unsupported metric: {name}")
    value = getattr(metrics, name)(*args, **kwargs)
    with open(sys.argv[2], "wb") as stream:
        pickle.dump(value, stream)
