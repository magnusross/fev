"""Diagnostic script: inspect what AutoAR sees per task without running the full model."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_model import _ensure_autoar, _determine_numdiff_kpss, _past_data_to_wide

_ensure_autoar()

import fev

BENCHMARK = fev.Benchmark.from_yaml(
    str(Path(__file__).parent.parent.parent / "benchmarks/autoar_dev/tasks.yaml")
)

errors = []

for task in BENCHMARK.tasks:
    try:
        task.load_full_dataset(trust_remote_code=True)
    except Exception as e:
        print(f"\nFailed to load {task.dataset_config}: {e}")
        continue
    task_name = task.dataset_config
    first_window = next(iter(task.iter_windows()))
    horizon = first_window.horizon

    past_df, _future_df, _static_df = fev.convert_input_data(
        first_window, adapter="nixtla", as_univariate=True
    )
    past_data, col_order = _past_data_to_wide(past_df)
    T, N = past_data.shape

    input_length = 512
    effective_input_length = max(2, min(input_length, T - horizon - 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numdiff = _determine_numdiff_kpss(past_data)

    # Check how much real data each series has (non-mean rows)
    col_means = past_data.mean(axis=0)
    nan_counts = []
    for c in range(N):
        # proxy: count rows that equal the column mean (filled rows)
        nan_counts.append(np.sum(np.isclose(past_data[:, c], col_means[c])))

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"  T={T}, N={N}, horizon={horizon}")
    print(f"  effective_input_length={effective_input_length}")
    print(f"  numdiff={numdiff}")
    print(f"  data range: [{past_data.min():.3f}, {past_data.max():.3f}]")
    print(f"  NaN-filled rows per series (proxy): min={min(nan_counts)}, "
          f"max={max(nan_counts)}, median={np.median(nan_counts):.0f}")
    # Check if data has strong trend (mean of abs first diff)
    diffs = np.diff(past_data, axis=0)
    print(f"  mean |first diff|: {np.abs(diffs).mean():.4f}")
    print(f"  std of series: {past_data.std(axis=0).mean():.4f}")
