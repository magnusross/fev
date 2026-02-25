"""AutoAR: Supervised autoregressive baseline with automatic differencing and lag selection.

From the paper "Specialized Foundation Models Struggle to Beat Supervised Baselines"
(https://github.com/Zongzhe-Xu/AutoAR).

Design notes:
- KPSS stationarity testing determines the differencing order (0–2)
- BIC-based lag selection via a small hyperparameter search on the first window
- OLS for efficient coefficient estimation (global model across all series)
- Lag selection reuses best_lags from first window for all subsequent windows
"""

import subprocess
import sys
import time
import warnings
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fev

datasets.disable_progress_bars()

AUTOAR_DIR = Path(__file__).parent / "AutoAR"


def _ensure_autoar() -> None:
    """Clone AutoAR repo if not present and insert into sys.path."""
    if not AUTOAR_DIR.exists():
        print(f"Cloning AutoAR into {AUTOAR_DIR}...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/Zongzhe-Xu/AutoAR.git",
                str(AUTOAR_DIR),
            ],
            check=True,
        )
    if str(AUTOAR_DIR) not in sys.path:
        sys.path.insert(0, str(AUTOAR_DIR))


_ensure_autoar()

from models import AR_diff  # noqa: E402  (available after _ensure_autoar)


def _determine_numdiff_kpss(data: np.ndarray) -> int:
    """Return differencing order (0, 1, or 2) using column-wise KPSS tests.

    Mirrors the KPSS logic in autoar.py: try numdiff = 0, 1, 2 in order and
    return the first value for which all columns pass the stationarity test.

    Parameters
    ----------
    data:
        Array of shape [time_steps, num_series].
    """
    from statsmodels.tsa.stattools import kpss

    for numdiff in range(3):
        differenced = np.diff(data, n=numdiff, axis=0) if numdiff > 0 else data
        all_stable = True
        for i in range(differenced.shape[1]):
            col = differenced[:, i]
            if len(col) < 10:
                # Too short for a reliable KPSS test; assume stable
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, pvalue, _, _ = kpss(col, regression="ct")
                    if pvalue <= 0.05:
                        all_stable = False
                        break
                except Exception:
                    all_stable = False
                    break
        if all_stable:
            return numdiff

    return 2


def _past_data_to_wide(past_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Pivot a long-format FEV DataFrame to a [time_steps, num_series] numpy array.

    Columns are ordered by sorted unique_id (matching the order expected by
    ``fev.utils.combine_univariate_predictions_to_multivariate``).

    Missing values are forward-filled within each column, then filled with the
    column mean for any remaining NaN (e.g. series that start later than others).

    Returns
    -------
    data:
        Float array of shape [time_steps, num_series].
    col_order:
        Sorted unique_id values corresponding to each column of ``data``.
    """
    wide_df = past_df.pivot(index="ds", columns="unique_id", values="y")
    wide_df = wide_df.sort_index()
    wide_df = wide_df.ffill()
    wide_df = wide_df.fillna(wide_df.mean())
    return wide_df.to_numpy().astype(float), list(wide_df.columns)


class AutoARModel:
    """AutoAR: global AR model with KPSS differencing and BIC lag selection.

    Parameters
    ----------
    input_length:
        Maximum context window size. Will be reduced automatically if the
        available past data is shorter than this value.
    time_limit_hours:
        Wall-clock time budget for the lag-selection HPO step (only run on the
        first evaluation window).
    use_kpss:
        Use KPSS tests to determine the differencing order.  Set to False to
        always apply one round of differencing (``numdiff=1``).
    use_ols:
        Use the fast closed-form OLS solver instead of statsmodels AutoReg.
    search_space:
        Lag candidates to evaluate during HPO.  Defaults to ``[192, 128, 96, 64]``
        (the non-zero-shot candidates from the AutoAR paper).
    new_metric:
        Use BIC instead of raw MSE when ranking lag candidates (recommended).
    """

    def __init__(
        self,
        input_length: int = 512,
        time_limit_hours: float = 0.5,
        use_kpss: bool = True,
        use_ols: bool = True,
        search_space: dict | None = None,
        new_metric: bool = True,
    ):
        self.input_length = input_length
        self.time_limit_hours = time_limit_hours
        self.use_kpss = use_kpss
        self.use_ols = use_ols
        self.search_space = search_space or {"window_len": [192, 128, 96, 64]}
        self.new_metric = new_metric

    def _get_device(self):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _predict_window(
        self,
        window: fev.EvaluationWindow,
        task: fev.Task,
        best_lags: int | None = None,
        effective_input_length: int | None = None,
    ) -> tuple[datasets.DatasetDict, float, float, int | None, int]:
        """Fit and predict for a single evaluation window.

        When ``best_lags`` is None (first call), runs the HPO lag-selection step
        via ``fit_raw`` and caches the result.  On subsequent calls, ``fit_preset``
        is called directly with the cached lags.

        Returns
        -------
        predictions:
            DatasetDict in FEV format.
        train_time:
            Seconds spent on model fitting.
        inf_time:
            Seconds spent on inference.
        best_lags:
            Lag count selected by HPO (or reused from cache).
        effective_input_length:
            Actual ``input_length`` used for this task.
        """
        device = self._get_device()
        horizon = window.horizon

        # ── 1. Convert past data to wide format ──────────────────────────────
        past_df, _future_df, _static_df = fev.convert_input_data(window, adapter="nixtla", as_univariate=True)
        past_data, _col_order = _past_data_to_wide(past_df)
        T, N = past_data.shape

        # ── 2. Determine effective input_length (fixed for all windows) ───────
        if effective_input_length is None:
            effective_input_length = max(2, min(self.input_length, T - horizon - 1))

        # ── 3. Stationarity test ──────────────────────────────────────────────
        if self.use_kpss:
            numdiff = _determine_numdiff_kpss(past_data)
        else:
            numdiff = 1
        do_diff = numdiff > 0

        # ── 4. Prepare training DataFrames ────────────────────────────────────
        # fit_raw expects:
        #   scaled_train_df = differenced data  [T - numdiff, N]
        #   scaled_val_df   = raw data          [T, N]
        # fit_preset expects:
        #   scaled_train_df = differenced data
        if numdiff > 0:
            train_arr_diff = np.diff(past_data, n=numdiff, axis=0)
        else:
            train_arr_diff = past_data

        # ── 5. Determine valid lag candidates ─────────────────────────────────
        # AutoAR's internal limit: max_lags = input_length - numdiff
        max_lag = effective_input_length - numdiff
        valid_lags = sorted(
            [lag for lag in self.search_space["window_len"] if 0 < lag <= max_lag],
            reverse=True,
        )
        if not valid_lags:
            valid_lags = [max(1, max_lag)]
        search_space = {"window_len": valid_lags}

        # ── 5b. Truncate DataFrames to bound memory usage ─────────────────────
        # _unfold_df materialises an [N, num_windows, win_size] float64 tensor.
        # For large N or long T this blows up (e.g. Traffic: 862 × 17 000 × 536
        # ≈ 63 GB).  We cap the number of rolling windows to keep each tensor
        # below _MAX_UNFOLD_ELEMENTS elements (~400 MB at float64).
        _MAX_ELEMENTS = 50_000_000  # ~400 MB per tensor

        win_size = effective_input_length + horizon
        max_val_windows = max(10, _MAX_ELEMENTS // max(1, N * win_size))
        max_val_rows = min(T, win_size + max_val_windows - 1)

        max_train_windows = max(10, _MAX_ELEMENTS // max(1, N * (max_lag + 1)))
        max_train_rows = min(len(train_arr_diff), max_lag + max_train_windows)

        train_df_diff = pd.DataFrame(train_arr_diff[-max_train_rows:])
        train_df_raw_val = pd.DataFrame(past_data[-max_val_rows:])

        # ── 6. Fit ────────────────────────────────────────────────────────────
        ar = AR_diff(effective_input_length, horizon)

        train_start = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if best_lags is None:
                # First window: run HPO if we have enough data for ≥1 val window
                if T >= effective_input_length + horizon:
                    ar.fit_raw(
                        train_df_diff,
                        train_df_raw_val,
                        device,
                        self.time_limit_hours,
                        do_diff,
                        numdiff,
                        use_ols=self.use_ols,
                        first_lag=0,
                        search_space=search_space,
                        new_metric=self.new_metric,
                    )
                    best_lags = ar.best_lags
                else:
                    # Series too short for HPO; fall back to smallest candidate
                    best_lags = valid_lags[-1]

            ar.fit_preset(
                train_df_diff,
                best_lags,
                do_diff,
                device,
                numdiff=numdiff,
                use_ols=self.use_ols,
            )
        train_time = time.monotonic() - train_start

        # ── 7. Build test DataFrame ────────────────────────────────────────────
        # test_loss_acc_df internally applies differencing via torch.diff, so we
        # pass raw data.  The function expects [input_length + horizon, N] rows;
        # the last `horizon` rows act as placeholder "targets" (zeros work fine
        # because we only use the returned predictions tensor, not the MSE).
        context = past_data[-effective_input_length:]
        if len(context) < effective_input_length:
            pad_len = effective_input_length - len(context)
            col_means = past_data.mean(axis=0, keepdims=True)
            context = np.vstack([np.tile(col_means, (pad_len, 1)), context])
        test_arr = np.vstack([context, np.zeros((horizon, N))])
        test_df = pd.DataFrame(test_arr)

        # ── 8. Predict ────────────────────────────────────────────────────────
        inf_start = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mse, mae, preds_tensor, _ = ar.test_loss_acc_df(test_df, device, numdiff=numdiff, return_prediction=True)
        inf_time = time.monotonic() - inf_start

        # preds_tensor shape: [num_series, num_windows=1, horizon]
        preds = preds_tensor[:, 0, :].detach().cpu().numpy()  # [N, horizon]

        # ── 9. Format for FEV ─────────────────────────────────────────────────
        # Each row i of preds corresponds to col_order[i] (sorted unique_id).
        # combine_univariate_predictions_to_multivariate expects predictions
        # cycled through target columns: [item1_col1, item1_col2, item2_col1, ...]
        # which matches alphabetical unique_id order. ✓
        quantile_levels = task.quantile_levels
        predictions_list = []
        for i in range(N):
            pred_values = preds[i].tolist()
            pred_dict = {"predictions": pred_values}
            for q in quantile_levels:
                pred_dict[str(q)] = pred_values
            predictions_list.append(pred_dict)

        result_ds = fev.utils.combine_univariate_predictions_to_multivariate(
            predictions_list, target_columns=window.target_columns
        )
        return result_ds, train_time, inf_time, best_lags, effective_input_length

    def fit_predict(self, task: fev.Task) -> tuple[list[datasets.DatasetDict], float, float, dict]:
        """Fit and predict on all evaluation windows of a task.

        Lag selection (HPO) is run only on the first window; subsequent windows
        reuse the selected lags for faster evaluation.

        Returns
        -------
        predictions:
            List of DatasetDicts, one per evaluation window.
        training_time:
            Total time spent on fitting (seconds).
        inference_time:
            Total time spent on predicting (seconds).
        extra_info:
            Dictionary with model metadata.
        """
        task.load_full_dataset()

        predictions = []
        total_train_time = 0.0
        total_inf_time = 0.0
        best_lags: int | None = None
        effective_input_length: int | None = None

        for window in task.iter_windows():
            preds, train_time, inf_time, best_lags, effective_input_length = self._predict_window(
                window,
                task,
                best_lags=best_lags,
                effective_input_length=effective_input_length,
            )
            predictions.append(preds)
            total_train_time += train_time
            total_inf_time += inf_time

        extra_info = {
            "best_lags": best_lags,
            "effective_input_length": effective_input_length,
            "use_kpss": self.use_kpss,
            "use_ols": self.use_ols,
            "new_metric": self.new_metric,
        }
        return predictions, total_train_time, total_inf_time, extra_info


if __name__ == "__main__":
    model_name = "autoar"

    # benchmark = fev.Benchmark.from_yaml("../../benchmarks/example/tasks.yaml")
    benchmark = fev.Benchmark.from_yaml(
        "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/tasks.yaml"
    )

    summaries = []
    for task in tqdm(benchmark.tasks):
        model = AutoARModel()
        predictions, training_time, inference_time, extra_info = model.fit_predict(task)
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name=model_name,
            training_time_s=training_time,
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        print(evaluation_summary)
        summaries.append(evaluation_summary)

    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv(f"{model_name}.csv", index=False)
