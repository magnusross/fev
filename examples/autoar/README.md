# AutoAR for FEV

AutoAR ([Xu et al., 2024](https://github.com/Zongzhe-Xu/AutoAR)) is a supervised autoregressive baseline that uses KPSS stationarity testing, BIC-based lag selection, and OLS for coefficient estimation. This example adapts AutoAR to the FEV evaluation framework.

## Setup

```bash
uv sync
uv run python evaluate_model.py
```

The first run automatically clones the AutoAR repository into `AutoAR/`.

## Implementation decisions

### Lag-selection HPO runs only on the first window

The original AutoAR repo runs lag selection fresh for each experiment (one fixed train/val/test split per dataset). In FEV, evaluation uses rolling windows, so running the full BIC search on every window would be very slow.

This implementation runs `fit_raw` (the HPO search) only on the **first evaluation window**, then reuses the selected `best_lags` for all subsequent windows via `fit_preset`. This closely matches the spirit of the paper — the model does not have access to future evaluation windows when the lags are chosen — while keeping evaluation practical.

### Memory-bounded training data

AutoAR's internal `_unfold_df` materialises an `[N, num_rolling_windows, window_size]` float64 tensor. For large datasets (e.g. Traffic: 862 series × 17,000 timesteps × 536 window size ≈ 63 GB) this causes OOM.

This implementation caps the number of rows passed to `fit_raw`/`fit_preset` and `test_loss_acc_df` using:

```python
_MAX_ELEMENTS = 50_000_000  # ~400 MB per float64 tensor
```

Only the **training DataFrames** are truncated; the test context (`past_data[-effective_input_length:]`) used for the actual forecast is always the full available history. Evaluation targets are never touched, so results remain comparable across models.

### Adaptive `effective_input_length`

The paper uses a fixed `input_length = 512`. FEV datasets can have much shorter series. On the first window, `effective_input_length` is computed as:

```python
effective_input_length = max(2, min(input_length, T - horizon - 1))
```

and then fixed for all subsequent windows so the model configuration is consistent across windows.

### Lag candidates filtered to valid range

AutoAR internally limits maximum lags to `input_length - numdiff`. Candidates that exceed this are silently invalid. This implementation pre-filters the search space:

```python
valid_lags = [lag for lag in search_space["window_len"] if 0 < lag <= max_lag]
```

and falls back to `[max(1, max_lag)]` if no candidates fit (e.g. very short series).

### Short-series fallback

If the available history is shorter than `effective_input_length + horizon`, there are no rolling windows for HPO validation. In this case the smallest lag candidate is used directly, skipping `fit_raw`.

### Default search space and metric

- **Search space**: `[192, 128, 96, 64]` (the non-zero-shot candidates from the paper).
- **Metric**: BIC (`new_metric=True`) rather than raw MSE, as recommended in the paper.

### FEV data format conversion

FEV provides data in long-format HuggingFace datasets (`unique_id | ds | y`). AutoAR expects a wide `[time_steps, num_series]` DataFrame. The conversion:

1. Pivots on `unique_id` (sorted alphabetically to match FEV's expected column order).
2. Forward-fills NaN within each column (handles gaps in irregular series).
3. Fills any remaining NaN with the column mean (handles series that start later than others).

### Quantile predictions

AutoAR is a point-forecast model. FEV tasks may request quantile levels. All quantile levels are set equal to the point forecast.

### KPSS logic

The original `autoar.py` uses a `while numdiff < 3` loop with an inner `for series` loop and `break`/`continue` control flow. This implementation uses an equivalent `for numdiff in range(3)` structure that is slightly easier to follow, with the same result: return the first differencing order (0, 1, or 2) for which all columns pass the KPSS stationarity test at the 5% level.
