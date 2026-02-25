"""Run baseline vs. improved AutoAR on the dev benchmark and print a side-by-side comparison."""
import sys
import traceback
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_model import AutoARModel

import fev

BENCHMARK_PATH = str(Path(__file__).parent.parent.parent / "benchmarks/autoar_dev/tasks.yaml")

CONFIGS = {
    "baseline":     dict(standardize=False, use_seasonal_lags=False, search_space={"window_len": [192, 128, 96, 64]}),
    "std_only":     dict(standardize=True,  use_seasonal_lags=False, search_space={"window_len": [192, 128, 96, 64]}),
    "std+seasonal": dict(standardize=True,  use_seasonal_lags=True,  search_space={"window_len": [192, 128, 96, 64]}),
}


def run_config(config_name: str, model_kwargs: dict) -> pd.DataFrame:
    benchmark = fev.Benchmark.from_yaml(BENCHMARK_PATH)
    rows = []
    for task in tqdm(benchmark.tasks, desc=config_name):
        try:
            model = AutoARModel(**model_kwargs)
            preds, train_t, inf_t, extra = model.fit_predict(task)
            summary = task.evaluation_summary(
                preds, model_name=config_name,
                training_time_s=train_t, inference_time_s=inf_t, extra_info=extra,
            )
            rows.append(summary)
            cfg = task.dataset_config
            mase = summary.get("MASE", "N/A")
            print(f"  {cfg:<30} MASE={mase:.4f}  lags={extra.get('best_lags')}  "
                  f"eff_len={extra.get('effective_input_length')}")
        except Exception as e:
            print(f"  ERROR {task.dataset_config}: {e}")
            traceback.print_exc()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = {}
    for name, kwargs in CONFIGS.items():
        print(f"\n{'='*60}\n  Config: {name}\n{'='*60}")
        results[name] = run_config(name, kwargs)
        results[name].to_csv(f"results_{name}.csv", index=False)

    # Side-by-side comparison
    print("\n\n" + "="*80)
    print("COMPARISON  (MASE — lower is better)")
    print("="*80)
    dfs = {k: v.set_index("dataset_config")["MASE"] for k, v in results.items() if not v.empty}
    combined = pd.DataFrame(dfs)
    for col in list(combined.columns)[1:]:
        combined[f"Δ_{col}"] = (
            (combined[col] - combined[list(combined.columns)[0]])
            / combined[list(combined.columns)[0]] * 100
        ).round(1)
    print(combined.to_string(float_format="{:.4f}".format))
    print("\nMean MASE per config:")
    for col in dfs:
        print(f"  {col:<20}: {dfs[col].mean():.4f}")
