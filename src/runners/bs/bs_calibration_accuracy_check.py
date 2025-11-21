from typing import Mapping

import numpy as np
import pandas as pd

from src.models.black_sholes import BlackScholes
from src.models.model import Model
from src.runners.runner import Runner
from src.utils.config import OUTPUTS_DIR, settings

OUTPUT_CSV = "bs_calibration_accuracy.csv"


class BlackSholesCalibrationAccuracyCheckRunner(Runner):
    def __init__(self):
        settings.ppl.output_csv = "bs_calibration_accuracy.csv"
        settings.ppl.total_hours = 100

        settings_accurate = settings.model_copy(deep=True)
        settings_accurate.bs.points_calibrate = 100000

        settings_inaccurate = settings.model_copy(deep=True)
        settings_inaccurate.bs.points_calibrate = 1000

        self._running_pairs = {
            "sigma_default": BlackScholes(settings),
            "sigma_accurate": BlackScholes(settings_accurate),
            "sigma_inaccurate": BlackScholes(settings_inaccurate),
        }

    @property
    def running_pairs(self) -> Mapping[str, Model]:
        return self._running_pairs

    def price(self, df):
        for tag, model in self._running_pairs.items():
            df[tag] = model.sigma
            df[tag.replace("sigma_", "close_")] = model.price(df)
        return df


# Run with "python -m src.runners.bs.bs_calibration_accuracy_check"
if __name__ == "__main__":
    df = pd.read_csv(OUTPUTS_DIR / OUTPUT_CSV, index_col=0, parse_dates=["current_time"], infer_datetime_format=True)

    def rel_diff(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (b - a) / a
        return r

    df_sigma = df.groupby("current_time").first().reset_index()
    df_sigma["sigma_diff_acc"] = df_sigma["sigma_accurate"] - df_sigma["sigma_default"]
    df_sigma["sigma_diff_inacc"] = df_sigma["sigma_inaccurate"] - df_sigma["sigma_default"]
    df_sigma["sigma_rel_diff_acc"] = rel_diff(df_sigma["sigma_default"], df_sigma["sigma_accurate"])
    df_sigma["sigma_rel_diff_inacc"] = rel_diff(df_sigma["sigma_default"], df_sigma["sigma_inaccurate"])

    print("Max delta of sigma_default:", df_sigma["sigma_default"].diff().abs().max())
    print("Max delta of sigma_accurate:", df_sigma["sigma_accurate"].diff().abs().max())
    print("Max delta of sigma_inaccurate:", df_sigma["sigma_inaccurate"].diff().abs().max())

    df["close_diff_acc"] = df["close_accurate"] - df["close_default"]
    df["close_diff_inacc"] = df["close_inaccurate"] - df["close_default"]
    df["close_rel_diff_acc"] = rel_diff(df["close_default"], df["close_accurate"])
    df["close_rel_diff_inacc"] = rel_diff(df["close_default"], df["close_inaccurate"])

    def summary(series):
        s = series.dropna()
        return {
            "mean_abs": float(np.abs(s).mean()),
            "median_abs": float(np.abs(s).median()),
            "std": float(s.std()),
            "max_abs": float(np.abs(s).max()),
        }

    summary_table = {
        "sigma_diff_acc": summary(df_sigma["sigma_diff_acc"]),
        "sigma_diff_inacc": summary(df_sigma["sigma_diff_inacc"]),
        "sigma_rel_diff_acc": summary(df_sigma["sigma_rel_diff_acc"]),
        "sigma_rel_diff_inacc": summary(df_sigma["sigma_rel_diff_inacc"]),
        "close_diff_acc": summary(df["close_diff_acc"]),
        "close_diff_inacc": summary(df["close_diff_inacc"]),
        "close_rel_diff_acc": summary(df["close_rel_diff_acc"]),
        "close_rel_diff_inacc": summary(df["close_rel_diff_inacc"]),
    }

    summary_df = pd.DataFrame(summary_table).T
    print("\nSummary stats (absolute & relative diffs):")
    print(summary_df)

    def print_max_abs(series, df, label_prefix=""):
        s = series.abs()
        idx = s.idxmax()
        time = df.loc[idx, "current_time"]
        print(f"{label_prefix}: max abs = {s.loc[idx]} at index = {idx}" + f", current_time = {time}")

    print_max_abs(df_sigma["sigma_diff_acc"], df_sigma, "sigma_diff_acc")
    print_max_abs(df_sigma["sigma_diff_inacc"], df_sigma, "sigma_diff_inacc")
    print_max_abs(df["close_diff_acc"], df, "close_diff_acc")
    print_max_abs(df["close_diff_inacc"], df, "close_diff_inacc")
