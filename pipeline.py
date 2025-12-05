import numpy as np
import pandas as pd
from tqdm import tqdm

from src.runners.bs.bs_finetuned import BlackSholesFinetunedRunner as Runner
from src.utils.config import INPUTS_DIR, OUTPUTS_DIR, settings


def prepare_spot_groups(spot_data, df) -> list[np.ndarray]:
    unique_times = df["current_time"].unique()
    unique_times.sort()
    spot_groups = []
    target_ts, spot_ts = pd.to_datetime(unique_times).astype(np.int64) // 10**6, spot_data[:, 0]
    for start_ms in target_ts:
        idx_start = np.searchsorted(spot_ts, start_ms, side="left")
        idx_end = np.searchsorted(spot_ts, start_ms + 3600 * 1000, side="right")
        spot_groups.append(spot_data[idx_start:idx_end, 1])

    return spot_groups


def run_pipeline():
    df = pd.read_csv(INPUTS_DIR / settings.ppl.input_csv).reset_index(drop=True)
    if not (INPUTS_DIR / settings.ppl.spot_npy).is_file():
        raise ValueError("""spot_npy is abscent - probably, parse_spot.ipynb wasnt used, check
                         notebooks/parse_spot.ipynb for details""")

    spot_full = np.load(INPUTS_DIR / settings.ppl.spot_npy)
    groups, spot_groups = [g for _, g in df.groupby("current_time")], prepare_spot_groups(spot_full, df)
    results, runner = [], Runner()
    runner.find_initial_params(groups[0], spot_groups[0], settings.ppl.risk_free_rate)

    with tqdm(total=settings.ppl.total_hours - 1, desc="Overall progress") as pbar:
        for i in range(settings.ppl.total_hours - 1):
            cur, nxt, cur_spot = groups[i], groups[i + 1], spot_groups[i]

            runner.calibrate(cur, cur_spot, settings.ppl.risk_free_rate)
            results.append(runner.price(nxt, settings.ppl.risk_free_rate))

            pbar.update(1)

    if settings.ppl.output_csv is None:
        raise ValueError("output_csv must be set before running the pipeline")

    pd.concat(results).to_csv(OUTPUTS_DIR / settings.ppl.output_csv)


if __name__ == "__main__":
    run_pipeline()
