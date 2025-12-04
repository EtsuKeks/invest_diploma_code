import pandas as pd
from tqdm import tqdm

from src.runners.bs.bs_finetuned import BlackSholesFinetunedRunner as Runner

# from src.runners.bs.bs_calibration_accuracy_check import BlackSholesCalibrationAccuracyCheckRunner as Runner
# from src.runners.bs.bs_initial_accuracy_check import BlackSholesInitialAccuracyCheckRunner as Runner
from src.utils.config import INPUTS_DIR, OUTPUTS_DIR, settings


def run_pipeline():
    df = pd.read_csv(INPUTS_DIR / settings.ppl.input_csv).reset_index(drop=True)
    groups = [g for _, g in df.groupby("current_time")]
    results, runner = [], Runner()
    runner.find_initial_params(groups[0], settings.ppl.risk_free_rate)

    with tqdm(total=settings.ppl.total_hours - 1, desc="Overall progress") as pbar:
        for i in range(settings.ppl.total_hours - 1):
            cur, nxt = groups[i], groups[i + 1]

            runner.calibrate(cur, settings.ppl.risk_free_rate)
            results.append(runner.price(nxt, settings.ppl.risk_free_rate))

            pbar.update(1)

    if settings.ppl.output_csv is None:
        raise ValueError("output_csv must be set before running the pipeline")

    pd.concat(results).to_csv(OUTPUTS_DIR / settings.ppl.output_csv)


if __name__ == "__main__":
    run_pipeline()
