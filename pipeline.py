import pandas as pd
from tqdm import tqdm
from src.black_sholes.model import BlackScholes
from src.utils.helpers import add_nan_columns
from src.utils.config import settings, INPUTS_DIR, OUTPUTS_DIR


def run_pipeline():
    df = pd.read_csv(INPUTS_DIR / settings.ppl.input_csv).reset_index(drop=True)
    groups = [g for _, g in df.groupby("current_time")]
    results, model_bs = [], None

    with tqdm(total=max(0, len(groups) - 1), desc="Overall progress") as pbar:
        for i in range(len(groups) - 1):
            current, nxt = groups[i], add_nan_columns(groups[i + 1])

            if model_bs is None:
                model_bs = BlackScholes()
                model_bs.find_initial_params(current)
            else:
                model_bs.calibrate(current)

            nxt["close_bs"] = model_bs.price(nxt)

            results.append(nxt)
            pbar.update(1)

    pd.concat(results).to_csv(OUTPUTS_DIR / settings.ppl.output_csv)


if __name__ == "__main__":
    run_pipeline()
