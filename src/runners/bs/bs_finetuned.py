from typing import Mapping

import pandas as pd

from src.models.black_sholes import BlackScholes
from src.models.model import Model
from src.runners.runner import Runner
from src.utils.config import OUTPUTS_DIR, settings

OUTPUT_CSV = "bs_finetuned.csv"


class BlackSholesFinetunedRunner(Runner):
    def __init__(self):
        settings.ppl.output_csv = OUTPUT_CSV
        self._running_pairs = {
            "sigma": BlackScholes(settings),
        }

    @property
    def running_pairs(self) -> Mapping[str, Model]:
        return self._running_pairs

    def price(self, df):
        for tag, model in self._running_pairs.items():
            df[tag] = model.sigma
            df[tag.replace("sigma", "close_bs")] = model.price(df)
        return df


# Run with "python -m src.runners.bs.bs_finetuned"
if __name__ == "__main__":
    df = pd.read_csv(OUTPUTS_DIR / OUTPUT_CSV, index_col=0, parse_dates=["current_time"], infer_datetime_format=True)
    df_sigma = df.groupby("current_time").first().reset_index()
    print("Max delta of sigma:", df_sigma["sigma"].diff().abs().max())
