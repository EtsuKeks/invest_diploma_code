from abc import ABC, abstractmethod
from typing import Mapping

import pandas as pd

from src.models.abc.model import Model


def split_df(df: pd.DataFrame):
    return (
        df["underlying_price"].values,
        df["strike"].values,
        df["ttm"].values,
        df["is_call"].values,
        df["close"].values,
    )


class Runner(ABC):
    """
    Runner represents a concrete test case. When initialized, changes global settings
    for each model individually and possibly changes their behaviour (i.e., passes
    self.find_initial_params as self.calibrate).
    """

    @property
    @abstractmethod
    def running_pairs(self) -> Mapping[str, Model]:
        """Return a mapping name -> Model."""
        raise NotImplementedError

    def find_initial_params(self, df: pd.DataFrame, r: float) -> None:
        S, K, T, is_call, close = split_df(df)
        for _, model in self.running_pairs.items():
            model.find_initial_params(S, K, T, is_call, close, r)

    def calibrate(self, df: pd.DataFrame, r: float) -> None:
        S, K, T, is_call, close = split_df(df)
        for _, model in self.running_pairs.items():
            model.calibrate(S, K, T, is_call, close, r)

    def price(self, df: pd.DataFrame, r: float) -> pd.DataFrame:
        S, K, T, is_call, _ = split_df(df)
        for tag, model in self.running_pairs.items():
            df[tag] = model.price(S, K, T, is_call, r)
        return df
