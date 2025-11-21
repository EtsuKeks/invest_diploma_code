from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np
import pandas as pd

from src.models.model import Model


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

    def add_nan_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for tag, _ in self.running_pairs.items():
            df[tag] = np.nan
        return df

    def find_initial_params(self, df: pd.DataFrame) -> None:
        for _, model in self.running_pairs.items():
            model.find_initial_params(df)

    def calibrate(self, df: pd.DataFrame) -> None:
        for _, model in self.running_pairs.items():
            model.calibrate(df)

    def price(self, df: pd.DataFrame) -> pd.DataFrame:
        for tag, model in self.running_pairs.items():
            df[tag] = model.price(df)
        return df
