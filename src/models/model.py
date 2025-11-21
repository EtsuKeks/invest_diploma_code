from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):

    @abstractmethod
    def find_initial_params(self, df: pd.DataFrame) -> None:
        """Performs the initial calibration of the model."""
        pass

    @abstractmethod
    def calibrate(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def price(self, df: pd.DataFrame) -> np.ndarray:
        """Performs options based on data in df. Returns np.ndarray of prices."""
        pass
