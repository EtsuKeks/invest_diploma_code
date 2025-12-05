from abc import abstractmethod

import numpy as np

from src.models.abc.model import Model


class SpotCautiousModel(Model):

    @abstractmethod
    def calibrate_spot_cautious_params(self, S: np.ndarray, r: float) -> None:
        """
        Performs the calibration of spot-cautious parameters of the model with current interest rate r.

        Parameters
        ----------
        S : Underlying Prices of shape (n,)
        r : Current Interest Rate
        """
        raise NotImplementedError
