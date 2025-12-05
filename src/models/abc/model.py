from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):

    @abstractmethod
    def find_initial_params(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, close: np.ndarray, r: float
    ) -> None:
        """
        Performs the initial calibration of the model with initial interest rate r.

        Parameters
        ----------
        S : Underlying Prices of shape (n,)
        K : Strikes of shape (n,)
        T : Time To Maturities of shape (n,)
        is_call : The option type flag - Put or Call of shape (n,)
        close : Actual Close prices of shape (n,)
        r : Current Interest Rate
        """
        raise NotImplementedError

    @abstractmethod
    def calibrate(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, close: np.ndarray, r: float
    ) -> None:
        """
        Calibrate model parameters with current interest rate r.

        Parameters
        ----------
        S : Underlying Prices of shape (n,)
        K : Strikes of shape (n,)
        T : Time To Maturities of shape (n,)
        is_call : The option type flag - Put or Call of shape (n,)
        close : Actual Close prices of shape (n,)
        r : Current Interest Rate
        """
        raise NotImplementedError

    @abstractmethod
    def price(self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, r: float) -> np.ndarray:
        """
        Evaluates options with current interest rate r.

        Parameters
        ----------
        S : Underlying Prices of shape (n,)
        K : Strikes of shape (n,)
        T : Time To Maturities of shape (n,)
        is_call : The option type flag - Put or Call of shape (n,)
        r : Current Interest Rate

        Returns an array of shape (n,) with prices predicted by a model
        """
        raise NotImplementedError
