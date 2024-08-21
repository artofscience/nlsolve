from abc import ABC, abstractmethod
from typing import List

import numpy as np

from point import Point
from structure import Structure


class Constraint(ABC):
    def __init__(self, nonlinear_function: Structure) -> None:
        """
        Initialization of the constraint function used to solve the system of nonlinear equations.

        :param nonlinear_function: system of nonlinear equations to solve
        """

        # some aliases for commonly used functions
        self.nlf = nonlinear_function

    @abstractmethod
    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> float:
        """
        Determines the iterative load parameter for any corrector step (iterate i > 1).

        :param p: current state (load, motion and load parameter)
        :param dp: incremental state
        :param ddx: resultants of the solve following Batoz and Dhatt
        :param dl: characteristic constraint magnitude (e.g. arc-length)
        :return: updated iterative state
        """
        pass

    @abstractmethod
    def predictor(self, p: Point, sol: List[Point], ddx: np.ndarray, dl: float) -> float:
        """
        Determines the iterative load parameter for any predictor step (iterate i = 0 AND increment j > 0).

        :param p: current state (load, motion and load parameter)
        :param ddx: resultants of the solve following Batoz and Dhatt
        :param dl: characteristic constraint magnitude (e.g. arc-length)
        :param sol: previous found equilibrium points
        :return: updated iterative state
        """
        pass


