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
        self.ff = self.nlf.external_load()
        self.up = self.nlf.prescribed_motion()
        self.kpp = self.nlf.tangent_stiffness_prescribed_prescribed
        self.kpf = self.nlf.tangent_stiffness_prescribed_free
        self.kfp = self.nlf.tangent_stiffness_free_prescribed
        self.rp = self.nlf.residual_prescribed

        # get dimension of free and prescribed degrees of freedom
        self.nf = np.shape(self.ff)[0] if self.ff is not None else None
        self.np = np.shape(self.up)[0] if self.up is not None else None

        # squared norm of load external load and prescribed motion
        self.ff2 = np.dot(self.ff, self.ff) if self.nf is not None else None
        self.up2 = np.dot(self.up, self.up) if self.np is not None else None

    @abstractmethod
    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> Point:
        """
        Determines the iterative load parameter for any corrector step (iterate i > 1) and updates the iterative load and motion accordingly.

        :param p: current state (load, motion and load parameter)
        :param dp: incremental state
        :param ddx: resultants of the solve following Batoz and Dhatt
        :param dl: characteristic constraint magnitude (e.g. arc-length)
        :return: updated iterative state
        """
        pass

    @abstractmethod
    def predictor(self, p: Point, dp: Point, ddx: np.ndarray, dl: float, sol: List[Point]) -> Point:
        """
        Determines the iterative load parameter for any predictor step (iterate i = 0 AND increment j > 0) and updates the iterative load and motion accordingly.

        :param p: current state (load, motion and load parameter)
        :param dp: incremental state
        :param ddx: resultants of the solve following Batoz and Dhatt
        :param dl: characteristic constraint magnitude (e.g. arc-length)
        :param sol: previous found equilibrium points
        :return: updated iterative state
        """
        pass


