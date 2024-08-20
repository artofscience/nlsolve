from abc import ABC, abstractmethod
from typing import List

import numpy as np

from point import Point
from structure import Structure


class Constraint(ABC):
    def __init__(self, nonlinear_function: Structure) -> None:
        self.nlf = nonlinear_function
        self.ff = self.nlf.external_load()
        self.up = self.nlf.prescribed_motion()
        self.ff2 = np.dot(self.ff, self.ff) if self.ff is not None else None
        self.up2 = np.dot(self.up, self.up) if self.up is not None else None
        self.nf = np.shape(self.ff)[0] if self.ff is not None else None
        self.np = np.shape(self.up)[0] if self.up is not None else None
        self.kpp = self.nlf.tangent_stiffness_prescribed_prescribed
        self.kpf = self.nlf.tangent_stiffness_prescribed_free
        self.kfp = self.nlf.tangent_stiffness_free_prescribed
        self.rp = self.nlf.residual_prescribed

    @abstractmethod
    def predictor(self, p: Point, dp: Point, ddx: np.ndarray, dl: float, sol: List[Point]) -> Point:
        pass

    @abstractmethod
    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> Point:
        pass
