from abc import ABC

import numpy as np
from structure import Structure
from point import Point
from typing import List


class Constraint(ABC):
    def __init__(self, nonlinear_function: Structure) -> None:
        self.a = nonlinear_function
        self.f = self.a.external_load()
        self.v = self.a.prescribed_motion()
        self.f2 = np.dot(self.f, self.f) if self.f is not None else None
        self.v2 = np.dot(self.v, self.v) if self.v is not None else None
        self.nf = np.shape(self.f)[0] if self.f is not None else None
        self.np = np.shape(self.v)[0] if self.v is not None else None
        self.Kpp = self.a.tangent_stiffness_prescribed_prescribed
        self.Kpf = self.a.tangent_stiffness_prescribed_free
        self.Kfp = self.a.tangent_stiffness_free_prescribed
        self.rp = self.a.residual_prescribed

    def predictor(self, p: Point, dp: Point, ddx: np.ndarray, dl: float, sol: List[Point]) -> Point:
        ...

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> Point:
        ...
