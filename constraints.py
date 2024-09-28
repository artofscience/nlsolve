from __future__ import annotations

from typing import List

import numpy as np

from abc import ABC, abstractmethod

from utils import Structure, Point

from logger import CustomFormatter, create_logger
import logging

State = np.ndarray[float] | None

class DiscriminantError(Exception):
    pass

class Constraint(ABC):
    def __init__(self, dl: float = 0.1, name: str = None, logging_level: int = logging.DEBUG) -> None:
        """
        Initialization of the constraint function used to solve the system of nonlinear equations.

        :param nonlinear_function: system of nonlinear equations to solve
        """

        # some aliases for commonly used functions
        self.dl: float = dl # characteristic magnitude of constraint function (e.g. arc-length)

        self.__name__ = name if name is not None else (self.__class__.__name__ + " " + str(id(self)))

        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + self.__name__)

    @property
    def dl(self) -> float:
        return self._dl

    @dl.setter
    def dl(self, value: float) -> None:
        self._dl = value

    @abstractmethod
    def corrector(self, nlf: Structure, p: Point, dp: Point, ddx: np.ndarray) -> float:
        """
        Determines the iterative load parameter for any corrector step (iterate i > 1).

        :param nlf: system of nonlinear equations to solve
        :param p: current state (load, motion and load parameter)
        :param dp: incremental state
        :param ddx: resultants of the solve following Batoz and Dhatt
        :param dl: characteristic constraint magnitude (e.g. arc-length)
        :return: updated iterative state
        """
        pass

    @abstractmethod
    def predictor(self, nlf: Structure, p: Point, sol: List[Point], ddx: np.ndarray) -> float:
        """
        Determines the iterative load parameter for any predictor step (iterate i = 0 AND increment j > 0).

        :param nlf: system of nonlinear equations to solve
        :param p: current state (load, motion and load parameter)
        :param sol: previous found equilibrium points
        :param ddx: resultants of the solve following Batoz and Dhatt
        :param dl: characteristic constraint magnitude (e.g. arc-length)
        :return: updated iterative state
        """
        pass


class NewtonRaphson(Constraint):

    def corrector(self, nlf: Structure, p: Point, dp: Point, ddx: np.ndarray) -> float:
        return 0.0

    def predictor(self, nlf: Structure, p: Point, sol: List[Point], ddx: np.ndarray) -> Point:

        load = 0.0
        load += nlf.up2 if nlf.np else 0.0
        load += nlf.ff2 if nlf.nf else 0.0

        return self.dl / np.sqrt(load)


class ArcLength(Constraint):
    def __init__(self, dl: float = 0.1, name: str = None, logging_level: int = logging.DEBUG, beta: float = 1.0) -> None:
        super().__init__(dl, name, logging_level)
        self.beta = beta

    def predictor(self, nlf: Structure, p: Point, sol: List[Point], ddx: np.ndarray) -> float:
        y = self.get_roots_predictor(nlf, p, ddx, self.dl)
        cps = [nlf.ddp(p, ddx, i) for i in y]
        return self.select_root_predictor(nlf, p, sol, cps)

    def corrector(self, nlf: Structure, p: Point, dp: Point, ddx: np.ndarray) -> float:
        try:
            y = self.get_roots_corrector(nlf, p, dp, ddx, self.dl)
        except DiscriminantError:
            raise DiscriminantError
        cps = [nlf.ddp(p + dp, ddx, i) for i in y]
        return self.select_root_corrector(nlf, dp, cps)

    def get_roots_predictor(self, nlf: Structure, p: Point, u: np.ndarray, dl: float) -> np.ndarray:
        a = 0.0
        if nlf.nf:
            a += np.dot(u[:, 1], u[:, 1]) + self.beta ** 2 * nlf.ff2
        if nlf.np:
            tmpa = nlf.kpp(p) @ nlf.up
            if nlf.nf:
                tmpa += nlf.kpf(p) @ u[:, 1]
            a += self.beta ** 2 * np.dot(tmpa, tmpa) + nlf.up2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, nlf: Structure, p: Point, dp: Point, u: np.ndarray, dl: float) -> np.ndarray:
        a = np.zeros(3)

        a[2] -= dl ** 2
        if nlf.nf:
            a[0] += np.dot(u[:, 1], u[:, 1])
            a[0] += self.beta ** 2 * nlf.ff2
            a[1] += 2 * np.dot(u[:, 1], dp.uf + u[:, 0])
            a[1] += 2 * self.beta ** 2 * np.dot(dp.ff, nlf.ff)
            a[2] += np.dot(dp.uf + u[:, 0], dp.uf + u[:, 0])
            a[2] += self.beta ** 2 * np.dot(dp.ff, dp.ff)
        if nlf.np:
            a[0] += nlf.up2
            a[1] += 2 * np.dot(nlf.up, dp.up)
            a[2] += np.dot(dp.up, dp.up)
            tmpa = nlf.kpp(p + dp) @ nlf.up
            tmpc = dp.fp + nlf.rp(p + dp)
            if nlf.nf:
                tmpa += nlf.kpf(p + dp) @ u[:, 1]
                tmpc += nlf.kpf(p + dp) @ u[:, 0]
            a[0] += self.beta ** 2 * np.dot(tmpa, tmpa)
            a[1] += 2 * self.beta ** 2 * np.dot(tmpa, tmpc)
            a[2] += self.beta ** 2 * np.dot(tmpc, tmpc)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise DiscriminantError("Discriminant of quadratic constraint equation is not positive: %2f < 0" % float(d))

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])

    def select_root_corrector(self, nlf: Structure, dp: Point, cps: List[Point]) -> float:
        """
        This rule is based on the projections of the generalized correction vectors on the previous correction [Vasios, 2015].
        The corrector that forms the closest correction to the previous point is chosen.
        Note: this rule cannot be used in the first iteration since the initial corrections are equal to zero at the beginning of each increment.
        """
        if nlf.nf:
            cpd = lambda i: np.dot(dp.uf, dp.uf + cps[i].uf)
        if nlf.np:
            cpd = lambda i: np.dot(dp.up, dp.up + cps[i].up)
            if nlf.nf:
                cpd = lambda i: np.dot(dp.uf, dp.uf + cps[i].uf) + np.dot(dp.up, dp.up + cps[i].up)

        return cps[0].y if cpd(0) >= cpd(1) else cps[1].y

    def select_root_predictor(self, nlf: Structure, p: Point, sol: List[Point], cps: List[Point]) -> float:
        if p.y == 0:
            return cps[0].y if cps[0].y > cps[1].y else cps[1].y

        else:
            if nlf.nf:
                vec1 = np.append(sol[-2].uf - p.uf - cps[0].uf, sol[-2].ff - p.ff - cps[0].ff)
                vec2 = np.append(sol[-2].uf - p.uf - cps[1].uf, sol[-2].ff - p.ff - cps[1].ff)

            if nlf.np:
                vec1 = np.append(sol[-2].up - p.up - cps[0].up, sol[-2].fp - p.fp - cps[0].fp)
                vec2 = np.append(sol[-2].up - p.up - cps[1].up, sol[-2].fp - p.fp - cps[1].fp)

                if nlf.nf:
                    vec11 = np.append(sol[-2].uf - p.uf - cps[0].uf, sol[-2].ff - p.ff - cps[0].ff)
                    vec12 = np.append(sol[-2].up - p.up - cps[0].up, sol[-2].fp - p.fp - cps[0].fp)
                    vec1 = np.append(vec11, vec12)
                    vec21 = np.append(sol[-2].uf - p.uf - cps[1].uf, sol[-2].ff - p.ff - cps[1].ff)
                    vec22 = np.append(sol[-2].up - p.up - cps[1].up, sol[-2].fp - p.fp - cps[1].fp)
                    vec2 = np.append(vec21, vec22)

            return cps[0].y if np.linalg.norm(vec1) > np.linalg.norm(vec2) else cps[1].y


class NewtonRaphsonByArcLength(ArcLength):

    def predictor(self, nlf: Structure, p: Point, sol: List[Point], ddx: np.ndarray) -> float:
        a = 0.0
        if nlf.nf:
            a += self.beta ** 2 * nlf.ff2
        if nlf.np:
            a += nlf.up2

        return self.dl / np.sqrt(a)

    def corrector(self, nlf: Structure, p: Point, dp: Point, ddx: np.ndarray) -> float:
        a = np.zeros(3)

        a[2] -= self.dl ** 2
        if nlf.nf:
            a[0] += self.beta ** 2 * nlf.ff2
            a[1] += 2 * self.beta ** 2 * np.dot(dp.ff, nlf.ff)
            a[2] += self.beta ** 2 * np.dot(dp.ff, dp.ff)
        if nlf.np:
            a[0] += nlf.up2
            a[1] += 2 * np.dot(nlf.up, dp.up)
            a[2] += np.dot(dp.up, dp.up)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.sqrt(d)) / (2 * a[0])


class GeneralizedArcLength(ArcLength):
    def __init__(self, dl: float = 0.1, name: str = None, logging_level: int = logging.DEBUG, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(dl, name, logging_level, beta)
        self.alpha = alpha

    def predictor(self, nlf: Structure, p: Point, sol: List[Point], ddx: np.ndarray) -> float:
        y = self.get_roots_predictor(nlf, p, ddx, self.dl)
        cps = [nlf.ddp(p, ddx, i) for i in y]
        return self.select_root_predictor(nlf, p, sol, cps) if self.alpha > 0.0 else cps[0].y

    def corrector(self, nlf: Structure, p: Point, dp: Point, ddx: np.ndarray) -> float:
        try:
            y = self.get_roots_corrector(nlf, p, dp, ddx, self.dl)
        except ValueError as error:
            self.logger.error("{}: {}".format(type(error).__name__, error.args[0]))
            raise ValueError("Roots of constraint equation for the corrector cannot be found!")

        cps = [nlf.ddp(p + dp, ddx, i) for i in y]
        return self.select_root_corrector(nlf, dp, cps) if self.alpha > 0.0 else cps[0].y

    def get_roots_predictor(self, nlf: Structure, p: Point, u: np.ndarray, dl: float) -> float:
        a = 0.0
        if nlf.nf:
            a += self.alpha * np.dot(u[:, 1], u[:, 1]) + self.beta ** 2 * nlf.ff2
        if nlf.np:
            tmpa = nlf.kpp(p) @ nlf.up
            if nlf.nf:
                tmpa += nlf.kpf(p) @ u[:, 1]
            a += self.alpha * self.beta ** 2 * np.dot(tmpa, tmpa) + nlf.up2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, nlf: Structure, p: Point, dp: Point, u: np.ndarray, dl: float) -> float:
        a = np.zeros(3)

        a[2] -= dl ** 2
        if nlf.nf:
            a[0] += self.alpha * np.dot(u[:, 1], u[:, 1])
            a[0] += self.beta ** 2 * nlf.ff2
            a[1] += self.alpha * 2 * np.dot(u[:, 1], dp.uf + u[:, 0])
            a[1] += 2 * self.beta ** 2 * np.dot(dp.ff, nlf.ff)
            a[2] += self.alpha * np.dot(dp.uf + u[:, 0], dp.uf + u[:, 0])
            a[2] += self.beta ** 2 * np.dot(dp.ff, dp.ff)
        if nlf.np:
            a[0] += nlf.up2
            a[1] += 2 * np.dot(nlf.up, dp.up)
            a[2] += np.dot(dp.up, dp.up)
            tmpa = nlf.kpp(p + dp) @ nlf.up
            tmpc = dp.fp + nlf.rp(p + dp)
            if nlf.nf:
                tmpa += nlf.kpf(p + dp) @ u[:, 1]
                tmpc += nlf.kpf(p + dp) @ u[:, 0]
            a[0] += self.alpha * self.beta ** 2 * np.dot(tmpa, tmpa)
            a[1] += self.alpha * 2 * self.beta ** 2 * np.dot(tmpa, tmpc)
            a[2] += self.alpha * self.beta ** 2 * np.dot(tmpc, tmpc)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])


