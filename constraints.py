from __future__ import annotations

from copy import deepcopy
from typing import List

import numpy as np

from abc import ABC, abstractmethod


State = np.ndarray[float] | None

class DiscriminantError(Exception):
    pass

class Constraint(ABC):
    def __init__(self, dl: float = 0.1) -> None:
        """
        Initialization of the constraint function used to solve the system of nonlinear equations.

        :param nonlinear_function: system of nonlinear equations to solve
        """

        # some aliases for commonly used functions
        self.dl: float = dl # characteristic magnitude of constraint function (e.g. arc-length)

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

    def predictor(self, nlf: Structure, p: Point,  sol: List[Point], ddx: np.ndarray) -> Point:

        load = 0.0
        load += nlf.up2 if nlf.np else 0.0
        load += nlf.ff2 if nlf.nf else 0.0

        return self.dl / np.sqrt(load)


class ArcLength(Constraint):
    def __init__(self, dl: float = 0.1, beta: float = 1.0) -> None:
        super().__init__(dl)
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
    def __init__(self, dl: float = 0.1, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(dl, beta)
        self.alpha = alpha

    def predictor(self, nlf: Structure, p: Point, sol: List[Point], ddx: np.ndarray) -> float:
        y = self.get_roots_predictor(nlf, p, ddx, self.dl)
        cps = [nlf.ddp(p, ddx, i) for i in y]
        return self.select_root_predictor(nlf, p, sol, cps) if self.alpha > 0.0 else cps[0].y

    def corrector(self, nlf: Structure, p: Point, dp: Point, ddx: np.ndarray) -> float:
        y = self.get_roots_corrector(nlf, p, dp, ddx, self.dl)
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


class Structure(ABC):
    """
    Interface of a nonlinear function to the nonlinear solver.

    The external / internal / residual load, motion and stiffness matrix are partitioned based on the free and prescribed degrees of freedom.
    Both the free and prescribed degrees of freedom can be of dimension 0, 1 or higher.
    If dim(free) = 0, then dim(prescribed) > 0 and vice versa.
    That is, either external_load OR prescribed_motion OR BOTH are to be provided.
    """
    def __init__(self):
        self.ff = self.ff()
        self.up = self.up()

        # get dimension of free and prescribed degrees of freedom
        self.nf = np.shape(self.ff)[0] if self.ff is not None else None
        self.np = np.shape(self.up)[0] if self.up is not None else None

        # squared norm of load external load and prescribed motion
        self.ff2 = np.dot(self.ff, self.ff) if self.nf is not None else None
        self.up2 = np.dot(self.up, self.up) if self.np is not None else None

    def ff(self) -> State:
        """
        Applied external load.

        :return: None
        """
        return None

    def up(self) -> State:
        """
        Prescribed motion.

        :return: None
        """
        return None

    def internal_load_free(self, p: Point) -> State:
        """
        Internal load associated to the free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def internal_load_prescribed(self, p: Point) -> State:
        """
        Internal load associated to the prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def rf(self, p: Point) -> State:
        """
        Residual associated to the free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: residual associated to the free degrees of freedom
        """

        # free residual is defined as the free internal load PLUS the proportional loading parameter times the applied external load
        return self.internal_load_free(p) - p.y * self.ff

    def rp(self, p: Point) -> State:
        """
        Residual associated to the prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: residual associated to the prescribed degrees of freedom
        """

        # prescribed residual is defined as the prescribed internal load PLUS the reaction load
        return self.internal_load_prescribed(p) - p.fp

    def kff(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the free-free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def kfp(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the free-prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def kpf(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the prescribed-free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def kpp(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the prescribed-prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def ddp(self, p: Point, u: np.ndarray, y: float) -> Point:
        """
        Provides the iterative updated state given some iterative load parameter.

        :param p: current state (p + dp)
        :param u: resultants from solve
        :param y: iterative load parameter
        :return:
        """
        dduf, ddup, ddff, ddfp = 0.0, 0.0, 0.0, 0.0

        if self.nf:
            dduf = u[:, 0] + y * u[:, 1]
            ddff = y * self.ff
        if self.np:
            ddup = y * self.up
            ddfp = self.rp(p) + y * self.kpp(p) @ self.up
            ddfp += self.kpf(p) @ dduf if self.nf else 0.0

        return Point(dduf, ddup, ddff, ddfp, y)


class Point:
    def __init__(self, uf: State = 0.0, up: State = 0.0, ff: State = 0.0, fp: State = 0.0, y: float = 0.0) -> None:
        """
        Initialize an (equilibrium) point given it's load and corresponding motion in partitioned format.

        :param uf: free / unknown motion
        :param up: prescribed motion
        :param ff: external / applied load
        :param fp: reaction load
        :param y: load proportionality parameter
        """
        self.uf = uf
        self.up = up
        self.ff = ff
        self.fp = fp
        self.y = y

    def __iadd__(self, other: Point) -> Point:
        """
        Adds the content of another Point to this Point.

        :param other: another Point object
        :return: sum of Points
        """
        self.uf += other.uf
        self.up += other.up
        self.ff += other.ff
        self.fp += other.fp
        self.y += other.y
        return self

    def __rmul__(self, other: Point) -> Point:
        """
        Multiplications of two point entries.

        Note rmul makes a deepcopy of itself!

        :param other: another Point
        :return: a copy of itself with the entries multiplied by the other Points entries
        """
        out = deepcopy(self)
        out.uf *= other
        out.up *= other
        out.ff *= other
        out.fp *= other
        out.y *= other
        return out

    def __add__(self, other: Point) -> Point:
        """
        Addition of two points, returing a third Point.

        :param other: another Point object
        :return: a third Point object that is the addition
        """
        out = deepcopy(Point(self.uf, self.up, self.ff, self.fp, self.y))
        out += other
        return out

    def __sub__(self, other: Point) -> Point:
        """
        Substraction of two points, returing a third Point.

        :param other: another Point object
        :return: a third Point object that is the substraction
        """
        out = deepcopy(Point(self.uf, self.up, self.ff, self.fp, self.y))
        out -= other
        return out
