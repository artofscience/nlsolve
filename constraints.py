from typing import List

import numpy as np

from constraint import Constraint
from point import Point
from structure import Structure


class NewtonRaphson(Constraint):

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> float:
        return 0.0

    def predictor(self, p: Point,  sol: List[Point], ddx: np.ndarray, dl: float) -> Point:

        load = 0.0
        load += self.up2 if self.np else 0.0
        load += self.ff2 if self.nf else 0.0

        return dl / np.sqrt(load)


class ArcLength(Constraint):
    def __init__(self, nonlinear_function: Structure, beta: float = 1.0) -> None:
        super().__init__(nonlinear_function)
        self.beta = beta

    def predictor(self, p: Point, sol: List[Point], ddx: np.ndarray, dl: float) -> float:
        y = self.get_roots_predictor(p, ddx, dl)
        cps = [self.get_point(p, ddx, i) for i in y]
        return self.select_root_predictor(p, sol, cps)

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> float:
        y = self.get_roots_corrector(p, dp, ddx, dl)
        cps = [self.get_point(p + dp, ddx, i) for i in y]
        return self.select_root_corrector(dp, cps)

    def get_roots_predictor(self, p: Point, u: np.ndarray, dl: float) -> np.ndarray:
        a = 0.0
        if self.nf:
            a += np.dot(u[:, 1], u[:, 1]) + self.beta ** 2 * self.ff2
        if self.np:
            tmpa = self.kpp(p) @ self.up
            if self.nf:
                tmpa += self.kpf(p) @ u[:, 1]
            a += self.beta ** 2 * np.dot(tmpa, tmpa) + self.up2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, p: Point, dp: Point, u: np.ndarray, dl: float) -> np.ndarray:
        a = np.zeros(3)

        a[2] -= dl ** 2
        if self.nf:
            a[0] += np.dot(u[:, 1], u[:, 1])
            a[0] += self.beta ** 2 * self.ff2
            a[1] += 2 * np.dot(u[:, 1], dp.uf + u[:, 0])
            a[1] += 2 * self.beta ** 2 * np.dot(dp.ff, self.ff)
            a[2] += np.dot(dp.uf + u[:, 0], dp.uf + u[:, 0])
            a[2] += self.beta ** 2 * np.dot(dp.ff, dp.ff)
        if self.np:
            a[0] += self.up2
            a[1] += 2 * np.dot(self.up, dp.up)
            a[2] += np.dot(dp.up, dp.up)
            tmpa = self.kpp(p + dp) @ self.up
            tmpc = dp.fp - self.nlf.residual_prescribed(p + dp)
            if self.nf:
                tmpa += self.kpf(p + dp) @ u[:, 1]
                tmpc -= self.kpf(p + dp) @ u[:, 0]
            a[0] += self.beta ** 2 * np.dot(tmpa, tmpa)
            a[1] -= 2 * self.beta ** 2 * np.dot(tmpa, tmpc)
            a[2] += self.beta ** 2 * np.dot(tmpc, tmpc)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])

    def select_root_corrector(self, dp: Point, cps: List[Point]) -> float:
        """
        This rule is based on the projections of the generalized correction vectors on the previous correction [Vasios, 2015].
        The corrector that forms the closest correction to the previous point is chosen.
        Note: this rule cannot be used in the first iteration since the initial corrections are equal to zero at the beginning of each increment.
        """
        if self.nf:
            cpd = lambda i: np.dot(dp.uf, dp.uf + cps[i].uf)
        if self.np:
            cpd = lambda i: np.dot(dp.up, dp.up + cps[i].up)
            if self.nf:
                cpd = lambda i: np.dot(dp.uf, dp.uf + cps[i].uf) + np.dot(dp.up, dp.up + cps[i].up)

        return cps[0].y if cpd(0) >= cpd(1) else cps[1].y

    def select_root_predictor(self, p: Point, sol: List[Point], cps: List[Point]) -> float:
        if p.y == 0:
            return cps[0].y if cps[0].y > cps[1].y else cps[1].y

        else:
            if self.nf:
                vec1 = np.append(sol[-2].uf - p.uf - cps[0].uf, sol[-2].ff - p.ff - cps[0].ff)
                vec2 = np.append(sol[-2].uf - p.uf - cps[1].uf, sol[-2].ff - p.ff - cps[1].ff)

            if self.np:
                vec1 = np.append(sol[-2].up - p.up - cps[0].up, sol[-2].fp - p.fp - cps[0].fp)
                vec2 = np.append(sol[-2].up - p.up - cps[1].up, sol[-2].fp - p.fp - cps[1].fp)

                if self.nf:
                    vec11 = np.append(sol[-2].uf - p.uf - cps[0].uf, sol[-2].ff - p.ff - cps[0].ff)
                    vec12 = np.append(sol[-2].up - p.up - cps[0].up, sol[-2].fp - p.fp - cps[0].fp)
                    vec1 = np.append(vec11, vec12)
                    vec21 = np.append(sol[-2].uf - p.uf - cps[1].uf, sol[-2].ff - p.ff - cps[1].ff)
                    vec22 = np.append(sol[-2].up - p.up - cps[1].up, sol[-2].fp - p.fp - cps[1].fp)
                    vec2 = np.append(vec21, vec22)

            return cps[0].y if np.linalg.norm(vec1) > np.linalg.norm(vec2) else cps[1].y


class NewtonRaphsonByArcLength(ArcLength):

    def predictor(self, p: Point, sol: List[Point], ddx: np.ndarray, dl: float) -> float:
        return self.get_roots_predictor(p, ddx, dl)

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> float:
        return self.get_roots_corrector(p, dp, ddx, dl)

    def get_roots_predictor(self, p: Point, u: np.ndarray, dl: float) -> float:
        a = 0.0
        if self.nf:
            a += self.beta ** 2 * self.ff2
        if self.np:
            a += self.up2

        return dl / np.sqrt(a)

    def get_roots_corrector(self, p: Point, dp: Point, u: np.ndarray, dl: float) -> float:
        a = np.zeros(3)

        a[2] -= dl ** 2
        if self.nf:
            a[0] += self.beta ** 2 * self.ff2
            a[1] += 2 * self.beta ** 2 * np.dot(dp.ff, self.ff)
            a[2] += self.beta ** 2 * np.dot(dp.ff, dp.ff)
        if self.np:
            a[0] += self.up2
            a[1] += 2 * np.dot(self.up, dp.up)
            a[2] += np.dot(dp.up, dp.up)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.sqrt(d)) / (2 * a[0])


class GeneralizedArcLength(ArcLength):
    def __init__(self, nonlinear_function: Structure, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(nonlinear_function, beta)
        self.alpha = alpha

    def predictor(self, p: Point, sol: List[Point], ddx: np.ndarray, dl: float) -> float:
        y = self.get_roots_predictor(p, ddx, dl)
        cps = [self.get_point(p, ddx, i) for i in y]
        return self.select_root_predictor(p, sol, cps) if self.alpha > 0.0 else cps[0].y

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> float:
        y = self.get_roots_corrector(p, dp, ddx, dl)
        cps = [self.get_point(p + dp, ddx, i) for i in y]
        return self.select_root_corrector(dp, cps) if self.alpha > 0.0 else cps[0].y

    def get_roots_predictor(self, p: Point, u: np.ndarray, dl: float) -> float:
        a = 0.0
        if self.nf:
            a += self.alpha * np.dot(u[:, 1], u[:, 1]) + self.beta ** 2 * self.ff2
        if self.np:
            tmpa = self.kpp(p) @ self.up
            if self.nf:
                tmpa += self.kpf(p) @ u[:, 1]
            a += self.alpha * self.beta ** 2 * np.dot(tmpa, tmpa) + self.up2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, p: Point, dp: Point, u: np.ndarray, dl: float) -> float:
        a = np.zeros(3)

        a[2] -= dl ** 2
        if self.nf:
            a[0] += self.alpha * np.dot(u[:, 1], u[:, 1])
            a[0] += self.beta ** 2 * self.ff2
            a[1] += self.alpha * 2 * np.dot(u[:, 1], dp.uf + u[:, 0])
            a[1] += 2 * self.beta ** 2 * np.dot(dp.ff, self.ff)
            a[2] += self.alpha * np.dot(dp.uf + u[:, 0], dp.uf + u[:, 0])
            a[2] += self.beta ** 2 * np.dot(dp.ff, dp.ff)
        if self.np:
            a[0] += self.up2
            a[1] += 2 * np.dot(self.up, dp.up)
            a[2] += np.dot(dp.up, dp.up)
            tmpa = self.kpp(p + dp) @ self.up
            tmpc = dp.fp - self.nlf.residual_prescribed(p + dp)
            if self.nf:
                tmpa += self.kpf(p + dp) @ u[:, 1]
                tmpc -= self.kpf(p + dp) @ u[:, 0]
            a[0] += self.alpha * self.beta ** 2 * np.dot(tmpa, tmpa)
            a[1] -= self.alpha * 2 * self.beta ** 2 * np.dot(tmpa, tmpc)
            a[2] += self.alpha * self.beta ** 2 * np.dot(tmpc, tmpc)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])
