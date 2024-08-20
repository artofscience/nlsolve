from typing import List

import numpy as np

from constraint import Constraint
from point import Point
from structure import Structure


class NewtonRaphson(Constraint):

    def predictor(self, p: Point, dp: Point, ddx: np.ndarray, dl: float, sol: List[Point]) -> Point:

        load = 0.0
        load += self.up2 if self.np else 0.0
        load += self.ff2 if self.nf else 0.0

        ddy = dl / np.sqrt(load)

        point = Point(y=ddy)

        if self.nf:
            point.uf += ddy * ddx[:, 1]
            point.ff += ddy * self.ff
        if self.np:
            point.up = ddy * self.up
            point.fp = -self.kpp(p) @ point.up
            point.fp -= ddy * self.kpf(p) @ ddx[:, 1] if self.nf else 0.0

        return point

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> Point:

        point = Point()

        if self.nf:
            point.uf += ddx[:, 0]
        if self.np:
            point.fp = -self.rp(p + dp)
            point.fp -= self.kpf(p + dp) @ ddx[:, 0] if self.nf else 0.0

        return point


class ArcLength(Constraint):
    def __init__(self, nonlinear_function: Structure, beta: float = 1.0) -> None:
        super().__init__(nonlinear_function)
        self.beta = beta

    def predictor(self, p: Point, dp: Point, ddx: np.ndarray, dl: float, sol: List[Point]) -> Point:
        y = self.get_roots_predictor(p, ddx[:, 1], dl)
        cps = self.get_point(p, dp, ddx, y)
        return self.select_root_predictor(p, sol, cps)

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> Point:
        y = self.get_roots_corrector(p, dp, ddx, dl)
        cps = self.get_point(p, dp, ddx, y)
        return self.select_root_corrector(dp, cps)

    def get_roots_predictor(self, p: Point, u: np.ndarray, dl: float) -> np.ndarray:
        a = 0.0
        if self.nf:
            a += np.dot(u, u) + self.beta ** 2 * self.ff2
        if self.np:
            tmpa = self.kpp(p) @ self.up
            if self.nf:
                tmpa += self.kpf(p) @ u
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

    def get_point(self, p: Point, dp: Point, u: np.ndarray, y: np.ndarray) -> List[Point]:
        if self.np is not None and self.nf is None:
            ddp = [-self.rp(p + dp) - y[i] * self.kpp(p + dp) @ self.up for i in range(2)]
            return [Point(up=y[i] * self.up, fp=ddp[i], y=y[i]) for i in range(2)]
        if self.nf is not None:
            x = [u[:, 0] + i * u[:, 1] for i in y]
            if self.np is None:
                return [Point(uf=x[i], ff=y[i] * self.ff, y=y[i]) for i in range(2)]
            else:
                ddp = [-self.rp(p + dp) - self.kpf(p + dp) @ u[:, 0] - y[i] * (
                        self.kpf(p + dp) @ u[:, 1] + self.kpp(p + dp) @ self.up) for i in range(2)]
                return [Point(uf=x[i], up=y[i] * self.up, ff=y[i] * self.ff, fp=ddp[i], y=y[i]) for i in range(2)]

    def select_root_corrector(self, dp: Point, cps: List[Point]) -> Point:
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

        return cps[0] if cpd(0) >= cpd(1) else cps[1]

    def select_root_predictor(self, p: Point, sol: List[Point], cps: List[Point]) -> Point:
        if p.y == 0:
            return cps[0] if cps[0].y > cps[1].y else cps[1]

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

            return cps[0] if np.linalg.norm(vec1) > np.linalg.norm(vec2) else cps[1]


class NewtonRaphsonByArcLength(ArcLength):

    def predictor(self, p: Point, dp: Point, ddx: np.ndarray, dl: float, sol: List[Point]) -> Point:
        y = self.get_roots_predictor(p, ddx[:, 1], dl)
        cps = self.get_point(p, dp, ddx, y)
        return cps[0]

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> Point:
        y = self.get_roots_corrector(p, dp, ddx, dl)
        cps = self.get_point(p, dp, ddx, y)
        return cps[0]

    def get_roots_predictor(self, p: Point, u: np.ndarray, dl: float) -> np.ndarray:
        a = 0.0
        if self.nf:
            a += self.beta ** 2 * self.ff2
        if self.np:
            a += self.up2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, p: Point, dp: Point, u: np.ndarray, dl: float) -> np.ndarray:
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

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])


class GeneralizedArcLength(ArcLength):
    def __init__(self, nonlinear_function: Structure, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(nonlinear_function, beta)
        self.alpha = alpha

    def predictor(self, p: Point, dp: Point, ddx: np.ndarray, dl: float, sol: List[Point]) -> Point:
        y = self.get_roots_predictor(p, ddx[:, 1], dl)
        cps = self.get_point(p, dp, ddx, y)
        return self.select_root_predictor(p, sol, cps) if self.alpha > 0.0 else cps[0]

    def corrector(self, p: Point, dp: Point, ddx: np.ndarray, dl: float) -> Point:
        y = self.get_roots_corrector(p, dp, ddx, dl)
        cps = self.get_point(p, dp, ddx, y)
        return self.select_root_corrector(dp, cps) if self.alpha > 0.0 else cps[0]

    def get_roots_predictor(self, p: Point, u: np.ndarray, dl: float) -> np.ndarray:
        a = 0.0
        if self.nf:
            a += self.alpha * np.dot(u, u) + self.beta ** 2 * self.ff2
        if self.np:
            tmpa = self.kpp(p) @ self.up
            if self.nf:
                tmpa += self.kpf(p) @ u
            a += self.alpha * self.beta ** 2 * np.dot(tmpa, tmpa) + self.up2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, p: Point, dp: Point, u: np.ndarray, dl: float) -> np.ndarray:
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
