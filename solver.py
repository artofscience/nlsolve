from typing import List, Tuple

import numpy as np

from constraint import Constraint
from point import Point


class IterativeSolver:
    def __init__(self, constraint: Constraint) -> None:
        """
        Initialization of the iterative solver.

        :param constraint: constraint function used to solve the system of nonlinear equations
        """

        # create some aliases for commonly used functions
        self.constraint = constraint
        self.nlf = self.constraint.nlf
        self.ff = self.constraint.ff
        self.up = self.constraint.up
        self.nf = self.constraint.nf
        self.np = self.constraint.np
        self.kfp = self.nlf.tangent_stiffness_free_prescribed
        self.kff = self.nlf.tangent_stiffness_free_free
        self.kpf = self.nlf.tangent_stiffness_prescribed_free
        self.kpp = self.nlf.tangent_stiffness_prescribed_prescribed
        self.rf = self.nlf.residual_free
        self.rp = self.nlf.residual_prescribed

    def __call__(self, sol: List[Point], dl: float = 1.0) -> Tuple[Point, int, List[Point]]:

        p = sol[-1]

        dp = 0.0 * p
        ddx = np.zeros((self.nf, 2), dtype=float) if self.nf else None

        if self.nf:
            load = 1.0 * self.ff
            load += self.kfp(p) @ self.up if self.np else 0.0
            ddx[:, 1] = np.linalg.solve(self.kff(p), -load)

        y = self.constraint.predictor(p, sol, ddx, dl)
        dp += self.get_point(p, ddx, y)

        r = np.array([])
        if self.nf:
            rf = self.rf(p + dp)
            r = np.append(r, rf)
        if self.np:
            r = np.append(r, self.rp(p + dp))

        tries = [p]

        iterative_counter = 0
        while np.any(np.abs(r) > 1e-6):
            iterative_counter += 1

            if self.nf:
                load = 1.0 * self.ff
                load += self.kfp(p + dp) @ self.up if self.np else 0.0
                ddx[:, :] = np.linalg.solve(self.kff(p + dp), -np.array([rf, load]).T)

            y = self.constraint.corrector(p, dp, ddx, dl)
            dp += self.get_point(p + dp, ddx, y)

            tries.append(p + dp)

            r = np.array([])
            if self.nf:
                rf = self.rf(p + dp)
                r = np.append(r, rf)
            if self.np:
                r = np.append(r, self.rp(p + dp))

        # print("Number of corrections: %d" % iterative_counter)
        return dp, iterative_counter, tries

    def get_point(self, p: Point, u: np.ndarray, y: float) -> Point:
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
            ddfp = -self.rp(p) - y * self.kpp(p) @ self.up
            ddfp -= self.kpf(p) @ dduf if self.nf else 0.0

        return Point(dduf, ddup, ddff, ddfp, y)


class IncrementalSolver:
    def __init__(self, solution_method: IterativeSolver, dl: float = 0.1) -> None:
        self.dl = dl
        self.solution_method = solution_method

    def __call__(self, p: Point) -> Tuple[List[Point], List[List[Point]]]:
        equilibrium_solutions = [p]

        incremental_counter = 0
        iterative_counter = 0
        tries_storage = []

        while p.y <= 1.0:
            incremental_counter += 1

            dp, iterates, tries = self.solution_method(equilibrium_solutions, self.dl)
            iterative_counter += iterates
            tries_storage.append(tries)

            p = p + dp
            equilibrium_solutions.append(p)

        print("Number of increments: %d" % incremental_counter)
        print("Total number of iterates: %d" % iterative_counter)

        return equilibrium_solutions, tries_storage
