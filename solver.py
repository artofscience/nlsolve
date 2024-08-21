from typing import List, Tuple

import numpy as np

from constraints import Constraint, Point


class IterativeSolver:
    def __init__(self, constraint: Constraint) -> None:
        """
        Initialization of the iterative solver.

        :param constraint: constraint function used to solve the system of nonlinear equations
        """

        # create some aliases for commonly used functions
        self.constraint = constraint
        self.nlf = self.constraint.nlf

    def __call__(self, sol: List[Point], dl: float = 0.1) -> Tuple[Point, int, List[Point]]:

        p = sol[-1]

        dp = 0.0 * p
        ddx = np.zeros((self.nlf.nf, 2), dtype=float) if self.nlf.nf else None

        if self.nlf.nf:
            load = 1.0 * self.nlf.ff
            load += self.nlf.kfp(p) @ self.nlf.up if self.nlf.np else 0.0
            ddx[:, 1] = np.linalg.solve(self.nlf.kff(p), -load)

        y = self.constraint.predictor(p, sol, ddx, dl)
        dp += self.nlf.get_point(p, ddx, y)

        r = np.array([])
        if self.nlf.nf:
            rf = self.nlf.rf(p + dp)
            r = np.append(r, rf)
        if self.nlf.np:
            r = np.append(r, self.nlf.rp(p + dp))

        tries = [p]

        iterative_counter = 0
        while np.any(np.abs(r) > 1e-6):
            iterative_counter += 1

            if self.nlf.nf:
                load = 1.0 * self.nlf.ff
                load += self.nlf.kfp(p + dp) @ self.nlf.up if self.nlf.np else 0.0
                ddx[:, :] = np.linalg.solve(self.nlf.kff(p + dp), -np.array([rf, load]).T)

            y = self.constraint.corrector(p, dp, ddx, dl)
            dp += self.nlf.get_point(p + dp, ddx, y)

            tries.append(p + dp)

            r = np.array([])
            if self.nlf.nf:
                rf = self.nlf.rf(p + dp)
                r = np.append(r, rf)
            if self.nlf.np:
                r = np.append(r, self.nlf.rp(p + dp))

        # print("Number of corrections: %d" % iterative_counter)
        return dp, iterative_counter, tries


class IncrementalSolver:
    def __init__(self, solution_method: IterativeSolver) -> None:
        self.solution_method = solution_method

    def __call__(self, p: Point) -> Tuple[List[Point], List[List[Point]]]:
        equilibrium_solutions = [p]

        incremental_counter = 0
        iterative_counter = 0
        tries_storage = []

        while p.y <= 1.0:
            incremental_counter += 1

            dp, iterates, tries = self.solution_method(equilibrium_solutions)
            iterative_counter += iterates
            tries_storage.append(tries)

            p = p + dp
            equilibrium_solutions.append(p)

        print("Number of increments: %d" % incremental_counter)
        print("Total number of iterates: %d" % iterative_counter)

        return equilibrium_solutions, tries_storage
