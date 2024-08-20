from typing import List, Tuple

import numpy as np

from constraint import Constraint
from point import Point


class IterativeSolver:
    def __init__(self, constraint: Constraint) -> None:
        self.constraint = constraint
        self.nonlinear_function = self.constraint.a
        self.ff = self.constraint.ff
        self.up = self.constraint.up
        self.nf = self.constraint.nf
        self.np = self.constraint.np
        self.Kfp = self.nonlinear_function.tangent_stiffness_free_prescribed
        self.Kff = self.nonlinear_function.tangent_stiffness_free_free
        self.rf = self.nonlinear_function.residual_free
        self.rp = self.nonlinear_function.residual_prescribed

    def __call__(self, sol: List[Point], dl: float = 1.0) -> Tuple[Point, int, List[Point]]:

        p = sol[-1]

        dp = 0.0 * p
        ddx = np.zeros((self.nf, 2), dtype=float) if self.nf else None

        if self.nf:
            load = 1.0 * self.ff
            load += self.Kfp(p) @ self.up if self.np else 0.0
            ddx[:, 1] = np.linalg.solve(self.Kff(p), -load)

        dp += self.constraint.predictor(p, dp, ddx, dl, sol)

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
                load += self.Kfp(p + dp) @ self.up if self.np else 0.0
                ddx[:, :] = np.linalg.solve(self.Kff(p + dp), -np.array([rf, load]).T)

            dp += self.constraint.corrector(p, dp, ddx, dl)

            tries.append(p + dp)

            r = np.array([])
            if self.nf:
                rf = self.rf(p + dp)
                r = np.append(r, rf)
            if self.np:
                r = np.append(r, self.rp(p + dp))

        # print("Number of corrections: %d" % iterative_counter)
        return dp, iterative_counter, tries


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
