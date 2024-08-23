from typing import List, Tuple

import numpy as np

from constraints import Constraint, Point, Structure


class IterativeSolver:
    """
    The IterativeSolver is the core of this API, its function is to find a next equilibrium point,
    that is solving the provided system of nonlinear equations given some constraint function.
    """
    def __init__(self, constraint: Constraint) -> None:
        """
        Initialization of the iterative solver.

        :param constraint: constraint function used to solve the system of nonlinear equations

        Note: currently this class acts as a function, more functionality is expected in future.
        """

        # create some aliases for commonly used functions
        self.constraint: Constraint = constraint
        self.nlf: Structure = self.constraint.nlf
        self.maximum_iterates: int = 1000

    def __call__(self, sol: List[Point]) -> Tuple[Point, int, List[Point]]:
        print("Invoking iterative solver")

        p = sol[-1]

        dp = 0.0 * p
        ddx = np.zeros((self.nlf.nf, 2), dtype=float) if self.nlf.nf else None

        if self.nlf.nf:
            load = 1.0 * self.nlf.ff
            load += self.nlf.kfp(p) @ self.nlf.up if self.nlf.np else 0.0
            ddx[:, 1] = np.linalg.solve(self.nlf.kff(p), -load)

        y = self.constraint.predictor(p, sol, ddx)
        dp += self.nlf.get_point(p, ddx, y)

        r = np.array([])
        if self.nlf.nf:
            rf = self.nlf.rf(p + dp)
            r = np.append(r, rf)
        if self.nlf.np:
            r = np.append(r, self.nlf.rp(p + dp))

        tries = [p]

        iterative_counter = 1 # start with 1 as we always do a prediction step

        successful_termination = True # initialize termination to be successful

        # make corrections until termination criteria are met
        while np.any(np.abs(r) > 1e-6):
            iterative_counter += 1 # increase iterative solver counter

            # if there are free degrees of freedom,
            # then do the necessary solves to calculate the free motion
            if self.nlf.nf:
                load = 1.0 * self.nlf.ff
                load += self.nlf.kfp(p + dp) @ self.nlf.up if self.nlf.np else 0.0
                ddx[:, :] = np.linalg.solve(self.nlf.kff(p + dp), -np.array([rf, load]).T)

            y = self.constraint.corrector(p, dp, ddx)
            dp += self.nlf.get_point(p + dp, ddx, y)

            tries.append(p + dp)

            r = np.array([])
            if self.nlf.nf:
                rf = self.nlf.rf(p + dp)
                r = np.append(r, rf)
            if self.nlf.np:
                r = np.append(r, self.nlf.rp(p + dp))

            if iterative_counter > self.maximum_iterates:
                successful_termination = False
                break

        print("Algorithm succesfully terminated") if successful_termination else print("Algorithm unsuccesfully terminated")

        # print("Number of corrections: %d" % iterative_counter)
        return dp, iterative_counter, tries


class IncrementalSolver:
    """
    The IncrementalSolver solves a given system of nonlinear equations by pseudo-time stepping.
    """
    def __init__(self, solution_method: IterativeSolver) -> None:
        """
        Initialization of the incremental solver.

        :param solution_method: type of solution method used to find next equilibrium state

        Note: currently this class acts as a function, more functionality is expected in future.
        For example, currently "only" a single solution_method is used and the type of load increment is fixed.
        """
        self.solution_method = solution_method
        self.maximum_increments = 100

    def __call__(self, p: Point) -> Tuple[List[Point], List[List[Point]]]:
        """
        The __call__ of IncrementalSolver finds a range of equilibrium points given some initial equilibrium point.


        :param p: initial equilibrium state
        :return: a list of equilibrium solutions (Points), and a list of lists of attempted points
        """

        print("Invoking incremental solver")

        # Note: it is assumed the starting guess is an equilibrium point!
        equilibrium_solutions = [p] # adds initial point to equilibrium solutions

        incremental_counter = 0 # counts total number of increments
        iterative_counter = 0 # counts total number of iterates (cumulative throughout increments)
        tries_storage = [] # stores the attempted states of equilibrium (multiple per increment)

        successful_termination = True # set termination (un)succesfull parameter

        # currently very simple termination criteria (load proportionality parameter termination criteria)
        # ideally terminated at p.y == 1.0
        while p.y <= 1.0:
            incremental_counter += 1 # increase incremental counter

            # invoke solution method to find incremental state
            dp, iterates, tries = self.solution_method(equilibrium_solutions)
            p = p + dp # add incremental state to current state (if equilibrium found)

            print("New equilibrium point found at dp.y = %+f in %d iterates, new p.y = %+f " % (dp.y, iterates, p.y))

            iterative_counter += iterates # add iterates of current search to counter
            tries_storage.append(tries) # store tries of current increment to storage

            equilibrium_solutions.append(p) # append equilibrium solution to storage

            # terminate algorithm if too many increments are used
            if incremental_counter > self.maximum_increments:
                successful_termination = False
                break

        print("Algorithm succesfully terminated") if successful_termination else print("Algorithm unsuccesfully terminated")

        print("Total number of increments: %d" % incremental_counter)
        print("Total number of iterates: %d" % iterative_counter)

        return equilibrium_solutions, tries_storage
