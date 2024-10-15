from __future__ import annotations
from typing import List, Tuple

from constraints import Constraint, NewtonRaphson
from controllers import Controller

import operator

import numpy as np

from logger import CustomFormatter, create_logger
from utils import Structure, Point

State = np.ndarray[float] | None

class CounterError(Exception):
    pass


import logging

from criteria import Counter, ConvergenceCriterion


class IterativeSolver:
    """
    The IterativeSolver is the core of this API, its function is to find a next equilibrium point,
    that is solving the provided system of nonlinear equations given some constraint function.
    """

    def __init__(self, nlf: Structure, constraint: Constraint = None, converged = None,
                 name: str = None, logging_level: int = logging.DEBUG,
                 maximum_corrections: int = 1000) -> None:
        """
        Initialization of the iterative solver.

        :param constraint: constraint function used to solve the system of nonlinear equations

        Note: currently this class acts as a function, more functionality is expected in future.
        """

        # create some aliases for commonly used functions
        self.converged = converged
        self.nlf: Structure = nlf  # nonlinear system of equations
        self.constraint = constraint if constraint is not None else NewtonRaphson() # constraint function used (operates on nlf)
        self.maximum_corrections: int = maximum_corrections  # maximum allowed number of iterates before premature termination
        self.residual_norm_tolerance: float = 1e-3

        self.__name__ = name if name is not None else (self.__class__.__name__ + " " + str(id(self)))

        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + self.__name__)

    def __call__(self, sol: List[Point], length: float = 0.1) -> Tuple[Point, int, List[Point]]:
        self.logger.debug("Starting iterative solver")

        self.constraint.dl = length  # set characteristic length of constraint

        p = sol[-1]  # takes the initial equilibrium point (what if this is not in equilibrium?)
        tries = [p]  # initialize storage for attempted states and add initial point

        # region PREDICTOR

        # initialize structure of solve return values if free degrees of freedom
        ddx = np.zeros((self.nlf.nf, 2), dtype=float) if self.nlf.nf else None

        # solve the system of equations [-kff @ ddx1 = ff + kfp @ up] at state = p
        # note: for predictor ddx0 = 0, hence only a single rhs for this solve
        if self.nlf.nf:
            ddx[:, 1] = np.linalg.solve(self.nlf.kff(p), self.nlf.load(p))

        # call to the predictor of the constraint function returning iterative load parameter
        # note it has access to previous equilibrium points (sol) and dp = 0
        # note for first iterate dy = ddy and dp = ddp
        try:
            ddy = self.constraint.predictor(self.nlf, p, sol, ddx)
        except ValueError as error:
            self.logger.error("{}: {}".format(type(error).__name__, error.args[0]))
            raise ValueError("A suitable prediction cannot be found!", 0)

        self.logger.debug("Predictor 0: ddy = %+f" % ddy)

        dp = self.nlf.ddp(p, ddx, ddy)  # calculate prediction based on iterative load parameter

        # endregion


        counter = Counter(self.maximum_corrections)

        # make corrections until termination criteria are met
        while True:
            if counter:
                raise CounterError("Maximum number of corrections %2d > %2d" % (counter.count, counter.threshold), counter.count)

            if self.converged(self.nlf, p+dp):
                break

            # region CORRECTOR

            # solve the system of equations kff @ [ddx0, ddx1] = -[rf, ff + kfp @ up] at state = p + dp
            if self.nlf.nf:
                ddx[:, :] = np.linalg.solve(self.nlf.kff(p + dp), np.array([-self.nlf.rf(p + dp), self.nlf.load(p + dp)]).T)

            # calculate correction of proportional load parameter
            # note: p and dp are passed independently (instead of p + dp), as dp is used for root selection
            try:
                ddy = self.constraint.corrector(self.nlf, p, dp, ddx)
            except ValueError as error:
                self.logger.error("{}: {}".format(type(error).__name__, error.args[0]))
                raise ValueError("A suitable correction cannot be found!", counter.count)

            self.logger.debug("Corrector %d: ddy = %+e" % (counter.count, ddy))

            dp += self.nlf.ddp(p + dp, ddx, ddy) # calculate correction based on iterative load parameter and update incremental state

            #endregion

            tries.append(p + dp)  # add attempt to tries

        return dp, counter.count, tries


class IncrementalSolver:
    """
    The IncrementalSolver solves a given system of nonlinear equations by pseudo-time stepping.
    """

    def __init__(self, solution_method: IterativeSolver,
                 name: str = "MyIncrementalSolver", logging_level: int = logging.DEBUG,
                 maximum_increments: int = 1000,
                 terminated = None) -> None:
        """
        Initialization of the incremental solver.

        :param solution_method: type of solution method used to find next equilibrium state
        :param name: name of the incremental solver
        :param logging_level: logging level
        :param maximum_increments: maximum number of iterations

        Note: currently this class acts as a function, more functionality is expected in future.
        For example, currently "only" a single solution_method is used and the type of load increment is fixed.
        """
        self.solution_method = solution_method
        self.maximum_increments: int = maximum_increments
        self.terminated = terminated if terminated is not None else ConvergenceCriterion(lambda x, y, z: y.y, operator.ge, 1.0)

        self.__name__ = name

        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + name)

    def __call__(self, p: Point, controller: Controller = None) -> Tuple[List[Point], List[List[Point]]]:
        """
        The __call__ of IncrementalSolver finds a range of equilibrium points given some initial equilibrium point.


        :param p: initial equilibrium state
        :param controller: controller of the pseud-time step size
        :return: a list of equilibrium solutions (Points), and a list of lists of attempted points
        """
        controller = controller if controller is not None else Controller(0.1)

        self.logger.debug("Invoking incremental solver")

        # Note: it is assumed the starting guess is an equilibrium point!
        equilibrium_solutions = [p]  # adds initial point to equilibrium solutions

        incremental_counter = 0  # counts total number of succesful increments
        incremental_tries = 0  # counts total number of times the iterative solver is invoked
        iterative_counter = 0  # counts total number of iterates (cumulative throughout increments)
        iterative_tries = 0  # counts total number of iterates (cumulative throughout increments)

        tries_storage = []  # stores the attempted states of equilibrium (multiple per increment)

        try:
            # currently very simple termination criteria (load proportionality parameter termination criteria)
            # ideally terminated at p.y == 1.0

            while True:
                if self.terminated(self.solution_method.nlf, p):
                    break

                incremental_counter += 1

                print("")

                # invoke solution method to find incremental state
                while True:
                    try:
                        incremental_tries += 1
                        self.logger.info("Invoking iterative solver for %d-th time to find %d-th equilibrium point" % (incremental_tries, incremental_counter))
                        dp, iterates, tries = self.solution_method(equilibrium_solutions, controller.value)
                        iterative_tries += iterates
                        break
                    except (ValueError, CounterError) as error:
                        self.logger.error("{}: {}".format(type(error).__name__, error.args[0]))
                        iterative_tries += error.args[1]
                        self.logger.error("Iterative solver aborted after %d iterates" % error.args[1])
                        self.logger.warning("Decrease characteristic length of constraint equation and try again!")
                        controller.decrease() # decrease the characteristic length of the constraint

                p = p + dp  # add incremental state to current state (if equilibrium found)

                self.logger.debug(
                    "New equilibrium point found at dp.y = %+f in %d iterates, new p.y = %+f " % (
                        dp.y, iterates, p.y))

                equilibrium_solutions.append(p)  # append equilibrium solution to storage

                controller.increase()  # increase the characteristic length of the constraint for next iterate

                iterative_counter += iterates  # add iterates of current search to counter
                tries_storage.append(tries)  # store tries of current increment to storage

                self.logger.info("Total number of increments: %d" % incremental_tries)
                self.logger.debug("Total number of iterates: %d" % iterative_tries)

                self.logger.info("Total number of succesful increments: %d" % incremental_counter)
                self.logger.debug("Total number of effective iterates: %d" % iterative_counter)

                # terminate algorithm if too many increments are used
                if incremental_counter >= self.maximum_increments:
                    raise CounterError(
                        "Maximum number of increments %2d >= %2d" % (incremental_counter, self.maximum_increments))

        except CounterError as error:
            self.logger.warning("{}: {}".format(type(error).__name__, error.args[0]))
            return equilibrium_solutions, tries_storage

        return equilibrium_solutions, tries_storage


