from typing import List, Tuple
import sys

import numpy as np

from constraints import Constraint, Point, Structure, DiscriminantError

class CounterError(Exception):
    pass

import logging

class IterativeSolver:
    """
    The IterativeSolver is the core of this API, its function is to find a next equilibrium point,
    that is solving the provided system of nonlinear equations given some constraint function.
    """
    def __init__(self, nlf: Structure, constraint: Constraint,
                 name: str = "IterativeSolver", logging_level: int = logging.DEBUG) -> None:
        """
        Initialization of the iterative solver.

        :param constraint: constraint function used to solve the system of nonlinear equations

        Note: currently this class acts as a function, more functionality is expected in future.
        """

        # create some aliases for commonly used functions
        self.nlf: Structure = nlf # nonlinear system of equations
        self.constraint: Constraint = constraint # constraint function used (operates on nlf)
        self.maximum_iterates: int = 1000 # maximum allowed number of iterates before premature termination

        self.__name__ = name

        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + name)

    def __call__(self, sol: List[Point]) -> Tuple[Point, int, List[Point]]:
        self.logger.debug("Invoking iterative solver")

        p = sol[-1] # takes the initial equilibrium point (what if this is not in equilibrium?)
        tries = [p] # initialize storage for attempted states and add initial point

        # dp = 0.0 * p # make a copy of the structure of p and reset all entries to zeros

        iterative_counter = 0 # start with 0 as we count the corrections only

        #region PREDICTOR

        # initialize structure of solve return values if free degrees of freedom
        ddx = np.zeros((self.nlf.nf, 2), dtype=float) if self.nlf.nf else None

        # solve the system of equations [-kff @ ddx1 = ff + kfp @ up] at state = p
        # note: for predictor ddx0 = 0, hence only a single rhs for this solve
        if self.nlf.nf:
            load = 1.0 * self.nlf.ff
            load += self.nlf.kfp(p) @ self.nlf.up if self.nlf.np else 0.0 # adds to rhs if nonzero prescribed dof
            ddx[:, 1] = np.linalg.solve(self.nlf.kff(p), -load)

        # call to the predictor of the constraint function returning iterative load parameter
        # note it has access to previous equilibrium points (sol) and dp = 0
        # note for first iterate dy = ddy and dp = ddp
        ddy = self.constraint.predictor(self.nlf, p, sol, ddx)
        self.logger.debug("Predictor 0: ddy = %+f" % ddy)

        dp = self.nlf.ddp(p, ddx, ddy) # calculate prediction based on iterative load parameter

        #endregion

        # combine residuals associated to free and prescribed dofs (if available) for termiantion criteria
        r = self.get_r(p + dp)

        # make corrections until termination criteria are met
        while np.any(np.abs(r) > 1e-6):
            iterative_counter += 1 # increase iterative solver counter

            #region CORRECTOR

            # solve the system of equations kff @ [ddx0, ddx1] = -[rf, ff + kfp @ up] at state = p + dp
            if self.nlf.nf:
                load = 1.0 * self.nlf.ff
                load += self.nlf.kfp(p + dp) @ self.nlf.up if self.nlf.np else 0.0
                ddx[:, :] = np.linalg.solve(self.nlf.kff(p + dp), -np.array([self.nlf.rf(p + dp), load]).T)

            # calculate correction of proportional load parameter
            # note: p and dp are passed independently (instead of p + dp), as dp is used for root selection
            try:
                ddy = self.constraint.corrector(self.nlf, p, dp, ddx)
                self.logger.debug("Corrector %d: ddy = %+f" % (iterative_counter, ddy))
            except DiscriminantError:
                raise DiscriminantError

            dp += self.nlf.ddp(p + dp, ddx, ddy) # calculate correction based on iterative load parameter and update incremental state

            #endregion

            tries.append(p + dp) # add attempt to tries

            r = self.get_r(p + dp) # retrieve residual load for termination criteria

            # check if maximum number of corrections is not exceeded
            if iterative_counter > self.maximum_iterates:
                raise CounterError("Maximum number of corrections %2d >%2d" % (iterative_counter, self.maximum_iterates))

        self.logger.debug("Number of corrections: %d" % iterative_counter)
        return dp, iterative_counter, tries

    def get_r(self, p):
        """
        Retrieve residual load at state p

        :param p: state
        :return: residual load
        """
        r = np.array([])
        if self.nlf.nf:
            rf = self.nlf.rf(p)
            r = np.append(r, rf)
        if self.nlf.np:
            r = np.append(r, self.nlf.rp(p))
        return r


class IncrementalSolver:
    """
    The IncrementalSolver solves a given system of nonlinear equations by pseudo-time stepping.
    """
    def __init__(self, solution_method: IterativeSolver,
                 name: str = "MyIncrementalSolver", logging_level: int = logging.DEBUG) -> None:
        """
        Initialization of the incremental solver.

        :param solution_method: type of solution method used to find next equilibrium state

        Note: currently this class acts as a function, more functionality is expected in future.
        For example, currently "only" a single solution_method is used and the type of load increment is fixed.
        """
        self.solution_method = solution_method
        self.maximum_increments = 100

        self.__name__ = name

        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + name)

    def __call__(self, p: Point) -> Tuple[List[Point], List[List[Point]]]:
        """
        The __call__ of IncrementalSolver finds a range of equilibrium points given some initial equilibrium point.


        :param p: initial equilibrium state
        :return: a list of equilibrium solutions (Points), and a list of lists of attempted points
        """
        self.logger.debug("Invoking incremental solver")

        # Note: it is assumed the starting guess is an equilibrium point!
        equilibrium_solutions = [p] # adds initial point to equilibrium solutions

        incremental_counter = 0 # counts total number of increments
        iterative_counter = 0 # counts total number of iterates (cumulative throughout increments)
        tries_storage = [] # stores the attempted states of equilibrium (multiple per increment)

        try:
            # currently very simple termination criteria (load proportionality parameter termination criteria)
            # ideally terminated at p.y == 1.0
            while p.y <= 1.0:
                incremental_counter += 1 # increase incremental counter

                # invoke solution method to find incremental state
                try:
                    dp, iterates, tries = self.solution_method(equilibrium_solutions)
                except (DiscriminantError, CounterError) as error:
                    self.logger.warning("{}: {}".format(type(error).__name__, error.args[0]))

                p = p + dp # add incremental state to current state (if equilibrium found)

                self.logger.debug("New equilibrium point found at dp.y = %+f in %d iterates, new p.y = %+f " % (dp.y, iterates, p.y))

                iterative_counter += iterates # add iterates of current search to counter
                tries_storage.append(tries) # store tries of current increment to storage

                equilibrium_solutions.append(p) # append equilibrium solution to storage

                # terminate algorithm if too many increments are used
                if incremental_counter > self.maximum_increments:
                    raise CounterError("Maximum number of increments %2d > %2d" % (incremental_counter, self.maximum_increments))

        except CounterError as error:
            self.logger.warning("{}: {}".format(type(error).__name__, error.args[0]))

        self.logger.info("Total number of increments: %d" % incremental_counter)
        self.logger.info("Total number of iterates: %d" % iterative_counter)

        return equilibrium_solutions, tries_storage

class CustomFormatter(logging.Formatter):
    white = '\x1b[5m'
    green = '\x1b[92m'
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\u001b[31m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'
    format = "%(name)-6s %(levelname)-8s %(message)s"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger(name, logging_level, formatter: logging.Formatter = None):
    # formatter
    formatter = formatter if formatter is not None else logging.Formatter('%(levelname)s: %(name)s - %(message)s')

    # handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # logger
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging_level)
    logger.propagate = False
    return logger