from __future__ import annotations

import sys
from abc import ABC
from copy import deepcopy
from typing import List, Tuple
from controllers import Controller

import numpy as np

State = np.ndarray[float] | None

class CounterError(Exception):
    pass


import logging


class IterativeSolver:
    """
    The IterativeSolver is the core of this API, its function is to find a next equilibrium point,
    that is solving the provided system of nonlinear equations given some constraint function.
    """

    def __init__(self, nlf: Structure, constraint,
                 name: str = None, logging_level: int = logging.DEBUG,
                 maximum_corrections: int = 1000) -> None:
        """
        Initialization of the iterative solver.

        :param constraint: constraint function used to solve the system of nonlinear equations

        Note: currently this class acts as a function, more functionality is expected in future.
        """

        # create some aliases for commonly used functions
        self.nlf: Structure = nlf  # nonlinear system of equations
        self.constraint = constraint  # constraint function used (operates on nlf)
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

        # dp = 0.0 * p # make a copy of the structure of p and reset all entries to zeros

        iterative_counter = 0  # start with 0 as we count the corrections only

        # region PREDICTOR

        # initialize structure of solve return values if free degrees of freedom
        ddx = np.zeros((self.nlf.nf, 2), dtype=float) if self.nlf.nf else None

        # solve the system of equations [-kff @ ddx1 = ff + kfp @ up] at state = p
        # note: for predictor ddx0 = 0, hence only a single rhs for this solve
        if self.nlf.nf:
            load = 1.0 * self.nlf.ff
            load -= self.nlf.kfp(p) @ self.nlf.up if self.nlf.np else 0.0  # adds to rhs if nonzero prescribed dof
            ddx[:, 1] = np.linalg.solve(self.nlf.kff(p), load)

        # call to the predictor of the constraint function returning iterative load parameter
        # note it has access to previous equilibrium points (sol) and dp = 0
        # note for first iterate dy = ddy and dp = ddp
        ddy = self.constraint.predictor(self.nlf, p, sol, ddx)
        self.logger.debug("Predictor 0: ddy = %+f" % ddy)

        dp = self.nlf.ddp(p, ddx, ddy)  # calculate prediction based on iterative load parameter

        # endregion

        # combine residuals associated to free and prescribed dofs (if available) for termiantion criteria
        r = self.get_r(p + dp)

        rnorm_ref = np.linalg.norm(r)
        rmax_ref = np.amax(np.abs(r))

        # make corrections until termination criteria are met
        while (rmax := np.amax(np.abs(r))) > 1e-9 or (rnorm := np.linalg.norm(r)) > 1e-6:

            iterative_counter += 1  # increase iterative solver counter

            # region CORRECTOR

            # solve the system of equations kff @ [ddx0, ddx1] = -[rf, ff + kfp @ up] at state = p + dp
            if self.nlf.nf:
                load = 1.0 * self.nlf.ff
                load -= self.nlf.kfp(p + dp) @ self.nlf.up if self.nlf.np else 0.0
                ddx[:, :] = np.linalg.solve(self.nlf.kff(p + dp), np.array([-self.nlf.rf(p + dp), load]).T)

            # calculate correction of proportional load parameter
            # note: p and dp are passed independently (instead of p + dp), as dp is used for root selection
            ddy = self.constraint.corrector(self.nlf, p, dp, ddx)
            self.logger.debug("Corrector %d: ddy = %+e" % (iterative_counter, ddy))

            dp += self.nlf.ddp(p + dp, ddx, ddy) # calculate correction based on iterative load parameter and update incremental state

            #endregion

            tries.append(p + dp)  # add attempt to tries

            r = self.get_r(p + dp)  # retrieve residual load for termination criteria

            # check if maximum number of corrections is not exceeded
            if iterative_counter > self.maximum_corrections:
                raise CounterError(
                    "Maximum number of corrections %2d >%2d" % (iterative_counter, self.maximum_corrections))

        self.logger.debug("Maximum absolute residual reduced from %e to %e" % (rmax_ref, rmax))
        self.logger.debug("Residual norm reduced from %e to %e" % (rnorm_ref, rnorm))
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
                 name: str = "MyIncrementalSolver", logging_level: int = logging.DEBUG,
                 maximum_increments: int = 1000) -> None:
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

        self.__name__ = name

        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + name)

    def __call__(self, p: Point, controller: Controller = Controller(0.1)) -> Tuple[List[Point], List[List[Point]]]:
        """
        The __call__ of IncrementalSolver finds a range of equilibrium points given some initial equilibrium point.


        :param p: initial equilibrium state
        :param controller: controller of the pseud-time step size
        :return: a list of equilibrium solutions (Points), and a list of lists of attempted points
        """
        self.logger.debug("Invoking incremental solver")

        # Note: it is assumed the starting guess is an equilibrium point!
        equilibrium_solutions = [p]  # adds initial point to equilibrium solutions

        incremental_counter = 0  # counts total number of increments
        iterative_counter = 0  # counts total number of iterates (cumulative throughout increments)
        tries_storage = []  # stores the attempted states of equilibrium (multiple per increment)

        try:
            # currently very simple termination criteria (load proportionality parameter termination criteria)
            # ideally terminated at p.y == 1.0
            while -1.0 < p.y < 1.0:
                incremental_counter += 1

                # invoke solution method to find incremental state
                while True:
                    try:
                        self.logger.info("Invoking iterative solver")
                        dp, iterates, tries = self.solution_method(equilibrium_solutions, controller.value)
                        break
                    except (ValueError, CounterError) as error:
                        self.logger.warning("{}: {}".format(type(error).__name__, error.args[0]))
                        controller.decrease() # decrease the characteristic length of the constraint

                p = p + dp  # add incremental state to current state (if equilibrium found)

                self.logger.debug(
                    "New equilibrium point found at dp.y = %+f in %d iterates, new p.y = %+f " % (
                        dp.y, iterates, p.y))

                equilibrium_solutions.append(p)  # append equilibrium solution to storage

                controller.increase()  # increase the characteristic length of the constraint for next iterate

                iterative_counter += iterates  # add iterates of current search to counter
                tries_storage.append(tries)  # store tries of current increment to storage

                # terminate algorithm if too many increments are used
                if incremental_counter >= self.maximum_increments:
                    raise CounterError(
                        "Maximum number of increments %2d >= %2d" % (incremental_counter, self.maximum_increments))

        except CounterError as error:
            self.logger.warning("{}: {}".format(type(error).__name__, error.args[0]))
            return equilibrium_solutions, tries_storage

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
