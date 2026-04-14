from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

# from constraints import Constraint, GeneralizedArcLength
from controllers import Controller, Adaptive
from criteria import Counter, residual_norm, divergence_default, termination_default
from logger import CustomFormatter, create_logger
from utils import Problem, Point

State = np.ndarray[float] | None


class CounterError(Exception):
    pass


class DivergenceError(Exception):
    pass


class TerminationError(Exception):
    pass

class Out:
    def __init__(self):
        self.solutions = None
        self.time = None
        self.tries = None

class IncrementalSolver:
    """
    The IncrementalSolver solves a given system of nonlinear equations by pseudo-time stepping.
    """

    def __init__(self, solution_method: IterativeSolver,
                 controller: Controller = None,
                 p: Point = None,
                 name: str = None,
                 logging_level: int = logging.DEBUG,
                 maximum_increments: int = 1000,
                 terminated=termination_default(),
                 reset: bool = True) -> None:
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

        # controller
        self.controller = controller if controller is not None else Adaptive()

        # initial point
        self.p0 = p if p is not None else self.solution_method.problem.empty_point()

        self.reset = reset

        # termination
        self.maximum_increments: int = maximum_increments
        self.terminated = terminated

        # logging
        self.__name__ = name if name is not None else (self.__class__.__name__ + " " + str(id(self)))
        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + self.__name__)

        self.y = 0.0

        self.history = []

    def __call__(self, pd: Point = None, controller: Controller = None,
                 terminated=None, reset=None, y = None) -> Out:
        """
        The __call__ of IncrementalSolver finds a range of equilibrium points given some initial equilibrium point.


        :param p: initial equilibrium state
        :param controller: controller of the pseud-time step size
        :return: a list of equilibrium solutions (Points), and a list of lists of attempted points
        """
        if terminated is not None:
            self.terminated = terminated

        if controller is not None:
            self.controller = controller

        # if constraint is not None:
        #     self.solution_method.constraint = constraint

        if reset is not None:
            self.reset = reset

        if self.reset:
            self.y = 0.0
            self.controller.reset()

        if y is not None:
            self.y = y

        time = [self.y]

        p = self.p0 if self.reset or len(self.history) < 1 else self.history[-1].solutions[-1]
        p = pd if pd is not None else p

        self.logger.debug("Invoking incremental solver")

        # Note: it is assumed the starting guess is an equilibrium point!
        equilibrium_solutions = [p]  # adds initial point to equilibrium solutions

        incremental_counter = 0  # counts total number of succesful increments
        incremental_tries = 0  # counts total number of times the iterative solver is invoked
        iterative_counter = 0  # counts total number of iterates (cumulative throughout increments)
        iterative_tries = 0  # counts total number of iterates (cumulative throughout increments)

        tries_storage = []  # stores the attempted states of equilibrium (multiple per increment)

        while True:

            incremental_counter += 1

            print("")

            # invoke solution method to find incremental state
            while True:
                try:
                    incremental_tries += 1
                    self.logger.info("Invoking iterative solver for %d-th time to find %d-th equilibrium point" % (
                    incremental_tries, incremental_counter))

                    predictor_solutions = [self.history[-1].solutions[-2]] + equilibrium_solutions if len(
                        self.history) and not self.reset else equilibrium_solutions
                    dp, dy, iterates, tries = self.solution_method(predictor_solutions, self.y, self.controller.value)
                    iterative_tries += iterates
                    self.terminated(self.solution_method.problem, equilibrium_solutions, dp, self.y + dy, dy)
                    if self.terminated.exceed and not self.terminated.accept:
                        raise TerminationError("Threshold exceeded, but step not accepted: reduce step size!", iterates)
                    else:
                        break

                except (ValueError, CounterError, DivergenceError) as error:
                    self.logger.error("{}: {}".format(type(error).__name__, error.args[0]))
                    iterative_tries += error.args[1]
                    self.logger.error("Iterative solver aborted after %d iterates" % error.args[1])
                    self.logger.warning("Decrease characteristic length of constraint equation and try again!")
                    self.controller.decrease()  # decrease the characteristic length of the constraint

                except TerminationError as error:
                    iterative_tries += error.args[1]
                    self.logger.warning("Succesful step in %d iterates" % error.args[1])
                    self.logger.warning("{}: {}".format(type(error).__name__, error.args[0]))
                    self.logger.warning("Decrease characteristic length of constraint equation and try again!")
                    self.controller.decrease()  # decrease the characteristic length of the constraint

            p = p + dp  # add incremental state to current state (if equilibrium found)
            self.y += dy

            self.logger.debug(
                "New equilibrium point found at dy = %+f in %d iterates, new y = %+f " % (
                    dy, iterates, self.y))

            equilibrium_solutions.append(p)  # append equilibrium solution to storage
            time.append(float(self.y))

            iterative_counter += iterates  # add iterates of current search to counter
            tries_storage.append(tries)  # store tries of current increment to storage

            self.logger.info("Total number of increments: %d" % incremental_tries)
            self.logger.debug("Total number of iterates: %d" % iterative_tries)

            self.logger.info("Total number of succesful increments: %d" % incremental_counter)
            self.logger.debug("Total number of effective iterates: %d" % iterative_counter)

            if self.terminated.accept:
                self.logger.info("Termination criteria satisfied: stepper aborted.")
                break

            # terminate algorithm if too many increments are used
            if incremental_counter >= self.maximum_increments:
                self.logger.error(
                    "Maximum number of increments %2d >= %2d".format(incremental_counter, self.maximum_increments))
                break

            self.controller.increase()  # increase the characteristic length of the constraint for next iterate

        self.out = Out()
        self.out.solutions = equilibrium_solutions
        self.out.tries = tries_storage
        self.out.time = time
        self.history.append(self.out)
        return self.out

class IterativeSolver:
    """
    The IterativeSolver is the core of this API, its function is to find a next equilibrium point,
    that is solving the provided system of nonlinear equations given some constraint function.
    """

    def __init__(self, problem: Problem,
                 converged=None, diverged=None,
                 name: str = None, logging_level: int = logging.DEBUG,
                 maximum_corrections: int = 1000) -> None:
        """
        Initialization of the iterative solver.

        :param constraint: constraint function used to solve the system of nonlinear equations

        Note: currently this class acts as a function, more functionality is expected in future.
        """

        # create some aliases for commonly used functions
        self.converged = converged if converged is not None else residual_norm(1e-10)
        self.diverged = diverged if diverged is not None else divergence_default()
        self.problem: Problem = problem  # nonlinear system of equations
        # self.constraint = constraint if constraint is not None else GeneralizedArcLength()  # constraint function used (operates on nlf)
        self.maximum_corrections: int = maximum_corrections  # maximum allowed number of iterates before premature termination

        self.cqf = 1.0
        self.cqp = 1.0
        self.cff = 1.0
        self.cfp = 1.0

        self.default_positive_direction = True

        self.__name__ = name if name is not None else (self.__class__.__name__ + " " + str(id(self)))

        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + self.__name__)

    def solve(self, ddqf: np.ndarray, ddfp: np.ndarray, p: Point, y: float = 0.0):
        # solve the system of equations [-kff @ ddx1 = ff + kfp @ up] at state = p
        # note: for predictor ddx0 = 0, hence only a single rhs for this solve
        if self.problem.nf:
            # ddx[:, 1] = np.linalg.solve(self.nlf.kff(p), self.nlf.load(p))
            # Consider there is no equilibrium (yet)
            ddqf[:, :] = np.linalg.solve(self.problem.kff(p, y),
                                         np.array(
                                             [-self.problem.rf(p, y), self.problem.loadf(p, y)]).T)

        if self.problem.np:
            ddfp[:, 0] = self.problem.rp(p, y)
            ddfp[:, 1] = self.problem.loadp(p, y)
            if self.problem.nf:
                ddfp[:, 0] += self.problem.kpf(p, y) @ ddqf[:, 0]
                ddfp[:, 1] += self.problem.kpf(p, y) @ ddqf[:, 1]

        return ddqf, ddfp

    def __call__(self, sol: List[Point], y: float = 0.0, length: float = 0.0) -> Tuple[Point, float, int, List[Point]]:
        self.logger.debug("Starting iterative solver")
        self.converged.reset()
        self.diverged.reset()

        self.dl = length  # set characteristic length of constraint

        p = sol[-1]  # takes the initial equilibrium point (what if this is not in equilibrium?)
        tries = [p]  # initialize storage for attempted states and add initial point

        # region PREDICTOR

        # initialize structure of solve return values if free degrees of freedom
        ddqf = np.zeros((self.problem.nf, 2), dtype=float) if self.problem.nf else None
        ddfp = np.zeros((self.problem.np, 2), dtype=float) if self.problem.np else None

        ddqf[:, :], ddfp[:, :] = self.solve(ddqf, ddfp, p, y)


        # call to the predictor of the constraint function returning iterative load parameter
        # note it has access to previous equilibrium points (sol) and dp = 0
        # note for first iterate dy = ddy and dp = ddp
        try:
            ddy = self.predictor(p, sol, ddqf, ddfp, y)
        except ValueError as error:
            self.logger.error("{}: {}".format(type(error).__name__, error.args[0]))
            raise ValueError("A suitable prediction cannot be found!", 0)

        dp = self.ddp(p, ddqf, ddfp, y, ddy)  # calculate prediction based on iterative load parameter
        dy = 1.0 * ddy
        self.logger.debug("Predictor 0: ddy = %+e, norm(r) = %+e" % (ddy, np.linalg.norm(self.problem.r(p + dp, y + dy))))

        # endregion



        counter = Counter(self.maximum_corrections)

        # make corrections until termination criteria are met
        while True:
            if counter:
                self.logger.error("Maximum number of corrections %2d > %2d" % (counter.count, counter.threshold),
                                   counter.count)
                # raise CounterError("Maximum number of corrections %2d > %2d" % (counter.count, counter.threshold),
                #                    counter.count)
                break

            if self.converged(self.problem, p + dp, y + dy, ddy):
                # terminate the loop if converged
                break

            # if self.diverged(self.nlf, p + dp, ddy):
            #     # raise error if diverged
            #     raise DivergenceError("Solver diverged!", counter.count)

            # region CORRECTOR

            ddqf[:, :], ddfp[:, :] = self.solve(ddqf, ddfp, p + dp, y + dy)

            # calculate correction of proportional load parameter
            # note: p and dp are passed independently (instead of p + dp), as dp is used for root selection
            try:
                ddy = self.corrector(p, dp, ddqf, ddfp, y)
            except ValueError as error:
                self.logger.error("{}: {}".format(type(error).__name__, error.args[0]))
                raise ValueError("A suitable correction cannot be found!", counter.count)

            dp += self.ddp(p + dp, ddqf, ddfp, y + dy, ddy)  # calculate correction based on iterative load parameter and update incremental state
            dy += ddy

            self.logger.debug(
                "Corrector %d: ddy = %+e, norm(r) = %+e" % (counter.count, ddy, np.linalg.norm(self.problem.r(p + dp, y + dy))))


            # endregion

            tries.append(p + dp)  # add attempt to tries

        return dp, dy, counter.count, tries

    def ddp(self, p: Point, ddxf: np.ndarray, ddxp: np.ndarray, y: float, ddy: float) -> Point:
        """
        Provides the iterative updated state given some iterative load parameter.

        :param p: current state (p + dp)
        :param u: resultants from solve
        :param y: iterative load parameter
        :return:
        """
        ddqf, ddqp, ddff, ddfp = 0.0, 0.0, 0.0, 0.0

        if self.problem.nf:
            ddqf = ddxf[:, 0] + ddy * ddxf[:, 1]
            ddff = ddy * self.problem.external_load(p)
        if self.problem.np:
            ddqp = ddy * self.problem.external_state(p)
            ddfp = ddxp[:, 0] + ddy * ddxp[:, 1]
        return self.problem.point(ddqf, ddqp, ddff, ddfp)

    def predictor(self, p: Point, sol: List[Point], ddqf: np.ndarray, ddfp: np.ndarray, y: float = 0.0) -> float:
        roots = self.get_roots_predictor(p, ddqf, ddfp, self.dl)
        cps = [self.ddp(p, ddqf, ddfp, y, i) for i in roots]
        return self.select_root_predictor(p, sol, cps, roots)

    def corrector(self, p: Point, dp: Point, ddqf: np.ndarray, ddfp: np.ndarray, y: float = 0.0) -> float:
        try:
            roots = self.get_roots_corrector(p, dp, ddqf, ddfp, self.dl)
        except ValueError as error:
            self.logger.error("{}: {}".format(type(error).__name__, error.args[0]))
            raise ValueError("Roots of constraint equation for the corrector cannot be found!")

        cps = [self.ddp(p + dp, ddqf, ddfp, y, i) for i in roots]
        return self.select_root_corrector(dp, cps, roots)

    def get_roots_predictor(self, p: Point, ddqf: np.ndarray, ddfp: np.ndarray, dl: float):
        a = 0.0
        if self.problem.nf:
            tmp1 = ddqf[:, 1]
            tmp2 = self.problem.external_load(p)
            a += self.cqf * np.dot(tmp1, tmp1) + self.cff * np.dot(tmp2, tmp2)
        if self.problem.np:
            tmp3 = ddfp[:, 1]
            tmp4 = self.problem.external_state(p)
            a += self.cfp * np.dot(tmp3, tmp3) + self.cqp * np.dot(tmp4, tmp4)

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, p: Point, dp: Point, ddqf: np.ndarray, ddfp: np.ndarray, dl: float):
        nlf = self.problem

        a = np.zeros(3)

        a[2] -= dl ** 2


        if nlf.nf:
            tmp = nlf.qf(dp) + ddqf[:, 0]

            a[0] += np.dot(ddqf[:, 1], ddqf[:, 1])
            a[0] += np.dot(nlf.external_load(p), nlf.external_load(p))
            a[1] += 2 * np.dot(ddqf[:, 1], tmp)
            a[1] += 2 * np.dot(nlf.ff(dp), nlf.external_load(p))
            a[2] += np.dot(tmp, tmp)
            a[2] += np.dot(nlf.ff(dp), nlf.ff(dp))
        if nlf.np:
            a[0] += np.dot(nlf.external_state(p), nlf.external_state(p))
            a[1] += 2 * np.dot(nlf.external_state(p), nlf.qp(dp))
            a[2] += np.dot(nlf.qp(dp), nlf.qp(dp))

            tmp = nlf.fp(dp) + ddfp[:, 0]

            a[0] += np.dot(ddfp[:, 1], ddfp[:, 1])
            a[1] += 2 * np.dot( ddfp[:, 1], tmp)
            a[2] += np.dot(tmp, tmp)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])

    def select_root_predictor(self, p: Point, sol: List[Point], cps: List[Point], roots) -> float:
        nlf = self.problem

        if len(sol) < 2:
            return max(roots) if self.default_positive_direction else min(roots)

        else:
            if nlf.nf:
                vec1 = np.append(nlf.qf(sol[-2]) - nlf.qf(p) - nlf.qf(cps[0]),
                                 nlf.ff(sol[-2]) - nlf.ff(p) - nlf.ff(cps[0]))
                vec2 = np.append(nlf.qf(sol[-2]) - nlf.qf(p) - nlf.qf(cps[1]),
                                 nlf.ff(sol[-2]) - nlf.ff(p) - nlf.ff(cps[1]))

            if nlf.np:
                vec1 = np.append(nlf.qp(sol[-2]) - nlf.qp(p) - nlf.qp(cps[0]),
                                 nlf.fp(sol[-2]) - nlf.fp(p) - nlf.fp(cps[0]))
                vec2 = np.append(nlf.qp(sol[-2]) - nlf.qp(p) - nlf.qp(cps[1]),
                                 nlf.fp(sol[-2]) - nlf.fp(p) - nlf.fp(cps[1]))

                if nlf.nf:
                    vec11 = np.append(nlf.qf(sol[-2]) - nlf.qf(p) - nlf.qf(cps[0]),
                                      nlf.ff(sol[-2]) - nlf.ff(p) - nlf.ff(cps[0]))
                    vec12 = np.append(nlf.qp(sol[-2]) - nlf.qp(p) - nlf.qp(cps[0]),
                                      nlf.fp(sol[-2]) - nlf.fp(p) - nlf.fp(cps[0]))
                    vec1 = np.append(vec11, vec12)
                    vec21 = np.append(nlf.qf(sol[-2]) - nlf.qf(p) - nlf.qf(cps[1]),
                                      nlf.ff(sol[-2]) - nlf.ff(p) - nlf.ff(cps[1]))
                    vec22 = np.append(nlf.qp(sol[-2]) - nlf.qp(p) - nlf.qp(cps[1]),
                                      nlf.fp(sol[-2]) - nlf.fp(p) - nlf.fp(cps[1]))
                    vec2 = np.append(vec21, vec22)

            return roots[0] if np.linalg.norm(vec1) > np.linalg.norm(vec2) else roots[1]

    def select_root_corrector(self, dp: Point, cps: List[Point], roots) -> float:
        nlf = self.problem
        """
        This rule is based on the projections of the generalized correction vectors on the previous correction [Vasios, 2015].
        The corrector that forms the closest correction to the previous point is chosen.
        Note: this rule cannot be used in the first iteration since the initial corrections are equal to zero at the beginning of each increment.
        """
        if nlf.nf:
            cpd = lambda i: np.dot(nlf.qf(dp), nlf.qf(dp) + nlf.qf(cps[i]))
        if nlf.np:
            cpd = lambda i: np.dot(nlf.qp(dp), nlf.qp(dp) + nlf.qp(cps[i]))
            if nlf.nf:
                cpd = lambda i: np.dot(nlf.qf(dp), nlf.qf(dp) + nlf.qf(cps[i])) + np.dot(nlf.qp(dp),
                                                                                         nlf.qp(dp) + nlf.qp(cps[i]))

        return roots[0] if cpd(0) >= cpd(1) else roots[1]


