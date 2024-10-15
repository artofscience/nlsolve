"""Abstract implementation for convergence criteria."""
from abc import ABC, abstractmethod
from utils import Point, Structure
import logging
import operator
from operator import lt

from logger import CustomFormatter, create_logger
from typing import Callable

import numpy as np

class CounterError(Exception):
    pass

class Counter:
    """Enforces a maximum number of iterations.

    This keeps track of an internal counter that increments on each evaluation
    of ``__bool__``. Thus, when setting up a loop as follows, this criterion
    will ensure at most 50 iterations are performed.

    >>> terminated = Counter(50):
    >>> while not terminated
    """

    def __init__(self, threshold: int = 50) -> None:
        """
        Initializes the Counter, setting initial counter value and threshold
        :param threshold: the maximum number of iterations
        """
        self.count = 0
        self.threshold = threshold

    def __bool__(self) -> bool:
        """Evaluates its criterion function and updates it status."""
        self.count += 1
        return self.count > self.threshold


class CriterionBase(ABC):
    """
    Abstract class to setup a termination criterium (convergence, divergence).
    This class is dedicated to the use in IncrementalSolver or IterativeSolver class.
    """
    def __init__(self, name: str = None, logging_level: int = logging.INFO) -> None:
        """
        Initializes the criterion, sets up the logger.

        :param name: name of the criterion
        :param logging_level: logging level of the criterion
        """
        self.__name__ = name if name is not None else (self.__class__.__name__ + " " + str(id(self))[-3:])
        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + self.__name__)

    @abstractmethod
    def __call__(self, problem: Structure, p: Point, ddy: float) -> bool:
        """

        :param problem: the (non)linear function to be solved
        :param p: the state (Point)
        :param ddy: the change in load by predictor or corrector
        :return: bool
        """
        pass

    def __and__(self, other):
        """Return a combined ``Criteria`` from the ``and (&)`` operation."""
        return Criteria(self, other, operator.__and__)

    def __or__(self, other):
        """Return a combined ``Criteria`` from the ``or (|)`` operation."""
        return Criteria(self, other, operator.__or__)

    def __invert__(self):
        """Returns a combined ``Criteria`` with a flipped result.

        The boolean value is inverted by evaluating a "not equal", i.e.
        ``__ne__``, with respect to ``True``, causing the original value to be
        flipped in the returned, combined ``Criteria`` class.
        """
        return Criteria(self, lambda: True, operator.__ne__)

    @abstractmethod
    def reset(self):
        """
        Function that resets some class attributes, such that instance can be reused.
        :return: None
        """
        pass

class Criteria(CriterionBase):
    """A boolean combination of two ``Criterion`` instances.

    This class keeps track of two criteria, e.g. "left" and "right". These are
    combined given the provided operator, typically ``__and__`` or ``__or__``.
    This class simplifies chaining of various boolean operations with multiple
    (sub)classes from ``Criterion``.
    """

    def __init__(self, left, right, op,
                 name: str = None, logging_level: int = logging.INFO):
        super().__init__(name, logging_level)
        self.left, self.right = left, right
        self.operator = op

    def __call__(self, problem: Structure, p: Point, ddy: float) -> bool:
        """Ensure both criteria are called when called."""
        done = self.operator(bool(self.left(problem, p, ddy)), bool(self.right(problem, p, ddy)))
        if done:
            self.logger.info("Combined criteria satisfied")
        return done

    def reset(self):
        """Ensure both criteria are reset."""
        self.left.reset()
        self.right.reset()

class CriterionP(CriterionBase):
    """
    Criterion that works on the state.
    The input function is expected to only take a Point.

    One can define a function, e.g. the norm of the displacement vector,
    and compare to some float via an operator
    >>> def my_function(p: Point) -> float: return np.linalg.norm(p.u)
    >>> my_criterion = CriterionP(my_function, lt, 1.0)

    Alternativley one can use a lambda function.
    Example use to check whether the load is lower than some value:
    >>> my_criterion = CriterionP(lambda p: p.y, lt, 1.0)

    Next one can use the criterion via
    >>> while ~my_criterion(nlf, p, ddy):
    """
    def __init__(self, fnc: Callable = lambda p: p.y,
                 is_x_then: Callable = lt,
                 threshold: float = 1.0,
                 name: str = None, logging_level: int = logging.INFO,
                 ) -> None:
        super().__init__(name, logging_level)
        self.fnc = fnc
        self.ref = None # define a reference value for logging purposes
        self.threshold = threshold
        self.operator = is_x_then

    def __call__(self, nlf: Structure, p: Point, ddy: float) -> bool:
        value = self.call_to_fnc(nlf, p, ddy) # the value to work with it the output of the provided function
        self.ref = 1.0 * value if self.ref is None else self.ref # set reference if not done yet
        done = self.operator(value, self.threshold) # compare value to given threshold
        if done: # print to console that criterion is satisfied and provide info on values (reference, threshold and value)
            self.logger.info(
                "Criterion satisfied, value changed from %.2e exceeding threshold %.2e to %.2e" % (
                    self.ref, self.threshold, value))
        return done

    def reset(self):
        self.ref = None # reset reference value

    def call_to_fnc(self, nlf: Structure, p: Point, ddy: float) -> bool:
        return self.fnc(p) # only takes the Point

class CriterionX(CriterionP):
    """
    Criterion that works on both the nonlinear problem and the state.

    One can define a function, e.g. the norm of the residual vector,
    and compare to some float via an operator
    >>> def my_function(nlf: Structure, p: Point) -> float: return np.linalg.norm(nlf.r(p))

    Alternativley one can use a lambda function.
    Example use to check whether the maximum absolute value of the internal load of free dofs is lower than some value:
    >>> my_criterion = CriterionX(lambda nlf, p: np.max(np.abs(nlf.gf(p))), lt, 1.0)

    Next one can use the criterion similar to CriterionP.
    """
    def __init__(self, fnc: Callable = lambda nlf, p: np.linalg.norm(nlf.r(p)),
                 is_x_then: Callable = lt,
                 threshold: float = 1.0e-9,
                 name: str = None, logging_level: int = logging.INFO,
                 ):
        super().__init__(fnc, is_x_then, threshold, name, logging_level)

    def call_to_fnc(self, nlf: Structure, p: Point, ddy: float) -> bool:
        return self.fnc(nlf, p) # note: takes the nonlinear function and the state (point)

class CriterionXH(CriterionX):
    """
    Criterion that works on both the nonlinear problem and the state, just like CriterionX.
    However, an instance of this class also takes into account the previous state.

    One can define a function, e.g. the norm of the difference of residual vectors of two iterates,
    and compare to some float via an operator
    >>> def my_function(nlf: Structure, p: Point, p_old: Point) -> float:
    >>>     return np.linalg.norm(nlf.r(p) - nlf.r(p_old))

    Alternativley one can use a lambda function.
    Example use to check whether the maximum absolute value of the difference in internal load of free dofs between two iterates is lower than some value:
    >>> my_criterion = CriterionXH(lambda nlf, p, p_old: np.max(np.abs(nlf.gf(p) - nlf.gf(p_old))), lt, 1.0)

    Next one can use the criterion similar to CriterionP.
    """
    def __init__(self, fnc: Callable = lambda x, y, z: np.linalg.norm(x.r(y) - x.r(z)),
                 is_x_then: Callable = lt,
                 threshold: float = 1.0,
                 name: str = None, logging_level: int = logging.INFO):
        super().__init__(fnc, is_x_then, threshold, name, logging_level)
        self.point_old = None # initialize the old point

    def call_to_fnc(self, nlf: Structure, p: Point, ddy: float) -> bool:
        if self.point_old is None:
            self.point_old = 0.0 * p # set the old point of not done yet, here initialized to zero point
        value = self.fnc(nlf, p, self.point_old) # note the old point is the third input
        self.point_old = 1.0 * p # set the old point
        return value

    def reset(self):
        super().reset()
        self.point_old = None # reset the old point

class CriterionY(CriterionP):
    def __init__(self, fnc: Callable = lambda ddy: abs(ddy),
                 is_x_then: Callable = lt,
                 threshold: float = 1e-9,
                 name: str = None, logging_level: int = logging.INFO,
                 ):
        super().__init__(fnc, is_x_then, threshold, name, logging_level)

    def call_to_fnc(self, nlf: Structure, p: Point, ddy: float) -> bool:
        return self.fnc(ddy)

    def reset(self):
        super().reset()

class CriterionYH(CriterionY):
    def __init__(self, fnc: Callable = lambda ddy: abs(ddy),
                 is_x_then: Callable = lt,
                 threshold: float = 1e-9,
                 name: str = None, logging_level: int = logging.INFO,
                 value: float = 1e10):
        super().__init__(fnc, is_x_then, threshold, name, logging_level)
        self.old_ref = value
        self.ddy_old = value

    def call_to_fnc(self, nlf: Structure, p: Point, ddy: float) -> bool:
        value = self.fnc(ddy, self.ddy_old)
        self.ddy_old = 1.0 * ddy
        return value

    def reset(self):
        self.ddy_old = 1.0 * self.old_ref

def residual_norm(threshold, name: str ="Residual norm", logging_level: int = logging.INFO):
    """
    Creates instance of CriterionX that checks the 2-norm of the residual vector.
    """
    return CriterionX(lambda x, y: np.linalg.norm(x.r(y)), lt, threshold, name=name, logging_level=logging_level)