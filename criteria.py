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

    def __init__(self, threshold: int = 50):
        self.count = 0
        self.threshold = threshold

    def __bool__(self) -> bool:
        """Evaluates its criterion function and updates it status."""
        self.count += 1
        return self.count > self.threshold


class Criterion(ABC):
    def __init__(self, name: str = None, logging_level: int = logging.INFO) -> None:

        self.__name__ = name if name is not None else (self.__class__.__name__ + " " + str(id(self)))
        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing an " + self.__class__.__name__ + " called " + self.__name__)

    @abstractmethod
    def __call__(self, problem: Structure, p: Point) -> bool:
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

class Criteria(Criterion):
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

    def __call__(self, problem: Structure, p: Point) -> bool:
        """Ensure both criteria are called when called."""
        a = self.operator(bool(self.left(problem, p)), bool(self.right(problem, p)))
        if a:
            self.logger.info("Convergence criteria satisfied")
        return a

class ConvergenceCriterion(Criterion, ABC):
    def __init__(self, fnc: Callable = lambda x, y: np.linalg.norm(x.r(y)),
                 is_x_then: Callable = lt,
                 threshold: float = 1.0,
                 name: str = None, logging_level: int = logging.INFO,
                 ):
        super().__init__(name, logging_level)
        self.fcn = fnc
        self.ref = None
        self.threshold = threshold
        self.operator = is_x_then
        self.point_old = None

    def __call__(self, nlf: Structure, p: Point) -> bool:
        if self.point_old is None:
            self.point_old = 0.0 * p
        value = self.fcn(nlf, p, self.point_old if self.point_old is not None else p)
        self.point_old = 1.0 * p
        if not self.ref:
            self.ref = 1.0 * value
        done = self.operator(value, self.threshold)
        if done:
            self.logger.info("Convergence criterion satisfied, value changed from %.2e exceeding threshold %.2e to %.2e" % (
        self.ref, self.threshold, value))
        return done