from operator import ge
from typing import List
from typing import Callable
from utils import Problem, Point
from numpy.linalg import eigvals
from abc import ABC, abstractmethod
from operator import lt, gt
import logging
from criteria import CriterionBase, Criteria


# class TerminationCriteria(Criteria):
#     """A boolean combination of two ``Criterion`` instances of type termination.
#
#     This class keeps track of two criteria, e.g. "left" and "right". These are
#     combined given the provided operator, typically ``__and__`` or ``__or__``.
#     This class simplifies chaining of various boolean operations with multiple
#     (sub)classes from ``Criterion``.
#     """
#
#     def __call__(self, problem: Problem, p: Point, ddy: float) -> bool:
#         """Ensure both criteria are called when called."""
#         done = self.operator(bool(self.left(problem, p, ddy)), bool(self.right(problem, p, ddy)))
#         if done:
#             self.logger.info("Combined criteria satisfied")
#         return done

class DecisionCriterium(ABC):
    def __init__(self, threshold: float = 1.0, operator: Callable = ge, nmargin: float = 0.0, pmargin: float = 0.0):
        self.operator = operator
        self.threshold = threshold
        self.nmargin = nmargin
        self.pmargin = pmargin
        self.exceed = False
        self.accept = False

    def __call__(self, problem: Problem, p: List[Point], dp, y, dy):
        point = p[-1] + dp
        value = self.value(problem, point, y + dy)
        self.exceed = self.operator(value, self.threshold)
        self.accept = self.threshold - self.nmargin < value < self.threshold + self.pmargin

    @abstractmethod
    def value(self, problem: Problem, p: Point, y):
        pass

class LoadTermination(DecisionCriterium):
    def __init__(self, threshold: float = 1.0, margin: float = 1000.0):
        super().__init__(threshold, pmargin=margin)

    def value(self, problem: Problem, p: Point, y):
        return y

class EigenvalueTermination(DecisionCriterium):
    def __init__(self, threshold: float = 0.0, margin: float = 0.1):
        super().__init__(threshold, lt, nmargin=margin)

    def value(self, problem: Problem, p: Point, y):
        return min(eigvals(problem.kff(p)))

class EigenvalueChangeTermination:
    def __init__(self, margin: float = 0.1):
        self.margin = margin
        self.change = False

    def __call__(self, problem: Problem, p: List[Point], dp, y, dy):
        point = p[-1] + dp

        mu0 = sum(eigvals(problem.kff(p[-1])) < 0)

        eigs = eigvals(problem.kff(point))
        mu1 = sum(eigs < 0)

        value = min(abs(eigs))

        self.change = (mu0 != mu1)
        self.exceed = self.change and value > self.margin
        self.accept = self.change and value < self.margin

