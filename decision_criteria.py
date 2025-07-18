from operator import ge
from typing import List
from typing import Callable
from utils import Structure, Point
from numpy.linalg import eigvals
from abc import ABC, abstractmethod
from operator import lt

class DecisionCriterium(ABC):
    def __init__(self, threshold: float = 1.0, operator: Callable = ge, nmargin: float = 0.0, pmargin: float = 0.0):
        self.operator = operator
        self.threshold = threshold
        self.nmargin = nmargin
        self.pmargin = pmargin
        self.exceed = False
        self.accept = False

    def __call__(self, problem: Structure, p: List[Point], dp):
        point = p[-1] + dp
        value = self.value(problem, point)
        self.exceed = self.operator(value, self.threshold)
        self.accept = self.threshold - self.nmargin < value < self.threshold + self.pmargin

    @abstractmethod
    def value(self, problem: Structure, p: Point):
        pass

class LoadTermination(DecisionCriterium):
    def __init__(self, threshold: float = 1.0, margin: float = 0.1):
        super().__init__(threshold, pmargin=margin)

    def value(self, problem: Structure, p: Point):
        return p.y

class EigenvalueTermination(DecisionCriterium):
    def __init__(self, threshold: float = 0.0, margin: float = 0.1):
        super().__init__(threshold, lt, nmargin=margin)

    def value(self, problem: Structure, p: Point):
        eig = min(eigvals(problem.kff(p)))
        print(eig)
        return eig