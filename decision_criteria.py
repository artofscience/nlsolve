from operator import ge
from typing import List
from typing import Callable
from utils import Structure, Point


class LoadTermination:
    def __init__(self, threshold: float = 1.0, margin: float = 1.0):
        self.threshold = threshold
        self.margin = margin
        self.exceed = False
        self.accept = False

    def __call__(self, problem: Structure, p: List[Point], dp):
        new_point = p[-1] + dp
        self.exceed = ge(new_point.y, self.threshold)
        self.accept = (self.threshold < new_point.y < self.threshold + self.margin)