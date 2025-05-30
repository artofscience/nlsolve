import numpy as np
import pytest

from constraints import State, NewtonRaphson
from core import IterativeSolver
from utils import Structure, Point


class Spring1D(Structure):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def ff(self) -> State:
        return np.array([1.0], dtype=float)

    def gf(self, p: Point) -> State:
        return self.k * p.qf

    def kff(self, p: Point) -> State:
        return np.array([[self.k]])


@pytest.mark.parametrize('k', [0.1, 1, 10])
def test_spring1d(k):
    solver = IterativeSolver(Spring1D(k), NewtonRaphson(dl=1.0), name=str(k))
    initial_point = Point(qf=np.array([0.0]), ff=np.array([0.0]))
    solution, _, _ = solver([initial_point])
    assert solution.qf[0] == pytest.approx(1 / k)


class Spring2DLoad(Structure):
    def __init__(self, k: tuple[float] = (1.0, 1.0)):
        super().__init__()
        self.k = k

    def ff(self) -> State:
        return np.array([0.0, 1.0], dtype=float)

    def kff(self, p: Point) -> State:
        return np.array([[self.k[0] + self.k[1], -self.k[1]], [-self.k[1], self.k[1]]])

    def gf(self, p: Point) -> State:
        return self.kff(p) @ p.qf


@pytest.mark.parametrize('k1', [0.1, 1.0, 10.0])
@pytest.mark.parametrize('k2', [0.1, 1.0, 10.0])
def test_spring2dload(k1, k2):
    solver = IterativeSolver(Spring2DLoad((k1, k2)), NewtonRaphson(dl=1.0), name=str(k1) + str(k2))
    initial_point = Point(qf=np.zeros(2), ff=np.zeros(2))
    solution, _, _ = solver([initial_point])
    assert solution.qf[-1] == pytest.approx(1 / k1 + 1 / k2)


class Spring2DMotion(Structure):
    def __init__(self, k: tuple[float] = (1.0, 1.0)):
        super().__init__()
        self.k = k

    def ff(self) -> State:
        return np.array([0.0], dtype=float)

    def up(self) -> State:
        return np.array([1.0], dtype=float)

    def kff(self, p: Point) -> State:
        return np.array([[self.k[0] + self.k[1]]])

    def kpp(self, p: Point) -> State:
        return np.array([[self.k[1]]])

    def kfp(self, p: Point) -> State:
        return np.array([[-self.k[1]]])

    def kpf(self, p: Point) -> State:
        return self.kfp(p)

    def gp(self, p: Point) -> State:
        return self.kpf(p) @ p.qf + self.kpp(p) @ p.qp

    def gf(self, p: Point) -> State:
        return self.kff(p) @ p.qf + self.kfp(p) @ p.qp


@pytest.mark.parametrize('k1', [0.1, 1.0, 10.0])
@pytest.mark.parametrize('k2', [0.1, 1.0, 10.0])
def test_spring2dmotion(k1, k2):
    solver = IterativeSolver(Spring2DMotion((k1, k2)), NewtonRaphson(dl=1.0), name=str(k1) + str(k2))
    initial_point = Point(qf=np.zeros(1), qp=np.zeros(1), ff=np.zeros(1), fp=np.zeros(1))
    solution, _, _ = solver([initial_point])
    assert solution.qf[0] == pytest.approx(k2 / (k1 + k2))
