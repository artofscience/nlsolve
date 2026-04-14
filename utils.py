from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

State = np.ndarray[float] | None
from matplotlib import pyplot as plt
from itertools import cycle


class Plotter:
    colours = cycle(['black', 'red', 'green', 'blue'])

    def __init__(self, linestyle='-', marker='o'):
        self.linestyle = linestyle
        self.marker = marker

    def __call__(self, solution, idq, idf):
        plt.plot([i.q[idq] for i in solution], [i.f[idf] for i in solution],
                 marker=self.marker,
                 linestyle=self.linestyle,
                 color=next(self.colours))



class Problem(ABC):
    """
    Interface of a nonlinear function to the nonlinear solver.

    The external / internal / residual load, motion and stiffness matrix are partitioned based on the free and prescribed degrees of freedom.
    Both the free and prescribed degrees of freedom can be of dimension 0, 1 or higher.
    If dim(free) = 0, then dim(prescribed) > 0 and vice versa.
    That is, either external_load OR prescribed_motion OR BOTH are to be provided.
    """

    def __init__(self, nlf, ixf=None, ixp=None):
        self.nlf = nlf

        self.ixf = ixf if ixf is not None else []
        self.ixp = ixp if ixp is not None else []

        self.nf = len(self.ixf)
        self.np = len(self.ixp)
        self.n = self.nf + self.np

    def external_load(self, p: Point):
        return np.zeros(self.nf, dtype=float)

    def external_state(self, p: Point):
        return np.zeros(self.np, dtype=float)

    def jac_external_state(self, p: Point, y: float = 0.0):
        pass

    def jac_external_load(self, p: Point, y: float = 0.0):
        pass


    def g(self, p: Point) -> State:
        return self.nlf.force(p.q)

    def gp(self, p: Point) -> State:
        return self.g(p)[self.ixp]

    def gf(self, p: Point) -> State:
        return self.g(p)[self.ixf]

    def r(self, p: Point) -> State:
        return self.g(p) - p.f

    def rf(self, p: Point) -> State:
        return self.r(p)[self.ixf]

    def rp(self, p: Point) -> State:
        return self.r(p)[self.ixp]

    def dg(self, p: Point, y: float = 0.0) -> State:
        return self.nlf.jacobian(p.q, y)

    def dgff(self, p: Point, y: float = 0.0):
        return self.dg(p, y)[self.ixf, :][:, self.ixf]

    def dgpp(self, p: Point, y: float = 0.0):
        return self.dg(p, y)[self.ixp, :][:, self.ixp]

    def dgfp(self, p: Point, y: float = 0.0):
        return self.dg(p, y)[self.ixf, :][:, self.ixp]

    def dgpf(self, p: Point, y: float = 0.0):
        return self.dg(p, y)[self.ixp, :][:, self.ixf]

    def kff(self, p: Point, y: float = 0.0):
        tmp = self.dgff(p, y)
        # if x := self.jac_external_load(p) is not None:
        #     tmp -= y * x
        # if x := self.jac_external_state(p) is not None:
        #     tmp += y * self.dgfp(p, y) @ x
        return tmp

    def kpf(self, p: Point, y: float = 0.0) -> State:
        tmp = self.dgpf(p, y)
        # if x := self.jac_external_state(p) is not None:
        #     tmp += y * self.dgpp(p, y) @ x
        return tmp

    def loadf(self, p: Point, y: float = 0.0) -> State:
        return self.external_load(p) - self.dgfp(p, y) @ self.external_state(p)

    def loadp(self, p: Point, y: float = 0.0) -> State:
        return self.dgpp(p, y) @ self.external_state(p)

    def point(self, qf, qp, ff, fp):
        q = np.zeros(self.n)
        f = np.zeros(self.n)
        if self.nf:
            q[self.ixf] = qf
            f[self.ixf] = ff
        if self.np:
            q[self.ixp] = qp
            f[self.ixp] = fp
        return Point(q, f)

    def empty_point(self):
        return Point(np.zeros(self.n), np.zeros(self.n))

    def qf(self, p):
        return p.q[self.ixf]

    def qp(self, p):
        return p.q[self.ixp]

    def ff(self, p):
        return p.f[self.ixf]

    def fp(self, p):
        return p.f[self.ixp]


class Point:
    def __init__(self, q: State = 0.0, f: State = 0.0) -> None:
        self.q = self.make_float(q)
        self.f = self.make_float(f)

    def __iadd__(self, other: Point) -> Point:
        """
        Adds the content of another Point to this Point.

        :param other: another Point object
        :return: sum of Points
        """
        self.q += other.q
        self.f += other.f
        return self

    def __add__(self, other: Point) -> Point:
        """
        Addition of two points, returing a third Point.

        :param other: another Point object
        :return: a third Point object that is the addition
        """
        out = deepcopy(self)
        out += other
        return out

    def __rmul__(self, other: Point) -> Point:
        """
        Multiplications of two point entries.

        Note rmul makes a deepcopy of itself!

        :param other: another Point
        :return: a copy of itself with the entries multiplied by the other Points entries
        """
        out = deepcopy(self)
        out.q *= other
        out.f *= other
        return out

    def is_zero(self, tol) -> bool:
        norm_f = np.linalg.norm(self.f)
        norm_u = np.linalg.norm(self.q)
        return norm_f < tol and norm_u < tol

    def norm(self):
        norm_f = np.linalg.norm(self.f)
        norm_u = np.linalg.norm(self.q)
        return np.linalg.norm(np.array([norm_f, norm_u]))


    @staticmethod
    def make_float(x):
        return x.astype(float) if type(x) is np.ndarray else x
