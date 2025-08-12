from __future__ import annotations

from abc import ABC
from copy import deepcopy

import numpy as np

State = np.ndarray[float] | None
from matplotlib import pyplot as plt
from itertools import cycle


class Plotter:
    colours = cycle(['black', 'red', 'green', 'blue'])

    def __call__(self, solution, idq, idf):
        plt.plot([i.q[idq] for i in solution], [i.f[idf] for i in solution],
                 marker='o',
                 linestyle='dashed',
                 color=next(self.colours))


def ddp(nlf, p: Point, u: np.ndarray, y: float) -> Point:
    """
    Provides the iterative updated state given some iterative load parameter.

    :param p: current state (p + dp)
    :param u: resultants from solve
    :param y: iterative load parameter
    :return:
    """
    ddqf, ddqp, ddff, ddfp = 0.0, 0.0, 0.0, 0.0

    if nlf.nf:
        ddqf = u[:, 0] + y * u[:, 1]
        ddff = y * nlf.ffc
    if nlf.np:
        ddqp = y * nlf.qpc
        ddfp = nlf.rp(p) + y * nlf.kpp(p) @ nlf.qpc
        ddfp += nlf.kpf(p) @ ddqf if nlf.nf else 0.0

    return nlf.point(ddqf, ddqp, ddff, ddfp)


class Problem(ABC):
    """
    Interface of a nonlinear function to the nonlinear solver.

    The external / internal / residual load, motion and stiffness matrix are partitioned based on the free and prescribed degrees of freedom.
    Both the free and prescribed degrees of freedom can be of dimension 0, 1 or higher.
    If dim(free) = 0, then dim(prescribed) > 0 and vice versa.
    That is, either external_load OR prescribed_motion OR BOTH are to be provided.
    """

    def __init__(self, nlf, ixf=None, ixp=None, ff=None, qp=None):
        self.nlf = nlf

        self.ixf = ixf
        self.ixp = ixp

        self.nf = len(self.ixf) if ixf is not None else 0
        self.np = len(self.ixp) if ixp is not None else 0
        self.n = self.nf + self.np

        self.ffc = ff.astype(float) if ff is not None else np.zeros(self.nf)
        self.qpc = qp.astype(float) if qp is not None else np.zeros(self.np)

        # squared norm of load external load and prescribed motion
        self.ff2 = np.dot(self.ffc, self.ffc) if self.nf else None
        self.qp2 = np.dot(self.qpc, self.qpc) if self.np else None

    def load(self, p: Point) -> State:
        load = 1.0 * self.ffc
        load -= self.kfp(p) @ self.qpc if self.np else 0.0  # adds to rhs if nonzero prescribed dof
        return load

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

    def k(self, p: Point) -> State:
        return self.nlf.jacobian(p.q)

    def kff(self, p):
        return self.k(p)[self.ixf, :][:, self.ixf]

    def kpp(self, p):
        return self.k(p)[self.ixp, :][:, self.ixp]

    def kfp(self, p):
        return self.k(p)[self.ixf, :][:, self.ixp]

    def kpf(self, p):
        return self.k(p)[self.ixp, :][:, self.ixf]

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

    @staticmethod
    def make_float(x):
        return x.astype(float) if type(x) is np.ndarray else x
