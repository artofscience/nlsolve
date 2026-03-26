import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, GeneralizedArcLength, GeneralizedArcLengthVarCoeff
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from examples.active_springs.spring import SpringT
from utils import Problem, Point
from sympy import Symbol, exp
from examples.springable_curves.structure_from_springable import LongitudinalSpringFromUnivariateBehavior

class LCE:
    def __init__(self,
                 l00: float = 12.08,
                 l0d: float = 5.41,
                 l0e: float = 0.208,
                 l0T: float = 54.5,
                 k00: float = 2.3,
                 k0d: float = 2.25,
                 k0e: float = 0.31,
                 k0T: float = 47.06,
                 k0a: float = 0.0005,
                 k0b: float = 40):
        T = Symbol('T')
        self.l0 = l00 - self.sigmoid(l0d, l0e, l0T, T)
        self.k = k00 - self.sigmoid(k0d, k0e, k0T, T) + self.power(k0a, 2, k0b, T)

    def sigmoid(self, a, b, x0, x):
        return a / (1 + exp(-b * (x - x0)))

    def power(self, a, b, x0, x):
        return a * (x - x0)**b

class Jumper:
    def __init__(self, lce: LCE = LCE()):

        self.n = 9
        self.lce = SpringT(lce.l0, lce.k)
        self.soft = LongitudinalSpringFromUnivariateBehavior("csv_files/soft_2.csv")
        self.snap = LongitudinalSpringFromUnivariateBehavior("csv_files/snap_2.csv")
        self.ix_lce = [2, 3, 4, 5, 8]
        self.ix_soft = [0, 1, 2, 3]
        self.ix_snap = [4, 5, 6, 7]

    def force(self, q):
        f = np.zeros(self.n, dtype=float)
        f[self.ix_lce] += self.lce.force(q[self.ix_lce])
        f[self.ix_soft] += self.soft.force(q[self.ix_soft])
        f[self.ix_snap] += self.snap.force(q[self.ix_snap])
        return f

    def jacobian(self, q):
        K = np.zeros((self.n, self.n), dtype=float)
        K[np.ix_(self.ix_lce, self.ix_lce)] += self.lce.jacobian(q[self.ix_lce])
        K[np.ix_(self.ix_soft, self.ix_soft)] += self.soft.jacobian(q[self.ix_soft])
        K[np.ix_(self.ix_snap, self.ix_snap)] += self.snap.jacobian(q[self.ix_snap])
        return K

# [00, 01, 02, 03, 04, 05, 06, 07, 8]

# [x0, y0, x1, y1, x2, y2, x3, y3, T]

if __name__ == "__main__":
    # create Jumper
    nlf = Jumper()

    # set indices for free and presribed dofs
    ixf = [2, 4]
    ixp = [0, 1, 3, 5, 6, 7, 8]

    ndf, ndp = len(ixf), len(ixp)
    nd = ndf + ndp

    # reference temperature
    T0 = 20
    temp_increase = 100

    # set prestretch
    prestretch = 25 # in mm
    max_prestretch = 10.5 # in mm

    #region PRE-STRETCH
    qp = np.zeros(ndp)
    qp[4] = prestretch

    ff = np.zeros(ndf)

    # setup problem
    structure = Problem(nlf, ixf, ixp, ff, qp)

    # initial point
    x1 = nlf.soft.get_rest_length()
    x2 = x1 + nlf.lce.l0(T0)
    x3 = x2 + nlf.snap.get_rest_length()

    # initial point
    q0 = np.array([0, 0, x1, 0, x2, 0, x3, 0, T0])
    f0 = np.zeros(nd)

    s0 = Point(q0, f0)

    # setup NR solver
    constraint_nr = NewtonRaphson()
    solver_NR = IterativeSolver(structure, constraint_nr)

    # solve for equilibrium around initial point
    ds0 = solver_NR([s0])[0]

    # arc-length coefficients
    cf = 1 * np.ones(nd)
    cq = 5 * np.ones(nd)

    # setup arc-length solver
    constraint_arc = GeneralizedArcLengthVarCoeff(cqf=cq[ixf], cqp=cq[ixp], cff=cf[ixf], cfp=cf[ixp])
    solver_arc = IterativeSolver(structure, constraint_arc)

    # setup controller
    controller = Adaptive(value=0.5, max=1, incr=1.1, decr=0.1)

    # setup stepper
    stepper = IncrementalSolver(solver_arc, controller)

    # solve prestretch
    solutionps = stepper(s0 + ds0).solutions

    #endregion

    # obtain index for which prestretch < max_prestretch
    ps_vals = [i.q[6] for i in solutionps] - x3
    for i, val in enumerate(ps_vals):
        if val > max_prestretch:
            break
        idx = i
    initial_state_temp = solutionps[idx]

    #region TEMP

    # initial_state_temp is already at equilibrium

    # first set the loading conditions for temperature-based loading
    qpt = np.zeros(ndp)
    qpt[-1] = temp_increase
    structure.set_load(qp=qpt)

    # change arc-length constraint coefficient wrt T
    constraint_arc.cqp[-1] = 0.5

    # solve using arc-length
    solutiont = stepper(initial_state_temp).solutions

    # solve using NR
    stepper.solution_method = solver_NR
    solutiont_NR = stepper(initial_state_temp).solutions

    # solve using NR in opposite direction
    qpt[-1] = -temp_increase
    structure.set_load(qp=qpt)
    # stepper.solution_method = solver_arc
    solutiont_NR2 = stepper(solutiont_NR[-1]).solutions

    #endregion

    #region POSTPROC
    solstretch = [i.q[6] - x3 for i in solutionps]

    plt.figure(1)

    ax = plt.subplot(1, 3, 1)
    ax.set_xlim([0, prestretch])
    ax.set_ylim([20, 55])
    plt.plot(solstretch, [i.q[2] for i in solutionps], 'ko-', label=f'LCE X{1}')
    plt.plot(solstretch, [i.q[4] for i in solutionps], 'ro-', label=f'LCE X{2}')
    plt.xlabel('Pre-stretch')
    plt.ylabel('Position')
    plt.legend()

    plt.axvline(x=initial_state_temp.q[6] - x3, color='b', linestyle='-')
    plt.axhline(y=initial_state_temp.q[2], color='b', linestyle='-')
    plt.axhline(y=initial_state_temp.q[4], color='b', linestyle='-')

    ax = plt.subplot(1, 3, 2)
    ax.set_xlim([0, prestretch])
    plt.plot(solstretch, [i.q[4] - i.q[2] for i in solutionps], 'ko-')
    plt.xlabel('Pre-stretch')
    plt.ylabel('LCE Length')

    ax = plt.subplot(1, 3, 3)
    ax.set_xlim([0, prestretch])
    plt.plot(solstretch, [i.f[0] for i in solutionps], 'ko-')
    plt.xlabel('Pre-stretch')
    plt.ylabel('Force')

    plt.figure(2)
    ax = plt.subplot(1, 3, 1)
    ax.set_xlim([T0, temp_increase])
    ax.set_ylim([20, 55])

    # NR
    TNR = np.asarray([i.q[-1] for i in solutiont_NR])
    TNR2 = np.asarray([i.q[-1] for i in solutiont_NR2])

    plt.plot(TNR, [i.q[2] for i in solutiont_NR], 'k.--', label=f'LCE X{1} NR')
    plt.plot(TNR, [i.q[4] for i in solutiont_NR], 'r.--', label=f'LCE X{2} NR')

    plt.plot(TNR2, [i.q[2] for i in solutiont_NR2], 'k*--', label=f'LCE X{1} BACK')
    plt.plot(TNR2, [i.q[4] for i in solutiont_NR2], 'r*--', label=f'LCE X{2} BACK')

    # arclength
    T = np.asarray([i.q[-1] for i in solutiont])
    plt.plot(T, [i.q[2] for i in solutiont], 'ko-', label=f'LCE X{1} ARC')
    plt.plot(T, [i.q[4] for i in solutiont], 'ro-', label=f'LCE X{2} ARC')

    plt.xlabel('Temperature')
    plt.ylabel('Position')

    plt.axvline(x=T0, color='b', linestyle='-')
    plt.axhline(y=initial_state_temp.q[2], color='b', linestyle='-')
    plt.axhline(y=initial_state_temp.q[4], color='b', linestyle='-')

    ax = plt.subplot(1, 3, 2)
    ax.set_xlim([T0, temp_increase])
    plt.plot(TNR, [i.q[4] - i.q[2] for i in solutiont_NR], 'k.--', label='NR')
    plt.plot(TNR2, [i.q[4] - i.q[2] for i in solutiont_NR2], 'k*--', label='BACK')
    plt.plot(T, [i.q[4] - i.q[2] for i in solutiont], 'ko-', label='ARC')
    plt.xlabel('Temperature')
    plt.ylabel('LCE Length')
    plt.legend()

    ax = plt.subplot(1, 3, 3)
    ax.set_xlim([T0, temp_increase])
    plt.plot(TNR, [i.f[0] for i in solutiont_NR], 'k.-', label='NR')
    plt.plot(TNR2, [i.f[0] for i in solutiont_NR2], 'k*-', label='BACK')
    plt.plot(T, [i.f[0] for i in solutiont], 'ko-', label='ARC')
    plt.xlabel('Temperature')
    plt.ylabel('Force')
    plt.legend()

    plt.show()
    #endregion

