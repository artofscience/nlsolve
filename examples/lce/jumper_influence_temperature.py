import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, GeneralizedArcLength, GeneralizedArcLengthVarCoeff
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from examples.active_springs.spring import SpringT
from utils import Problem, Point
from sympy import Symbol, exp
from examples.springable_curves.structure_from_springable import LongitudinalSpringFromUnivariateBehavior
from jumper import Jumper, LCE
import matplotlib as mpl

# create Jumper
nlf = Jumper(LCE(l0d=9.5))

# set indices for free and presribed dofs
ixf = [2, 4]
ixp = [0, 1, 3, 5, 6, 7, 8]

ndf, ndp = len(ixf), len(ixp)
nd = ndf + ndp

# reference temperature
T0 = 20

# set prestretch
prestretch = 25 # in mm

temp = np.linspace(20, 100, 9) # in mm
max_temp = 80

#region PRE-STRETCH
qp = np.zeros(ndp)
qp[-1] = max_temp

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
# change arc-length constraint coefficient wrt T
constraint_arc.cqp[-1] = 0.5

solver_arc = IterativeSolver(structure, constraint_arc)

# setup controller
controller = Adaptive(value=0.4, max=0.4, incr=1.1, decr=0.1)

# setup stepper
stepper = IncrementalSolver(solver_arc, controller)

# solve prestretch
solution_temp = stepper(s0 + ds0).solutions

fig, ax = plt.subplots()
ax.set_xlim([T0, T0 + max_temp])
T = np.asarray([i.q[-1] for i in solution_temp])
plt.plot(T, [i.q[2] for i in solution_temp], 'ko-', label=f'LCE X{1}')
plt.plot(T, [i.q[4] for i in solution_temp], 'ro-', label=f'LCE X{2}')
plt.xlabel('Temperature')
plt.ylabel('Position')
plt.legend()

#endregion

# obtain index for which T < maxT
initial_states_temp = []
temps = [i.q[-1] for i in solution_temp]
idx = 0
for j in temp:
    for i, val in enumerate(temps):
        if val > j:
            break
        idx = i
    initial_states_temp.append(solution_temp[idx])

for j in range(len(temp)):
    plt.axvline(x=initial_states_temp[j].q[-1], color='b', linestyle='-')
    plt.axhline(y=initial_states_temp[j].q[2], color='b', linestyle='-')
    plt.axhline(y=initial_states_temp[j].q[4], color='b', linestyle='-')

# change arc-length constraint coefficient wrt T
constraint_arc.cqp[-1] = 0.5

qps = np.zeros(ndp)
qps[4] = prestretch
structure.set_load(qp=qps)

# solve using arc-length
solutions = []
for i in range(len(temp)):
    solutions.append(stepper(initial_states_temp[i]).solutions)

c_norm = mpl.colors.Normalize(vmin=np.min(temp), vmax=np.max(temp))
c_map = mpl.cm.gist_rainbow
s_map = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)

fig, ax = plt.subplots(2, sharey=True)

ax[0].set(ylabel='U1')
ax[1].set(xlabel='Stretch', ylabel='U2')

for j in range(len(solutions)):
    stretch = [i.q[6] - x3 for i in solutions[j]]
    ax[0].plot(stretch, [i.q[2] - initial_states_temp[j].q[2] for i in solutions[j]] ,color=s_map.to_rgba(temp[j]))
    ax[1].plot(stretch,  [i.q[4] - initial_states_temp[j].q[4] for i in solutions[j]] , color=s_map.to_rgba(temp[j]))

fig.colorbar(s_map, ax=ax[0])
fig.colorbar(s_map, ax=ax[1])

plt.show()
