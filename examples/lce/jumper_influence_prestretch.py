import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, GeneralizedArcLengthVarCoeff
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point
from jumper import Jumper, LCE
import matplotlib as mpl
from criteria import termination_default, EigenvalueChangeTermination

# create Jumper
nlf = Jumper(LCE(l0d=9.5))

# set indices for free and presribed dofs
ixf = [2, 4]
ixp = [0, 1, 3, 5, 6, 7, 8]

ndf, ndp = len(ixf), len(ixp)
nd = ndf + ndp

# reference temperature
T0 = 20
temp_increase = 80

# set prestretch
prestretch = 25 # in mm
# max_prestretch = [9, 9.05, 9.2, 14, 15] # in mm
# max_prestretch = [8.5, 8.9, 9, 9.25, 9.5, 10.1, 14, 15] # in mm


# max_prestretch = [0, 1, 2, 3, 4.5, 4.90, 5.20, 6.3, 7, 8, 9.01, 9.10, 11, 12, 13, 14] # in mm
max_prestretch = [0, 7, 14] # in mm


# 6.1 not working -> why?
# 6.2 snapping
# 6.3 snapping
# 6.5 snapping
# 8.9 snapping
# 9.01 snapping
# 9.1 -inf
# 14 -inf

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
controller = Adaptive(value=0.4, max=0.4, incr=1.1, decr=0.1)

# setup stepper
stepper = IncrementalSolver(solver_arc, controller)

# solve prestretch
solutionps = stepper(s0 + ds0).solutions

#endregion

# obtain index for which prestretch < max_prestretch
initial_states_temp = []
ps_vals = [i.q[6] - x3 for i in solutionps]
for j in max_prestretch:
    for i, val in enumerate(ps_vals):
        if val > j:
            break
        idx = i
    initial_states_temp.append(solutionps[idx])

# change arc-length constraint coefficient wrt T
constraint_arc.cqp[-1] = 0.5

qpt = np.zeros(ndp)
qpt[-1] = temp_increase
structure.set_load(qp=qpt)

# solve using arc-length
steppers = []
for i in range(len(max_prestretch)):
    steppers.append(IncrementalSolver(solver_arc, controller, terminated=termination_default() | EigenvalueChangeTermination(margin=1e-4), reset=False))
    steppers[i].p0 = initial_states_temp[i]
    steppers[i]()
    while not steppers[i].terminated.left.exceed:
        steppers[i]()

fig, ax = plt.subplots()
ax.set_xlim([0, prestretch])
ax.set_ylim([20, 55])
plt.plot(ps_vals, [i.q[2] for i in solutionps], 'ko-', label=f'LCE X{1}')
plt.plot(ps_vals, [i.q[4] for i in solutionps], 'ro-', label=f'LCE X{2}')
plt.xlabel('Pre-stretch')
plt.ylabel('Position')
plt.legend()

for j in range(len(max_prestretch)):
    plt.axvline(x=initial_states_temp[j].q[6] - x3, color='b', linestyle='-')
    plt.axhline(y=initial_states_temp[j].q[2], color='b', linestyle='-')
    plt.axhline(y=initial_states_temp[j].q[4], color='b', linestyle='-')

fig, ax = plt.subplots(2, sharey=True)

ax[0].set(ylabel='U1')
ax[1].set(xlabel='Temperature', ylabel='U2')

prestretches = np.asarray([i.q[6] - x3 for i in initial_states_temp])

c_norm = mpl.colors.Normalize(vmin=np.min(prestretches), vmax=np.max(prestretches))
c_map = mpl.cm.gist_rainbow
s_map = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)

for j, stp in enumerate(steppers):
    for out in stp.history:
        T = np.asarray([i.q[-1] for i in out.solutions])
        ax[0].plot(T, [i.q[2] - initial_states_temp[j].q[2] for i in out.solutions], color=s_map.to_rgba(prestretches[j]))
        ax[1].plot(T,  [i.q[4] - initial_states_temp[j].q[4] for i in out.solutions], color=s_map.to_rgba(prestretches[j]))

ax[0].set_xlim([T0, T0 + temp_increase])
ax[1].set_xlim([T0, T0 + temp_increase])

fig.colorbar(s_map, ax=ax[0])
fig.colorbar(s_map, ax=ax[1])


for j, stp in enumerate(steppers):
    if stp.out.solutions[-1].q[-1] < 0.0:
        print(j)




plt.show()

#endregion