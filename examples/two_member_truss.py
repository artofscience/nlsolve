from solver import Structure
import numpy as np
from math import cos, sin, atan

"""
Two-member truss problem from Pretti et al. (2022).

Lengths of two trusses A and B as function of initial height l0
and applied displacements uv and uh

lA = sqrt((l0 - uv)**2 + (l0/2 + uh)**2)
lB = sqrt((l0 - uv)**2 + (l0/2 - uh)**2)

Initial length
lA0 = lB0 = sqrt((l0/2)**2 + l0**2) = sqrt(3/2) l0

tan(thetaA) = l0 - uv / l0/2 + uh
tan(thetaB) = l0 - uv / l0/2 - uh

Deformation gradient
FA = lA / l0
FB = lB / l0

Let's assume incompressibility, then
J = 1 and v = V, so

areaA = lA0 * areaA0 / lA
areaB = lB0 * areaB0 / lB

Hencky model:

logarithmic strain
eA = ln(lA / lA0)
eB = ln(lB / lB0)

kirchhoff stress
sigmaA = E * eA
sigmaB = E * eB

forceA = sigmaA * areaA
forceB = sigmaB * areaB

Saint Venant-Kirchhoff model:

Green-Lagrange strain
eA = 0.5 * (FA**2 -1)
eB = 0.5 * (FB**2 -1)

kirchhoff stress
sigmaA = E * eA
sigmaB = E * eB

forceA = sigmaA * areaA
forceB = sigmaB * areaB

freacth = -forceA * cos(thetaA) + forceB * cos(thetaB)
freactv = forcaA * sin(thetaA) + forceB * sin(thetaB)
"""


class TwoMemberTrussMotionBased(Structure):
    L0 = 1.0 # initial height
    E = 1.0
    A0 = 1.0

    def get_length(self, uh, uv):
        lv = self.L0 - uv
        lhA = self.L0/2 + uh
        lhB = self.L0/2 - uh

        lA = np.sqrt(lv**2 + lhA**2)
        lB = np.sqrt(lv**2 + lhB**2)

        return lA, lB

    def prescribed_motion(self):
        return np.array([1.0/20, 1.0])

    def internal_load_prescribed(self, p):
        uh = p.v[0]
        uv = p.v[1]

        lA0, lB0 = self.get_length(0, 0)
        lA, lB = self.get_length(uh, uv)

        def_grad_A = lA / lA0
        def_grad_B = lB / lB0

        strain_A = 0.5 * (def_grad_A**2 - 1)
        strain_B = 0.5 * (def_grad_B**2 - 1)

        stress_A = self.E * strain_A
        stress_B = self.E * strain_B

        area_A = lA0 * self.A0 / lA
        area_B = lB0 * self.A0 / lB

        force_A = stress_A * area_A
        force_B = stress_B * area_B

        theta_A = atan((self.L0 - uv) / (self.L0/2 + uh))
        theta_B = atan((self.L0 - uv) / (self.L0/2 - uh))

        fh = -force_A * cos(theta_A) + force_B * cos(theta_B)
        fv = force_A * sin(theta_A) + force_B * sin(theta_B)

        return np.array([fh, fv])









