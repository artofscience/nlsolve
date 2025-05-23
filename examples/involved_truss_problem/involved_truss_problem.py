from math import pi, sin

import numpy as np

from utils import Structure


class InvolvedTrussProblem(Structure):
    w = 0.25
    theta0 = pi / 2.5

    def internal_load(self, a1, a2):
        f1 = (1 / np.sqrt(1 - 2 * a1 * sin(self.theta0) + a1 ** 2) - 1) * (
                sin(self.theta0) - a1) - self.w * (a2 - a1)
        return np.array([f1, self.w * (a2 - a1)])

    def tangent_stiffness(self, a1):
        df1da1 = self.w - 1 / (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (1 / 2) + (
                (a1 - sin(self.theta0)) * (2 * a1 - 2 * sin(self.theta0))) / (
                         2 * (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (3 / 2)) + 1
        return np.array([[df1da1, -self.w], [-self.w, self.w]], dtype=float)

class InvolvedTrussProblemLoadBased(InvolvedTrussProblem):

    def ff(self):
        return np.array([0, 1.0], dtype=float)

    def gf(self, p):
        return super().internal_load(p.qf[0], p.qf[1])

    def kff(self, p):
        return super().tangent_stiffness(p.qf[0])

class InvolvedTrussProblemMotionBased(InvolvedTrussProblem):

    def up(self):
        return np.array([4.0])

    def ff(self):
        return np.array([0.0])

    def gp(self, p):
        return super().internal_load(p.qf, p.qp)[1]

    def gf(self, p):
        return super().internal_load(p.qf, p.qp)[0]

    def kff(self, p):
        return np.array([[super().tangent_stiffness(p.qf[0])[0, 0]]])

    def kpp(self, p):
        a = np.array([super().tangent_stiffness(p.qf[0])[1, 1]])
        return a

    def kfp(self, p):
        return np.array([super().tangent_stiffness(p.qf[0])[1, 0]])

    def kpf(self, p):
        return np.array([super().tangent_stiffness(p.qf[0])[0, 1]])