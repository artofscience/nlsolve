from math import pi, sin

import numpy as np

from core import Structure


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

    def internal_load_free(self, p):
        return super().internal_load(p.uf[0], p.uf[1])

    def kff(self, p):
        return super().tangent_stiffness(p.uf[0])

class InvolvedTrussProblemMotionBased(InvolvedTrussProblem):

    def up(self):
        return np.array([4.0])

    def ff(self):
        return np.array([0.0])

    def internal_load_prescribed(self, p):
        return super().internal_load(p.uf, p.up)[1]

    def internal_load_free(self, p):
        return super().internal_load(p.uf, p.up)[0]

    def kff(self, p):
        return np.array([[super().tangent_stiffness(p.uf[0])[0, 0]]])

    def kpp(self, p):
        a = np.array([super().tangent_stiffness(p.uf[0])[1, 1]])
        return a

    def kfp(self, p):
        return np.array([super().tangent_stiffness(p.uf[0])[1, 0]])

    def kpf(self, p):
        return np.array([super().tangent_stiffness(p.uf[0])[0, 1]])