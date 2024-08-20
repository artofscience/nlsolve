import numpy as np

from constraint import Constraint
from point import Point
from structure import Structure


class NewtonRaphson(Constraint):

    def predictor(self, p, dp, ddx, dl, sol):

        load = 0.0
        load += self.v2 if self.np else 0.0
        load += self.f2 if self.nf else 0.0

        ddy = dl / np.sqrt(load)

        point = Point(y=ddy)

        if self.nf:
            point.u += ddy * ddx[:, 1]
            point.f += ddy * self.f
        if self.np:
            point.v = ddy * self.v
            point.p = -self.Kpp(p) @ point.v
            point.p -= ddy * self.Kpf(p) @ ddx[:, 1] if self.nf else 0.0

        return point

    def corrector(self, p, dp, ddx, dl):

        point = Point()

        if self.nf:
            point.u += ddx[:, 0]
        if self.np:
            point.p = -self.rp(p + dp)
            point.p -= self.Kpf(p + dp) @ ddx[:, 0] if self.nf else 0.0

        return point


class ArcLength(Constraint):
    def __init__(self, nonlinear_function, beta=1.0):
        super().__init__(nonlinear_function)
        self.beta = beta

    def predictor(self, p, dp, ddx, dl, sol):
        y = self.get_roots_predictor(p, ddx[:, 1], dl)
        cps = self.get_point(p, dp, ddx, y)
        return self.select_root_predictor(p, sol, cps)

    def corrector(self, p, dp, ddx, dl):
        y = self.get_roots_corrector(p, dp, ddx, dl)
        cps = self.get_point(p, dp, ddx, y)
        return self.select_root_corrector(dp, cps)

    def get_roots_predictor(self, p, u, dl):
        a = 0.0
        if self.nf:
            a += np.dot(u, u) + self.beta**2 * self.f2
        if self.np:
            tmpa = self.Kpp(p) @ self.v
            if self.nf:
                tmpa += self.Kpf(p) @ u
            a += self.beta**2 * np.dot(tmpa, tmpa) + self.v2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, p, dp, u, dl):
        a = np.zeros(3)

        a[2] -= dl**2
        if self.nf:
            a[0] += np.dot(u[:, 1], u[:, 1])
            a[0] += self.beta**2 * self.f2
            a[1] += 2 * np.dot(u[:, 1], dp.u + u[:, 0])
            a[1] += 2 * self.beta**2 * np.dot(dp.f, self.f)
            a[2] += np.dot(dp.u + u[:, 0], dp.u + u[:, 0])
            a[2] += self.beta**2 * np.dot(dp.f, dp.f)
        if self.np:
            a[0] += self.v2
            a[1] += 2 * np.dot(self.v, dp.v)
            a[2] += np.dot(dp.v, dp.v)
            tmpa = self.Kpp(p + dp) @ self.v
            tmpc = dp.p - self.a.residual_prescribed(p + dp)
            if self.nf:
                tmpa += self.Kpf(p + dp) @ u[:, 1]
                tmpc -= self.Kpf(p + dp) @ u[:, 0]
            a[0] += self.beta**2 * np.dot(tmpa, tmpa)
            a[1] -= 2 * self.beta**2 * np.dot(tmpa, tmpc)
            a[2] += self.beta**2 * np.dot(tmpc, tmpc)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])

    def get_point(self, p, dp, u, y):
        if self.np is not None and self.nf is None:
            ddp = [-self.rp(p + dp) - y[i] * self.Kpp(p + dp) @ self.v for i in range(2)]
            return [Point(v=y[i] * self.v, p=ddp[i], y=y[i]) for i in range(2)]
        if self.nf is not None:
            x = [u[:, 0] + i * u[:, 1] for i in y]
            if self.np is None:
                return [Point(u=x[i], f=y[i] * self.f, y=y[i]) for i in range(2)]
            else:
                ddp = [-self.rp(p + dp) - self.Kpf(p + dp) @ u[:, 0] - y[i] * (
                            self.Kpf(p + dp) @ u[:, 1] + self.Kpp(p + dp) @ self.v) for i in range(2)]
                return [Point(u=x[i], v=y[i] * self.v, f=y[i] * self.f, p=ddp[i], y=y[i]) for i in range(2)]

    def select_root_corrector(self, dp, cps):
        """
        This rule is based on the projections of the generalized correction vectors on the previous correction [Vasios, 2015].
        The corrector that forms the closest correction to the previous point is chosen.
        Note: this rule cannot be used in the first iteration since the initial corrections are equal to zero at the beginning of each increment.
        """
        if self.nf:
            cpd = lambda i: np.dot(dp.u, dp.u + cps[i].u)
        if self.np:
            cpd = lambda i: np.dot(dp.v, dp.v + cps[i].v)
            if self.nf:
                cpd = lambda i: np.dot(dp.u, dp.u + cps[i].u) + np.dot(dp.v, dp.v + cps[i].v)

        return cps[0] if cpd(0) >= cpd(1) else cps[1]

    def select_root_predictor(self, p, sol, cps):
        if p.y == 0:
            return cps[0] if cps[0].y > cps[1].y else cps[1]

        else:
            if self.nf:
                vec1 = np.append(sol[-2].u - p.u - cps[0].u, sol[-2].f - p.f - cps[0].f)
                vec2 = np.append(sol[-2].u - p.u - cps[1].u, sol[-2].f - p.f - cps[1].f)

            if self.np:
                vec1 = np.append(sol[-2].v - p.v - cps[0].v, sol[-2].p - p.p - cps[0].p)
                vec2 = np.append(sol[-2].v - p.v - cps[1].v, sol[-2].p - p.p - cps[1].p)

                if self.nf:
                    vec11 = np.append(sol[-2].u - p.u - cps[0].u, sol[-2].f - p.f - cps[0].f)
                    vec12 = np.append(sol[-2].v - p.v - cps[0].v, sol[-2].p - p.p - cps[0].p)
                    vec1 = np.append(vec11, vec12)
                    vec21 = np.append(sol[-2].u - p.u - cps[1].u, sol[-2].f - p.f - cps[1].f)
                    vec22 = np.append(sol[-2].v - p.v - cps[1].v, sol[-2].p - p.p - cps[1].p)
                    vec2 = np.append(vec21, vec22)

            return cps[0] if np.linalg.norm(vec1) > np.linalg.norm(vec2) else cps[1]


class NewtonRaphsonByArcLength(ArcLength):

    def predictor(self, p, dp, ddx, dl, sol):
        y = self.get_roots_predictor(p, ddx[:, 1], dl)
        cps = self.get_point(p, dp, ddx, y)
        return cps[0]

    def corrector(self, p, dp, ddx, dl):
        y = self.get_roots_corrector(p, dp, ddx, dl)
        cps = self.get_point(p, dp, ddx, y)
        return cps[0]

    def get_roots_predictor(self, p, u, dl):
        a = 0.0
        if self.nf:
            a += self.beta**2 * self.f2
        if self.np:
            a += self.v2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, p, dp, u, dl):
        a = np.zeros(3)

        a[2] -= dl**2
        if self.nf:
            a[0] += self.beta**2 * self.f2
            a[1] += 2 * self.beta**2 * np.dot(dp.f, self.f)
            a[2] += self.beta**2 * np.dot(dp.f, dp.f)
        if self.np:
            a[0] += self.v2
            a[1] += 2 * np.dot(self.v, dp.v)
            a[2] += np.dot(dp.v, dp.v)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])


class GeneralizedArcLength(ArcLength):
    def __init__(self, nonlinear_function: Structure, alpha=1.0, beta=1.0):
        super().__init__(nonlinear_function, beta)
        self.alpha = alpha

    def predictor(self, p, dp, ddx, dl, sol):
        y = self.get_roots_predictor(p, ddx[:, 1], dl)
        cps = self.get_point(p, dp, ddx, y)
        return self.select_root_predictor(p, sol, cps) if self.alpha > 0.0 else cps[0]

    def corrector(self, p, dp, ddx, dl):
        y = self.get_roots_corrector(p, dp, ddx, dl)
        cps = self.get_point(p, dp, ddx, y)
        return self.select_root_corrector(dp, cps) if self.alpha > 0.0 else cps[0]

    def get_roots_predictor(self, p, u, dl):
        a = 0.0
        if self.nf:
            a += self.alpha * np.dot(u, u) + self.beta**2 * self.f2
        if self.np:
            tmpa = self.Kpp(p) @ self.v
            if self.nf:
                tmpa += self.Kpf(p) @ u
            a += self.alpha * self.beta**2 * np.dot(tmpa, tmpa) + self.v2

        return np.array([1, -1]) * dl / np.sqrt(a)

    def get_roots_corrector(self, p, dp, u, dl):
        a = np.zeros(3)

        a[2] -= dl**2
        if self.nf:
            a[0] += self.alpha * np.dot(u[:, 1], u[:, 1])
            a[0] += self.beta**2 * self.f2
            a[1] += self.alpha * 2 * np.dot(u[:, 1], dp.u + u[:, 0])
            a[1] += 2 * self.beta**2 * np.dot(dp.f, self.f)
            a[2] += self.alpha * np.dot(dp.u + u[:, 0], dp.u + u[:, 0])
            a[2] += self.beta**2 * np.dot(dp.f, dp.f)
        if self.np:
            a[0] += self.v2
            a[1] += 2 * np.dot(self.v, dp.v)
            a[2] += np.dot(dp.v, dp.v)
            tmpa = self.Kpp(p + dp) @ self.v
            tmpc = dp.p - self.a.residual_prescribed(p + dp)
            if self.nf:
                tmpa += self.Kpf(p + dp) @ u[:, 1]
                tmpc -= self.Kpf(p + dp) @ u[:, 0]
            a[0] += self.alpha * self.beta**2 * np.dot(tmpa, tmpa)
            a[1] -= self.alpha * 2 * self.beta**2 * np.dot(tmpa, tmpc)
            a[2] += self.alpha * self.beta**2 * np.dot(tmpc, tmpc)

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        return (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])
