import numpy as np
from copy import deepcopy
from abc import ABC


class IncrementalSolver:
    def __init__(self, solution_method, alpha=0.1):
        self.alpha = alpha
        self.solution_method = solution_method
        self.nf = np.shape(self.solution_method.nonlinear_function.external_load())[0] if self.solution_method.nonlinear_function.external_load() is not None else None
        self.np = np.shape(self.solution_method.nonlinear_function.prescribed_motion())[0] if self.solution_method.nonlinear_function.prescribed_motion() is not None else None

    def __call__(self):
        print("let's get startin")
        equilibrium_solutions = []

        if not self.np:
            p = Point(u=np.zeros(self.nf), f=np.zeros(self.nf))
        elif not self.nf:
            p = Point(v=np.zeros(self.np), p=np.zeros(self.np))
        else:
            p = Point(np.zeros(self.nf), np.zeros(self.np), np.zeros(self.nf), np.zeros(self.np))

        equilibrium_solutions.append(p)

        while p.y <= 1.0:

            dp = self.solution_method(equilibrium_solutions, self.alpha)

            p = p + dp
            equilibrium_solutions.append(p)
            # print(p.u)
            # print(p.v)
            # print(p.f)
            # print(p.p)
            # print(p.y)

        return equilibrium_solutions


class IterativeSolver:
    def __init__(self, nonlinear_function, al=True):
        self.nonlinear_function = nonlinear_function
        self.f = self.nonlinear_function.external_load()
        self.v = self.nonlinear_function.prescribed_motion()

        self.nf = np.shape(self.f)[0] if self.f is not None else None
        self.np = np.shape(self.v)[0] if self.v is not None else None

        self.constraint = self.ArcLength(self.nonlinear_function, self.nf, self.np) if al else self.NewtonRaphson(self.nonlinear_function, self.nf, self.np)

    def __call__(self, sol, alpha=1.0):

        p = sol[-1]

        dp = 0.0 * p
        ddx = np.zeros((self.nf, 2), dtype=float) if self.nf else None

        if self.nf:
            k = self.nonlinear_function.tangent_stiffness_free_free(p)
            ddx[:, 1] = np.linalg.solve(k, -self.f) if any(self.f) != 0.0 else 0.0

        dp += self.constraint.predictor(p, dp, ddx, alpha, sol)

        r = np.array([])
        if self.nf:
            rf = self.nonlinear_function.residual_free(p + dp)
            r = np.append(r, rf)
        if self.np:
            r = np.append(r, self.nonlinear_function.residual_prescribed(p + dp))

        count = 0
        while np.any(np.abs(r) > 1e-6):
            count += 1

            if self.nf:
                k = self.nonlinear_function.tangent_stiffness_free_free(p + dp)
                ddx[:, :] = np.linalg.solve(k, -np.array([rf, self.f]).T)

            dp += self.constraint.corrector(p, dp, ddx, alpha)

            r = np.array([])
            if self.nf:
                rf = self.nonlinear_function.residual_free(p + dp)
                r = np.append(r, rf)
            if self.np:
                r = np.append(r, self.nonlinear_function.residual_prescribed(p + dp))


            # print(r)

        print("Number of corrections: %d" % count)
        return dp

    class NewtonRaphson:
        def __init__(self, nonlinear_function, nf, np):
            self.a = nonlinear_function
            self.nf = nf
            self.np = np

        def predictor(self, p, dp, ddx, alpha, sol):

            point = Point(y=alpha)

            if self.nf:
                point.u += alpha * ddx[:, 1]
                point.f += alpha * self.a.external_load()
            if self.np:
                point.v = alpha * self.a.prescribed_motion()
                point.p = -self.a.tangent_stiffness_prescribed_prescribed(p) @ point.v
                if self.nf:
                    point.p -= alpha * self.a.tangent_stiffness_prescribed_free(p) @ ddx[:, 1]

            return point

        def corrector(self, p, dp, ddx, alpha):

            point = Point()

            if self.nf:
                point.u += ddx[:, 0]
            if self.np:
                point.p = -self.a.residual_prescribed(p + dp)
                if self.nf:
                    point.p -= self.a.tangent_stiffness_prescribed_free(p + dp) @ ddx[:, 0]

            return point

    class ArcLength(NewtonRaphson):
        beta = 1.0

        def predictor(self, p, dp, ddx, alpha, sol):
            cps = self.get_roots(p, dp, ddx, alpha)
            return self.select_root_predictor(p, sol, cps)

        def corrector(self, p, dp, ddx, alpha):
            cps = self.get_roots(p, dp, ddx, alpha)
            return self.select_root_corrector(dp, cps)

        def get_roots(self, p, dp, u, dl):
            a = np.zeros(3)

            a[2] -= dl**2
            if self.nf:
                a[0] += np.dot(u[:, 1], u[:, 1])
                a[0] += self.beta**2 * np.dot(self.a.external_load(), self.a.external_load())
                a[1] += 2 * np.dot(u[:, 1], dp.u + u[:, 0])
                a[1] += 2 * self.beta**2 * np.dot(dp.f, self.a.external_load())
                a[2] += np.dot(dp.u + u[:, 0], dp.u + u[:, 0])
                a[2] += self.beta**2 * np.dot(dp.f, dp.f)
            if self.np:
                a[0] += np.dot(self.a.prescribed_motion(), self.a.prescribed_motion())
                a[1] += 2 * np.dot(self.a.prescribed_motion(), dp.v)
                a[2] += np.dot(dp.v, dp.v)
                tmpa = self.a.tangent_stiffness_prescribed_prescribed(p+dp) @ self.a.prescribed_motion()
                tmpc = dp.p - self.a.residual_prescribed(p + dp)
                if self.nf:
                    tmpa += self.a.tangent_stiffness_prescribed_free(p + dp) @ u[:, 1]
                    tmpc -= self.a.tangent_stiffness_prescribed_free(p + dp) @ u[:, 0]
                a[0] += self.beta**2 * np.dot(tmpa, tmpa)
                a[1] -= 2 * self.beta**2 * np.dot(tmpa, tmpc)
                a[2] += self.beta**2 * np.dot(tmpc, tmpc)

            if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
                raise ValueError("Discriminant of quadratic constraint equation is not positive!")

            y = (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])

            if not self.np:
                x = [u[:, 0] + i * u[:, 1] for i in y]

                return [Point(u=x[i], f=y[i] * self.a.external_load(), y=y[i]) for i in range(2)]
            elif not self.nf:
                ddp = [-self.a.residual_prescribed(p + dp) - y[i] * self.a.tangent_stiffness_prescribed_prescribed(p + dp) @ self.a.prescribed_motion() for i in range(2)]
                return [Point(v=y[i] * self.a.prescribed_motion(), p=ddp[i], y=y[i]) for i in range(2)]
            else:
                x = [u[:, 0] + i * u[:, 1] for i in y]

                ddp = [-self.a.residual_prescribed(p + dp) - self.a.tangent_stiffness_prescribed_free(p + dp) @ u[:, 0] - y[i] * (self.a.tangent_stiffness_prescribed_free(p + dp) @ u[:, 1] + self.a.tangent_stiffness_prescribed_prescribed(p + dp) @ self.a.prescribed_motion()) for i in range(2)]
                return [Point(u=x[i], v=y[i] * self.a.prescribed_motion(), f=y[i] * self.a.external_load(), p=ddp[i], y=y[i]) for i in range(2)]

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

            # + self.beta**2 * dp.y * (dp.y + cps[i].y) * np.dot(self.a.external_load(), self.a.external_load())

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


        def select_root_predictor_feng(self, p, dp, ddx, cps):
            """Secant path procedure proposed by Feng et al.
            The secant path method does not rely on quantities which are related to the tangent matrix,
            being therefore insensitive to the existence of bifurcations.
            In short: the same direction of the given displacement vector and the equilibrium path is not allowed to
            proceed in the direction opposite to the prescribed displacement for the first load step."""

            if p.y == 0:
                return cps[0] if cps[0].y > cps[1].y else cps[1]
            else:
                a = 0
                if self.nf:
                    a += np.dot(dp.u, ddx[:, 1])
                if self.np:
                    a += np.dot(dp.v, self.a.prescribed_motion())

                return cps[0] if np.sign(cps[0]) == np.sign(cps[1]) else cps[1]


class Structure(ABC):
    def external_load(self):
        return None

    def prescribed_motion(self):
        return None

    def internal_load_free(self, p):
        return None

    def internal_load_prescribed(self, p):
        return None

    def residual_free(self, p):
        return self.internal_load_free(p) + p.y * self.external_load()

    def residual_prescribed(self, p):
        return self.internal_load_prescribed(p) + p.p

    def tangent_stiffness_free_free(self, p):
        return None

    def tangent_stiffness_free_prescribed(self, p):
        return None
    def tangent_stiffness_prescribed_free(self, p):
        return None

    def tangent_stiffness_prescribed_prescribed(self, p):
        return None


class Point:
    def __init__(self, u=0, v=0, f=0, p=0, y=0.0):
        self.u = u
        self.v = v
        self.f = f
        self.p = p
        self.y = y

    def __iadd__(self, other):
        self.u += other.u
        self.v += other.v
        self.f += other.f
        self.p += other.p
        self.y += other.y
        return self

    def __rmul__(self, other):
        out = deepcopy(self)
        out.u *= other
        out.v *= other
        out.f *= other
        out.p *= other
        out.y *= other
        return out

    def __add__(self, other):
        out = deepcopy(Point(self.u, self.v, self.f, self.p, self.y))
        out += other
        return out

    def __sub__(self, other):
        out = deepcopy(Point(self.u, self.v, self.f, self.p, self.y))
        out -= other
        return out