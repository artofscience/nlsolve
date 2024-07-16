import numpy as np
from copy import deepcopy


class IncrementalSolver:
    def __init__(self, solution_method):
        self.solution_method = solution_method

    def __call__(self):
        equilibrium_solutions = []

        # Add initial point to equilibrium solutions (u = 0, lam = 0)
        p = Point(np.zeros_like(self.solution_method.nonlinear_function.external_load()), 0.0)
        equilibrium_solutions.append(p)

        # Continue while load parameter is smaller than X
        while p.y <= 1.0:

            # Compute motion and load increment at a given point with DL = 0.05 using solution method of choice
            dp = self.solution_method(equilibrium_solutions, 0.1)

            # Compute new point and add to the equilibrium solutions
            p = Point(p.x + dp.x, p.y + dp.y)
            equilibrium_solutions.append(p)

        return equilibrium_solutions


class IterativeSolver:
    def __init__(self, nonlinear_function, al=True):
        self.nonlinear_function = nonlinear_function
        self.f = self.nonlinear_function.external_load()
        self.beta = 0.0
        self.constraint = ArcLength(self.f) if al else NewtonRaphson(self.f)

    def __call__(self, sol, alpha=1.0):

        p = sol[-1]
        dp = Point(np.zeros_like(p.x), 0.0)
        ddx = np.zeros((np.shape(p.x)[0], 2), dtype=float)

        k = self.nonlinear_function.tangent_stiffness_free_free(p.x)
        ddx[:, 1] = np.linalg.solve(k, -self.f)
        dp += self.constraint.predictor(p, sol, dp, ddx, alpha)
        r = self.nonlinear_function.residual_free(p.x + dp.x, p.y + dp.y)

        while any(abs(r) > 1e-6):

            k = self.nonlinear_function.tangent_stiffness_free_free(p.x + dp.x)
            ddx[:, :] = np.linalg.solve(k, -np.array([r, self.f]))
            dp += self.constraint.corrector(dp, ddx, alpha)
            r = self.nonlinear_function.residual_free(p.x + dp.x, p.y + dp.y)

        return dp


class NewtonRaphson:
    def __init__(self, f):
        self.f = f

    def predictor(self, p, sol, dp, ddx, alpha):
        return Point(alpha * ddx[:, 1], alpha)

    def corrector(self, dp, ddx, alpha):
        return Point(ddx[:, 0], 0.0)


class ArcLength(NewtonRaphson):
    beta = 1.0

    def predictor(self, p, sol, dp, ddx, alpha):
        cps = self.get_roots(dp, ddx, alpha)
        return self.select_root_predictor(p, sol, cps)

    def corrector(self, dp, ddx, alpha):
        cps = self.get_roots(dp, ddx, alpha)
        return self.select_root_corrector(dp, cps)

    def get_roots(self, p, u, dl):
        a = np.zeros(3)

        tmp = p.x + u[:, 0]
        a[0] = np.dot(u[:, 1], u[:, 1]) + self.beta**2 * np.dot(self.f, self.f)
        a[1] = 2 * np.dot(tmp, u[:, 1]) + 2 * self.beta**2 * p.y * np.dot(self.f, self.f)
        a[2] = np.dot(tmp, tmp) + self.beta**2 * p.y**2 * np.dot(self.f, self.f) - dl**2

        if (d := a[1] ** 2 - 4 * a[0] * a[2]) <= 0:
            raise ValueError("Discriminant of quadratic constraint equation is not positive!")

        y = (-a[1] + np.array([1, -1]) * np.sqrt(d)) / (2 * a[0])

        x = [u[:, 0] + i * u[:, 1] for i in y]

        # return roots (as list of points)
        return [Point(x[i], y[i]) for i in range(2)]

    def select_root_corrector(self, dp, cps):
        """
        This rule is based on the projections of the generalized correction vectors on the previous correction [Vasios, 2015].
        The corrector that forms the closest correction to the previous point is chosen.
        Note: this rule cannot be used in the first iteration since the initial corrections are equal to zero at the beginning of each increment.
        """

        cpd = lambda i: np.dot(dp.x, dp.x + cps[i].x) + self.beta**2 * dp.y * (dp.y + cps[i].y) * np.dot(self.f, self.f)
        return cps[0] if cpd(0) >= cpd(1) else cps[1]

    @staticmethod
    def select_root_predictor(p, sol, cps):
        if p.y == 0:
            return cps[0] if cps[0].y > cps[1].y else cps[1]

        else:
            vec1 = np.append(sol[-2].x - p.x - cps[0].x, sol[-2].y - p.y - cps[0].y)
            vec2 = np.append(sol[-2].x - p.x - cps[1].x, sol[-2].y - p.y - cps[1].y)

            return cps[0] if np.linalg.norm(vec1) > np.linalg.norm(vec2) else cps[1]


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self):
        return np.append(self.x, self.y)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __add__(self, other):
        out = deepcopy(Point(self.x, self.y))
        out += other
        return out

    def __sub__(self, other):
        out = deepcopy(Point(self.x, self.y))
        out -= other
        return out