import numpy as np

from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point
from constraints import NewtonRaphson, GeneralizedArcLength
from matplotlib import pyplot as plt
from controllers import Adaptive
from criteria import LoadTermination

class SpringClean:
    def __init__(self, k1: float = 1.0, k2: float = 1.0):
        self.k1 = k1
        self.k2 = k2

    def stiffness(self, x1: float = 1.0, x2: float = 1.0):
        return 1/self.compliance(x1, x2)

    def compliance(self, x1: float = 1.0, x2: float = 1.0):
        return 1/(1+x1**3-x1)*self.k1 + 1/(x2*self.k2)

    def lagrangian(self, q: np.ndarray, lam: float = 1.0) -> float:
        x1, x2, mu = q
        return 1/(1+x1**3-x1)*self.k1 + 1/(x2*self.k2) + mu * (x1 + x2 - 2*(1 - lam))

    def force(self, q: np.ndarray) -> np.ndarray:
        """
        q are here the design variables, namely x1, x2 and mu.
        force refers to the internal load, which is here
        g = [dC/dx1 - mu, dC/dx2 - mu, -x1 - x2], so external load f = [0, 0, 1]
        """
        x1, x2, mu = q
        z = 1 + x1**3 - x1
        dfdz = -1/z**2
        dzdx1 = 3*x1**2 - 1
        return np.array([dfdz * dzdx1 / self.k1 + mu,
                         -1 / (self.k2 * x2 ** 2) + mu,
                         x1 + x2 - 2])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        x1, x2, mu = q
        z = 1 + x1**3 - x1
        dfdz = -1/z**2
        dzdx1 = 3*x1**2 - 1
        df2dz2 = 2/z**3
        dz2dx12 = 6*x1
        return np.array([[df2dz2 * dzdx1**2 / self.k1 + dfdz * dz2dx12 / self.k1, 0, 1],
                         [0, 2 / (self.k2 * x2**3), 1],
                        [1, 1, 0]])


class SpringSeries:
    """ Two springs in series.
    Lagrangian L(q) = C(x) - mu * c(x, lambda), q = (x, mu)
    C = 1/x1 + 1/x2 - alpha * sum_i ln(xi) + ln(1 - xi)
    c = x1 + x2 - lambda
    """
    def __init__(self, k1: float = 1.0, k2: float = 1.0, x0: float = 0.1):
        self.k1 = k1
        self.k2 = k2
        self.x0 = x0
        self.alpha = 0.1

    def compliance(self, x1, x2):
        return 1/x1 + 1/x2

    def objective(self, x1, x2):
        return self.compliance(x1, x2) - self.alpha * (np.log(x1) + np.log(x2) + np.log(1 - x1) + np.log(1 - x2))

    def constraint(self, x1, x2, lam):
        return x1 + x2 - self.x0

    def force(self, q: np.ndarray) -> np.ndarray:
        """
        q are here the design variables, namely x1, x2 and mu.
        force refers to the internal load, which is here
        g = [dC/dx1 - mu, dC/dx2 - mu, -x1 - x2], so external load f = [0, 0, 1]
        """
        x1, x2, mu = q
        return np.array([-1/(self.k1 * x1**2) - mu - self.alpha * (1/x1 - 1/(1-x1)), -1/(self.k2 * x2**2) - mu - self.alpha * (1/x2 - 1/(1-x2)), -x1 - x2 + self.x0])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        x1, x2, mu = q
        return np.array([[2 / (self.k1 * x1**3) - self.alpha * (-1/x1**2 + 1/(1-x1)**2), 0, -1],
                         [0, 2 / (self.k2 * x2**3)- self.alpha * (-1/x2**2 + 1/(1-x2)**2), -1],
                        [-1, -1, 0]])

class SpringParallel:
    def __init__(self, k1: float = 1.0, k2: float = 1.0, x0: float = 0.1):
        self.k1 = k1
        self.k2 = k2
        self.x0 = x0
        self.alpha = 0.1

    def compliance(self, x1, x2):
        return 1 / (x1**3 + x2**3)

    def objective(self, x1, x2):
        return 1/ (x1**3 + x2**3) #+ self.alpha * (np.log(x1) + np.log(x2) + np.log(1 - x1) + np.log(1 - x2))

    def constraint(self, x1, x2, lam):
        return x1 + x2 - self.x0

    def force(self, q: np.ndarray) -> np.ndarray:
        x1, x2, mu = q
        return np.array([-3 * x1**2 / (x1**3 + x2**3)**2, -3 * x2**2 / (x1**3 + x2**3)**2, -x1 - x2 + self.x0])

        # return np.array([-3 * x1**2 + self.alpha * (1/x1 - 1/(1-x1)), -3 * x2**2 + self.alpha * (1/x2 - 1/(1-x2)), -x1 - x2 + self.x0])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        x1, x2, mu = q
        return np.array([[6 * x1 * (2 * x1**3 - x2**3) / (x1**3 + x2**3)**3, 18 * x1**2 * x2**2 / (x1**3 + x2**3)**3, -1],
                         [18 * x1**2 * x2**2 / (x1**3 + x2**3)**3, 6 * x2 * (2 * x2**3 - x1**3) / (x1**3 + x2**3)**3, -1],
                        [-1, -1, 0]])


if __name__ == "__main__":
    # dofs = [x1, x2, mu]
    ixf = [0, 1, 2]

    # setup loading conditions
    ff = np.array([0, 0, -2])

    # setup problem
    system = SpringClean(k1=1.0, k2=1.0)

    x = np.linspace(0, 2, 100)
    y = np.linspace(0.01, 5, 100)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, np.sqrt(system.stiffness(X, Y)), 1000)
    plt.colorbar()


    problem = Problem(system, ixf=ixf, ff=ff)

    # # setup solver
    solver = IterativeSolver(problem, NewtonRaphson())

    # initial point
    p0 = Point(q=np.array([1.0, 1.0, 0.0]))

    # solve for equilibrium given initial point
    dp0 = solver([p0])[0]

    # get equilibrium point
    p0eq = p0 + dp0


    # setup stepper
    controller = Adaptive(value=0.1, min=0.01, max=0.1, decr=0.5)

    solver = IterativeSolver(problem, GeneralizedArcLength())
    steppah = IncrementalSolver(solution_method=solver, controller=controller, maximum_increments=3000)

    # solve problem from equilibrium point
    steppah(p0eq)

    solution = steppah.out.solutions



    plt.plot([i.q[0] for i in solution], [i.q[1] for i in solution], 'ko-')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()
