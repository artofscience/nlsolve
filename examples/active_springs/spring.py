import numpy as np
from collections.abc import Callable
from sympy import diff, lambdify

class Spring:
    """Inactive spring"""

    def __init__(self, k: float = 1.0, l0: float = 1.0):
        self.k = k
        self.l0 = l0

    def force(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 5 coordinates describing the 5 degrees of freedom of the spring:
        q = [x0, y0, x1, y1, l0], where x0, y0, x1, y1 are the coordinates of the nodes and l0 the rest length.

        :param k: spring constant (fixed parameter).
        :return: internal force vector of the individual spring:
        gradient of the elastic energy with respect to the 5 coordinates.
        """
        x0, y0, x1, y1 = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy]) / l
        return self.k * (l - self.l0) * dldq

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 5 coordinates describing the 5 degrees of freedom of the spring:
        q = [x0, y0, x1, y1, l0], where x0, y0, x1, y1 are the coordinates of the nodes and l0 the rest length.

        :param k: spring constant (fixed parameter).
        :return: stiffness matrix of the individual spring:
        hessian of the elastic energy with respect to the 5 coordinates
        """
        x0, y0, x1, y1 = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy]) / l
        k_matrix = self.k * np.outer(dldq, dldq)
        k_matrix += self.k * (l - self.l0) / l * np.array([[1, 0, -1, 0],
                                                           [0, 1, 0, -1],
                                                           [-1, 0, 1, 0],
                                                           [0, -1, 0, 1]])
        k_matrix -= self.k * (l - self.l0) / l * np.outer(dldq, dldq)
        return k_matrix

class SpringL0:
    """"Spring where initial length L0 is a DOF"""

    def __init__(self, k: float = 1.0):
        self.k = k

    def force(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 5 coordinates describing the 5 degrees of freedom of the spring:
        q = [x0, y0, x1, y1, l0], where x0, y0, x1, y1 are the coordinates of the nodes and l0 the rest length.

        :param k: spring constant (fixed parameter).
        :return: internal force vector of the individual spring:
        gradient of the elastic energy with respect to the 5 coordinates.
        """
        x0, y0, x1, y1, l0 = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0]) / l
        f = self.k * (l - l0) * dldq
        f[-1] -= self.k * (l - l0)
        return f

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 5 coordinates describing the 5 degrees of freedom of the spring:
        q = [x0, y0, x1, y1, l0], where x0, y0, x1, y1 are the coordinates of the nodes and l0 the rest length.

        :param k: spring constant (fixed parameter).
        :return: stiffness matrix of the individual spring:
        hessian of the elastic energy with respect to the 5 coordinates
        """
        x0, y0, x1, y1, l0 = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0]) / l
        k_matrix = self.k * np.outer(dldq, dldq)
        k_matrix[:, -1] -= self.k * dldq
        k_matrix[-1, :] -= self.k * dldq
        k_matrix[:4, :4] += self.k * (l - l0) / l * np.array([[1, 0, -1, 0],
                                                              [0, 1, 0, -1],
                                                              [-1, 0, 1, 0],
                                                              [0, -1, 0, 1]])
        k_matrix -= self.k * (l - l0) / l * np.outer(dldq, dldq)
        k_matrix[-1, -1] += self.k
        return k_matrix


class SpringK:
    """"Spring where stiffness k is a DOF"""

    def __init__(self, l0: float = 1.0):
        self.l0 = l0

    def force(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 6 coordinates describing the 6 degrees of freedom of the spring:
        q = [x0, y0, x1, y1, k], where x0, y0, x1, y1 are the coordinates of the nodes,
        l0 the rest length, and k the spring constant.

        :return: internal force vector of the individual spring:
        gradient of the elastic energy with respect to the 6 coordinates.
        """
        x0, y0, x1, y1, k = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0]) / l
        f = k * (l - self.l0) * dldq
        f[-1] += - 0.5 * (l - self.l0) ** 2
        return f

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 6 coordinates describing the 6 degrees of freedom of the spring:
        q = [x0, y0, x1, y1, l0, k], where x0, y0, x1, y1 are the coordinates of the nodes,
        l0 the rest length, and k the spring constant.

        :return: stiffness matrix of the individual spring:
        hessian of the elastic energy with respect to the 6 coordinates
        """
        x0, y0, x1, y1, k = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0]) / l
        k_matrix = k * np.outer(dldq, dldq)
        k_matrix[:, -1] += (l - self.l0) * dldq
        k_matrix[-1, :] += (l - self.l0) * dldq
        k_matrix[:4, :4] += k * (l - self.l0) / l * np.array([[1, 0, -1, 0],
                                                              [0, 1, 0, -1],
                                                              [-1, 0, 1, 0],
                                                              [0, -1, 0, 1]])
        k_matrix += -k * (l - self.l0) / l * np.outer(dldq, dldq)
        k_matrix[-1, -1] += 0
        return k_matrix


class SpringL0K:
    """"Spring where both initial length L0 and stiffness k are DOFs"""

    def force(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 6 coordinates describing the 6 degrees of freedom of the spring:
        q = [x0, y0, x1, y1, l0, k], where x0, y0, x1, y1 are the coordinates of the nodes,
        l0 the rest length, and k the spring constant.

        :return: internal force vector of the individual spring:
        gradient of the elastic energy with respect to the 6 coordinates.
        """
        x0, y0, x1, y1, l0, k = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0, 0]) / l
        f = k * (l - l0) * dldq
        f[-2] += - k * (l - l0)
        f[-1] += - 0.5 * (l - l0) ** 2
        return f

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 6 coordinates describing the 6 degrees of freedom of the spring:
        q = [x0, y0, x1, y1, l0, k], where x0, y0, x1, y1 are the coordinates of the nodes,
        l0 the rest length, and k the spring constant.

        :return: stiffness matrix of the individual spring:
        hessian of the elastic energy with respect to the 6 coordinates
        """
        x0, y0, x1, y1, l0, k = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0, 0]) / l
        k_matrix = k * np.outer(dldq, dldq)
        k_matrix[:, -2] += -k * dldq
        k_matrix[-2, :] += -k * dldq
        k_matrix[:, -1] += (l - l0) * dldq
        k_matrix[-1, :] += (l - l0) * dldq
        k_matrix[:4, :4] += k * (l - l0) / l * np.array([[1, 0, -1, 0],
                                                         [0, 1, 0, -1],
                                                         [-1, 0, 1, 0],
                                                         [0, -1, 0, 1]])
        k_matrix += -k * (l - l0) / l * np.outer(dldq, dldq)
        k_matrix[-2, -2] += k
        k_matrix[-2, -1] += -(l - l0)
        k_matrix[-1, -2] += -(l - l0)
        k_matrix[-1, -1] += 0
        return k_matrix


class SpringT:
    """Spring where temperature T is a DOF.
    Initial length L0(T) and k(T)"""

    def __init__(self, l0, k):
        x = "T"
        self.k = lambdify(x, k, modules='numpy')
        self.l0 = lambdify(x, l0, modules='numpy')
        self.dl0dt = lambdify("T", diff(l0, x), modules='numpy')
        self.dkdt = lambdify("T", diff(k, x), modules='numpy')
        self.d2l0dt2 = lambdify("T", diff(l0, x, 2), modules='numpy')
        self.d2kdt2 = lambdify("T", diff(k, x, 2), modules='numpy')

    def force(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 5 coordinates describing the 5 degrees of freedom of the spring:
               q = [x0, y0, x1, y1, t], where x0, y0, x1, y1 are the coordinates of the nodes,
               l0 the rest length, and k the spring constant.
        :param l0: function that returns the rest length given a value of coordinate t
        :param k: function that returns the spring constant given a value of coordinate t
        :param dl0dt: function that returns the derivative of the rest length with respect to coordinate t,
               given a value of coordinate t
        :param dkdt: function that returns the derivative of the spring constant with respect to coordinate t,
               given a value of coordinate t

        :return: internal force vector of the individual spring:
                 gradient of the elastic energy with respect to the 5 coordinates.
        """
        x0, y0, x1, y1, t = q
        l0 = self.l0(t)
        k = self.k(t)

        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0]) / l

        f = k * (l - l0) * dldq
        f[-1] += (0.5 * (l - l0) ** 2 * self.dkdt(t) - k * (l - l0) * self.dl0dt(t))
        return f

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 5 coordinates describing the 5 degrees of freedom of the spring:
               q = [x0, y0, x1, y1, t], where x0, y0, x1, y1 are the coordinates of the nodes,
               l0 the rest length, and k the spring constant.
        :param l0: function that returns the rest length given a value of coordinate t
        :param k: function that returns the spring constant given a value of coordinate t
        :param dl0dt: function that returns the first derivative of the rest length with respect to coordinate t,
               given a value of coordinate t
        :param dkdt: function that returns the first derivative of the spring constant with respect to coordinate t,
               given a value of coordinate t
        :param d2l0dt2: function that returns the second derivative of the rest length with respect to coordinate t,
               given a value of coordinate t
        :param d2kdt2: function that returns the second derivative of the spring constant with respect to coordinate t,
               given a value of coordinate t

        :return: stiffness matrix of the individual spring:
                 hessian of the elastic energy with respect to the 5 coordinates.
        """
        x0, y0, x1, y1, t = q
        l0 = self.l0(t)
        k = self.k(t)
        dl0dt = self.dl0dt(t)
        dkdt = self.dkdt(t)
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0]) / l

        k_matrix = k * np.outer(dldq, dldq)
        k_matrix[:4, :4] += k * (l - l0) / l * np.array([[1, 0, -1, 0],
                                                         [0, 1, 0, -1],
                                                         [-1, 0, 1, 0],
                                                         [0, -1, 0, 1]])
        k_matrix += -k * (l - l0) / l * np.outer(dldq, dldq)
        k_matrix[:, -1] += ((l - l0) * dkdt + -k * dl0dt) * dldq
        k_matrix[-1, :] += ((l - l0) * dkdt + -k * dl0dt) * dldq
        k_matrix[-1, -1] += (0.5 * (l - l0) ** 2 * self.d2kdt2(t) + -k * (l - l0) * self.d2l0dt2(t))
        k_matrix[-1, -1] += (0 * dkdt ** 2 + 2 * -(l - l0) * dkdt * dl0dt + k * dl0dt ** 2)
        return k_matrix


class SpringTCallable:
    """Spring where temperature T is a DOF.
    Initial length L0(T) and k(T)"""

    def __init__(self, l0: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                 k: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                 dl0dt: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                 dkdt: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                 d2l0dt2: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                 d2kdt2: Callable[[float], float] | Callable[[np.ndarray], np.ndarray]):
        self.myspring = SpringT(0,0)

        self.myspring.k = k
        self.myspring.l0 = l0
        self.myspring.dl0dt = dl0dt
        self.myspring.dkdt = dkdt
        self.myspring.d2l0dt2 = d2l0dt2
        self.myspring.d2kdt2 = d2kdt2

    def force(self, q: np.ndarray) -> np.ndarray:
        return self.myspring.force(q)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        return self.myspring.jacobian(q)