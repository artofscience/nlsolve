import numpy as np
from collections.abc import Callable
from utils import Problem



class SpringT:
    """Spring where temperature T is a DOF.
    Initial length L0(T) and k(T)"""
    def __init__(self,  l0: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                 k: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                dl0dt: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                dkdt: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                d2l0dt2: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
                d2kdt2: Callable[[float], float] | Callable[[np.ndarray], np.ndarray]):
        self.k = k
        self.l0 = l0
        self.dl0dt = dl0dt
        self.dkdt = dkdt
        self.d2l0dt2 = d2l0dt2
        self.d2kdt2 = d2kdt2

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