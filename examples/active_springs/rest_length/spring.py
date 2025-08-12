import numpy as np


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
