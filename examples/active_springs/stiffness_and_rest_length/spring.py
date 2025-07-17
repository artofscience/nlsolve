import numpy as np


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