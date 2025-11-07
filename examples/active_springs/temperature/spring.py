import numpy as np
from collections.abc import Callable
from utils import Problem
from typing import Sequence
from numpy.polynomial.polynomial import Polynomial



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
               t the temperature
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
               t the temperature
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


class NonlinearPolySpringT:
    """ Nonlinear spring whose force-displacement curve is a polynomial in u, with each coefficient of that polynomial being itself a polynomial in T.
        f(u, T) = sum_i sum_j a_ij * u^i * T^j. The dependence on the rest length on T is also a polynomial: l0(T) = sum_i l_i T^i.
        /!\\ To make sure the force at rest (u=0) is zero, make sure that a_0j = 0 for all j /!\\ 
    """

    def __init__(self, li: np.ndarray, aij: np.ndarray):
        self._n = aij.shape[0]
        self._l0 = Polynomial(li)
        self._a = lambda t: np.array([Polynomial(aij[i, :])(t) for i in range(self._n)])

        # first order deriv
        self._dl0dt = self._l0.deriv()
        self._dadt = lambda t: np.array([Polynomial(aij[i, :]).deriv()(t) for i in range(self._n)])

        # second order deriv
        self._d2l0dt2 = self._l0.deriv(2)
        self._d2adt2 = lambda t: np.array([Polynomial(aij[i, :]).deriv(2)(t) for i in range(self._n)])

    def _dvdl(self, l, t):
        return Polynomial(self._a(t))(l - self._l0(t))
    
    def _dvdl0(self, l, t):
        return -Polynomial(self._a(t))(l - self._l0(t))
    
    def _dvda(self, l, t):
        return np.array([(l - self._l0(t))**(i+1)/(i+1) for i in range(self._n)])
    
    def _d2vdl2(self, l, t):
        return Polynomial(self._a(t)).deriv()(l - self._l0(t))
    
    def _d2vdldl0(self, l, t):
        return -Polynomial(self._a(t)).deriv()(l - self._l0(t))
    
    def _d2vdlda(self, l, t):
        return np.array([(l - self._l0(t))**i for i in range(self._n)])
    
    def _d2vdl02(self, l, t):
        return Polynomial(self._a(t)).deriv()(l - self._l0(t))
    
    def _d2vdl0da(self, l, t):
        return -np.array([(l - self._l0(t))**i for i in range(self._n)])
    
    def _d2vda2(self, l, t):
        return np.zeros(shape=(self._n, self._n))
    
    def _dvdp(self, l, t):
        return np.array([self._dvdl0(l, t)] + self._dvda(l, t).tolist())
    
    def _d2vdldp(self, l, t):
        return np.array([self._d2vdldl0(l, t)] + self._d2vdlda(l, t).tolist())
    
    def _d2vdp2(self, l, t):
        d2vdp2 = np.zeros(shape=(self._n+1, self._n+1))
        d2vdp2[0, 0] = self._d2vdldl0(l, t)
        d2vdp2[0, 1:] = self._d2vdlda(l, t)
        d2vdp2[1:, 0] = self._d2vdlda(l, t)
        d2vdp2[1:, 1:] = self._d2vda2(l, t)
        return d2vdp2
    
    def _dpdt(self, t):
        return np.array([self._dl0dt(t)] + self._dadt(t).tolist())
    
    def _d2pdt2(self, t):
        return np.array([self._d2l0dt2(t)] + self._d2adt2(t).tolist())



    def force(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 5 coordinates describing the 5 degrees of freedom of the spring:
               q = [x0, y0, x1, y1, t], where x0, y0, x1, y1 are the coordinates of the nodes,
               t the temperature
        """
        x0, y0, x1, y1, t = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0]) / l

        f = self._dvdl(l, t) * dldq
        f[-1] += self._dvdp(l, t) @ self._dpdt(t)
        return f

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        :param q: array containing the 5 coordinates describing the 5 degrees of freedom of the spring:
               q = [x0, y0, x1, y1, t], where x0, y0, x1, y1 are the coordinates of the nodes,
               t the temperature
        """
        x0, y0, x1, y1, t = q
        dx = x0 - x1
        dy = y0 - y1
        l = np.sqrt(dx ** 2 + dy ** 2)
        dldq = np.array([dx, dy, -dx, -dy, 0]) / l

        k_matrix = self._d2vdl2(l, t) * np.outer(dldq, dldq)
        k_matrix[:4, :4] += self._dvdl(l, t) * (1 / l * np.array([[1, 0, -1, 0],
                                                                  [0, 1, 0, -1],
                                                                  [-1, 0, 1, 0],
                                                                  [0, -1, 0, 1]])
                                                -1 / l * np.outer(dldq, dldq)[:4, :4])
        
        k_matrix[:, -1] +=  (self._d2vdldp(l, t) @ self._dpdt(t)) * dldq
        k_matrix[-1, :] +=  (self._d2vdldp(l, t) @ self._dpdt(t)) * dldq
        k_matrix[-1, -1] += (self._dvdp(l, t) @ self._d2pdt2(t) + self._dpdt(t) @ self._d2vdp2(l, t) @ self._dpdt(t))
        return k_matrix