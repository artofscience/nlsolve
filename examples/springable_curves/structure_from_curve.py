from springable.readwrite.fileio import read_behavior
import numpy as np


class StructureFromCurve:
    def __init__(self, filepath: str):
        self._behavior = read_behavior(filepath)

    def force(self, a):
        u, t = a[0], a[1]
        alpha = u + self._behavior.get_natural_measure()
        dvdalpha, dvdt = self._behavior.gradient_energy(alpha, t)
        return np.array([dvdalpha, dvdt])
    
    def jacobian(self, a):
        u, t = a[0], a[1]
        alpha = u + self._behavior.get_natural_measure()
        d2vdalpha2, d2vdalphadt, d2vdt2 = self._behavior.hessian_energy(alpha, t)
        return np.array([[d2vdalpha2, d2vdalphadt], [d2vdalphadt, d2vdt2]])