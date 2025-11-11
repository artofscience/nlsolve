import numpy as np

from springable.behavior_creation import start_behavior_creation
from springable.readwrite.fileio import read_behavior, read_model

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


class StructureFromSpringableModelFile:
    def __init__(self, model_filepath: str):
        self._mdl = read_model(model_filepath)
        self._asb = self._mdl.get_assembly()
        self._q0 = self._asb.get_coordinates()
        # initial coordinates, as described in the input file.
        # Watch out: those initial coordinates are not necessarily describing a state at equilibrium (under zero forces)

    def force(self, a):
        """

        Args:
            a (np.ndarray): displacements of the coordinates

        Returns:
            np.ndarray: gradient of elastic energy wrt the displacements
        """
        self._asb.set_coordinates(self._q0 + a)
        return self._asb.compute_elastic_force_vector()

    def jacobian(self, a):
        """

        Args:
            a (np.ndarray): displacements of the coordinates

        Returns:
            np.ndarray: hessian matrix of elastic energy wrt the displacements
        """
        self._asb.set_coordinates(self._q0 + a)
        return self._asb.compute_structural_stiffness_matrix()

    def get_default_ixf(self) -> list[int]:
        return self._asb.get_free_dof_indices()

    def get_default_ixp(self) -> list[int]:
        return self._asb.get_fixed_dof_indices()

    def get_default_ff(self) -> np.ndarray:
        return self._mdl.get_force_vector()[self._asb.get_free_dof_indices()]

    def get_default_qp(self) -> np.ndarray:
        return np.zeros_like(self._asb.get_fixed_dof_indices())

if __name__ == "__main__":
    start_behavior_creation()