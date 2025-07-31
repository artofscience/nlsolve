from springable.readwrite.fileio import read_model
import numpy as np


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
            np.ndarray: gradient of elastic energy wrt the free displacements
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