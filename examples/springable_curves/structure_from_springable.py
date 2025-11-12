import numpy as np

from springable.behavior_creation import start_behavior_creation
from springable.readwrite.fileio import read_behavior, read_model
from springable.mechanics.element import Element
from springable.mechanics.shape import SegmentLength, Shape
from springable.mechanics.node import Node


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
    
class SpringFromUnivariateBehavior:
    def __init__(self, filepath: str):
        """

        Args:
            filepath (str): path to behavior csv file
        """
        self._behavior = read_behavior(filepath)

    def force(self, a):
        u = a
        alpha = u + self._behavior.get_natural_measure()
        dvdalpha = self._behavior.gradient_energy(alpha)[0]
        return dvdalpha

    def jacobian(self, a):
        u = a
        alpha = u + self._behavior.get_natural_measure()
        d2vdalpha2 = self._behavior.hessian_energy(alpha)[0]
        return d2vdalpha2
    
    def get_rest_length(self):
        return self._behavior.get_natural_measure()
    
class LongitudinalSpringFromUnivariateBehavior:
    def __init__(self, filepath: str):
        """

        Args:
            filepath (str): path to behavior csv file
        """
        self._behavior = read_behavior(filepath)

    def force(self, x1, y1, x2, y2):
        n1 = Node(x1, y1, False, False)
        n2 = Node(x2, y2, False, False)
        l, dl = SegmentLength(n1, n2).compute(mode=Shape.MEASURE_JACOBIAN)
        dv = self._behavior.gradient_energy(l)[0]
        return dv * dl
    
    def jacobian(self, x1, y1, x2, y2):
        n1 = Node(x1, y1, False, False)
        n2 = Node(x2, y2, False, False)
        l, dl, d2l = SegmentLength(n1, n2).compute(mode=Shape.MEASURE_JACOBIAN_AND_HESSIAN)
        dv = self._behavior.gradient_energy(l)[0]
        d2v = self._behavior.hessian_energy(l)[0]
        return d2v * np.outer(dl, dl) + dv * d2l






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


class ActiveStructureBasedOnSptringableModelFile:

    def __init__(self, springable_model_filepath: str,
                 connecting_active_springs: list,
                 connected_nodes: list[tuple[int]]):
        sga_model = read_model(springable_model_filepath)
        # WORK IN PROGRESS

if __name__ == "__main__":
    start_behavior_creation()
