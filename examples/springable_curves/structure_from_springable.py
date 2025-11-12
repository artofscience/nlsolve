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

    def get_rest_length(self):
        return self._behavior.get_natural_measure()




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


class ActiveStructureBasedOnSpringableModelFile:
    """
    Structure composed of passive and active springs, based on a Springable model file.
    All active elements in the structures share the same active DOF.
    This class implements a force() and jacobian function() given an array of coordinates [all spatial coordinates, temperature coordinate]

    """

    def __init__(self, springable_model_filepath: str,
                 connecting_active_springs_list: list,
                 connected_nodes_list: list[tuple[int]]):
        """
        Parameters
        ----------
        springable_model_filepath: str
            path to csv model file that describes the springable assembly.
            Nodes, boundary conditions and (passive) springs will be used. LOADING section will be ignored.
        connecting_active_springs_list: list
            list of active springs that have temperature T
            as an additional degree of freedom. They will all share the same temperature. Each active spring should implement a force() and jacobian function() that takes an array of [multiple spatial coordinates and one temperature coordinate] as input.
        connected_nodes_list: list[tuple[int]]
            list of tuples containing the indices of the nodes connected by the active spring. Each tuple corresponds to one active spring. The node indices should match the node indices from the springable model csv file.
        
        Example
        -------
        >>> model_filepath = 'mymodel.csv'
        >>> s1 = SpringT(l0=3-T, k=5-T)
        >>> s2 = SpringT(l0=1-1*T, k=2-T)
        >>> s3 = SpringT(l0=4-T, k=5-2*T)
        >>> s4 = NonlinearPolySpringT(li=np.array([1.0, 1.0]), aij=np.array([[0, 0, 0], [1, 1.2, 1], [-0.2, 3, 1.4]]))
        >>> connecting_active_springs_list = [s1, s2, s3, s4]
        >>> connected_nodes_list = [(0, 2), (5, 4), (0, 3), (4, 1)]  # nodes 0,1,2,3,4,5 should exist in the springable model file
        >>> assert(len(connected_nodes_list) == len(connecting_active_springs))
        >>> active_struct = ActiveStructureBasedOnSpringableModelFile(model_filepath, connecting_active_springs, connected_nodes_list)
        """
        if len(connected_nodes_list) != len(connecting_active_springs_list):
            raise ValueError("There should be as many connecting active springs as tuples of connected nodes in both respective lists.")
        self._sga_assembly = read_model(springable_model_filepath).get_assembly()
        self._spatial_q0 = self._sga_assembly.get_coordinates().copy()
        self._nb_dofs = self._sga_assembly.get_nb_dofs() + 1
        self._connecting_active_springs_list = connecting_active_springs_list
        self._active_springs_dof_indices = []

        node2dof_indices = self._sga_assembly.get_nodes_dof_indices()
        for connected_nodes in connected_nodes_list:
            indices = []
            for connected_node in connected_nodes:
                indices += node2dof_indices[connected_node]

            indices.append(self._nb_dofs - 1)
            # because each active spring depends on the last coordinate, temperature

            self._active_springs_dof_indices.append(indices)

    def get_initial_spatial_coordinates(self):
        """
        returns the initial spatial coordinates (does not contain the temperature coordinate)
        as described in the model csv file.
        Watch out: those initial coordinates are not necessarily describing a state at equilibrium (under zero forces)
        """
        return self._spatial_q0
        


    def force(self, q):
        """ q is a numpy array containing the current coordinates of each DOF.
        The order of the coordinates is the following:
        q = [x0, y0, x1, y1, ..., xn-1, yn-1, T]"""
        force = np.zeros(self._nb_dofs)
        self._sga_assembly.set_coordinates(q[:-1])
        force[:-1] = self._sga_assembly.compute_elastic_force_vector()

        for active_spring, dof_indices in zip(self._connecting_active_springs_list,
                                              self._active_springs_dof_indices):
            force[dof_indices] += active_spring.force(q[dof_indices])
        return force
    
    def jacobian(self, q):
        """ q is a numpy array containing the current coordinates of each DOF.
        The order of the coordinates is the following:
        q = [x0, y0, x1, y1, ..., xn-1, yn-1, T]"""
        jacobian = np.zeros(shape=(self._nb_dofs, self._nb_dofs))
        self._sga_assembly.set_coordinates(q[:-1])
        jacobian[:-1, :-1] = self._sga_assembly.compute_structural_stiffness_matrix()

        for active_spring, dof_indices in zip(self._connecting_active_springs_list,
                                              self._active_springs_dof_indices):
            jacobian[np.ix_(dof_indices, dof_indices)] += active_spring.jacobian(q[dof_indices])
        return jacobian
    
    def get_default_ixf(self) -> list[int]:
        return self._sga_assembly.get_free_dof_indices() + [self._nb_dofs - 1]

    def get_default_ixp(self) -> list[int]:
        return self._sga_assembly.get_fixed_dof_indices()
    

        

if __name__ == "__main__":
    start_behavior_creation()
