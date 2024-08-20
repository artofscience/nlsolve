from __future__ import annotations

from abc import ABC

import numpy as np

from point import Point

State = np.ndarray[float] | None


class Structure(ABC):
    """
    Interface of a nonlinear function to the nonlinear solver.

    Both the free and prescribed degrees of freedom can be of dimension 0, 1 or higher.
    If dim(free) = 0, then dim(prescribed) > 0 and vice versa.
    That is, either external_load OR prescribed_motion OR BOTH are to be provided.
    """

    def external_load(self) -> State:
        """
        Applied external load.

        :return: None
        """
        return None

    def prescribed_motion(self) -> State:
        """
        Prescribed motion.

        :return: None
        """
        return None

    def internal_load_free(self, p: Point) -> State:
        """
        Internal load associated to the free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def internal_load_prescribed(self, p: Point) -> State:
        """
        Internal load associated to the prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def residual_free(self, p: Point) -> State:
        """
        Residual associated to the free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: residual associated to the free degrees of freedom
        """

        # free residual is defined as the free internal load PLUS the proportional loading parameter times the applied external load
        return self.internal_load_free(p) + p.y * self.external_load()

    def residual_prescribed(self, p: Point) -> State:
        """
        Residual associated to the prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: residual associated to the prescribed degrees of freedom
        """

        # prescribed residual is defined as the prescribed internal load PLUS the reaction load
        return self.internal_load_prescribed(p) + p.fp

    def tangent_stiffness_free_free(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the free-free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def tangent_stiffness_free_prescribed(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the free-prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def tangent_stiffness_prescribed_free(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the prescribed-free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def tangent_stiffness_prescribed_prescribed(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the prescribed-prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None
