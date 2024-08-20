from __future__ import annotations

from abc import ABC

import numpy as np

from point import Point

State = np.ndarray[float] | float | None


class Structure(ABC):
    def external_load(self) -> State:
        return None

    def prescribed_motion(self) -> State:
        return None

    def internal_load_free(self, p: Point) -> State:
        return None

    def internal_load_prescribed(self, p: Point) -> State:
        return None

    def residual_free(self, p: Point) -> State:
        return self.internal_load_free(p) + p.y * self.external_load()

    def residual_prescribed(self, p: Point) -> State:
        return self.internal_load_prescribed(p) + p.fp

    def tangent_stiffness_free_free(self, p: Point) -> State:
        return None

    def tangent_stiffness_free_prescribed(self, p: Point) -> State:
        return None

    def tangent_stiffness_prescribed_free(self, p: Point) -> State:
        return None

    def tangent_stiffness_prescribed_prescribed(self, p: Point) -> State:
        return None
