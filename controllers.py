"""
This file contains objects that act as a controller for the incremental solver.
These objects control the characteristic length of the constraint function used and modify its value.
Different controllers have different behaviour in terms of what observations are used to modify and how to modify its value.
"""

from logger import CustomFormatter, create_logger
import logging

class Controller:
    """
    This controller holds a constant value equal to its initial value,
     both incr and decr functions do not alter the imposed value.
    """

    def __init__(self, value: float = 0.1, name: str = None, logging_level: int = logging.DEBUG) -> None:
        self.value = value
        self.value0 = value

        self.__name__ = name if name is not None else (self.__class__.__name__ + " " + str(id(self)))

        self.logger = create_logger(self.__name__, logging_level, CustomFormatter())
        self.logger.info("Initializing a " + self.__class__.__name__ + " called " + self.__name__)

    def increase(self) -> None:
        self.logger.warning("Controller increase invoked, but value remains unchanged to %2.2f" % self.value)

    def decrease(self) -> None:
        self.logger.warning("Controller decrease invoked, but value remains unchanged to %2.2f" % self.value)

    def reset(self) -> None:
        self.value = self.value0


class Adaptive(Controller):
    """
    Controller that changes value based on incr and decr values,
    taking into account preset minimum and maximum values.
    """

    def __init__(self, value: float = 0.05, name: str = None, logging_level: int = logging.DEBUG,
                 incr: float = 1.3, decr: float = 0.3,
                 min: float = 0.0001, max: float = 0.1) -> None:
        super().__init__(value, name, logging_level)
        self.incr = incr
        self.decr = decr
        self.min = min
        self.max = max

    def increase(self) -> None:
        if (value := self.incr * self.value) < self.max:
            self.logger.debug("Controller value increased from %2.2f to %2.2f" % (self.value, value))
            self.value = value
        else:
            self.logger.warning("Controller value reached maximum value of %2.2f" % self.max)
            self.value = self.max

    def decrease(self) -> None:
        if (value := self.decr * self.value) > self.min:
            self.logger.warning("Controller value decreased from %2.2f to %2.2f" % (self.value, value))
            self.value = value
        else:
            self.logger.warning("Controller value reached minimum value of %2.2f" % self.min)
            self.value = self.min