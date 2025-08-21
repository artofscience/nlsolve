from core import IterativeSolver, IncrementalSolver
from criteria import EigenvalueChangeTermination, termination_default
from constraints import GeneralizedArcLength

def stepper(problem, default_positive_direction: bool = True, max_load: float = 1.0):
    constraint = GeneralizedArcLength(default_positive_direction=default_positive_direction)
    solver = IterativeSolver(problem, constraint, maximum_corrections=10)
    criteria = termination_default(max_load) | EigenvalueChangeTermination()
    return IncrementalSolver(solver, terminated=criteria, reset=False)