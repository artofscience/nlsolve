from involved_truss_problem import InvolvedTrussProblemLoadBased
import numpy as np
from core import IterativeSolver
from utils import Point

solution_method = IterativeSolver(InvolvedTrussProblemLoadBased())
solution, _ ,_ = solution_method([Point(qf=np.zeros(2), ff=np.zeros(2))])

print("Load:", solution.ff) # print applied load at dof 1 and 2
print("Motion:", solution.qf) # print resulting motion of dof 1 and 2

