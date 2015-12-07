cimport chess
from chess cimport Position

cpdef int alpha_beta_serial(Position pos, int agentIndex, int depth, int alpha, int beta) nogil