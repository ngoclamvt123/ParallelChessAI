from libc.stdint cimport int32_t, uint8_t
cimport chess
from chess cimport Position

cdef int alpha_beta_serial(Position pos, int agentIndex, int depth, int alpha, int beta, int32_t *move) nogil