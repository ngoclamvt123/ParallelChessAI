import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t, uint8_t
from constants cimport *

ctypedef struct Position:
	np.int32_t board[MAX_BOARD_SIZE]
	np.uint8_t wc[2]
	np.uint8_t bc[2]
	np.int32_t ep
	np.int32_t kp
	np.int32_t score

cpdef Position init_position(np.int32_t[:] board,
							np.uint8_t[:] wc,
							np.uint8_t[:] bc,
							np.int32_t ep,
							np.int32_t kp,
							np.int32_t score) nogil

cdef int32_t gen_moves(Position pos, int32_t *sources, int32_t *dests) nogil

cdef Position make_move(Position pos, int32_t i, int32_t j) nogil

cdef np.int32_t evaluate(np.int32_t* board) nogil

cdef void rotate(Position* pos) nogil