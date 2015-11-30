#cython: boundscheck=False,
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython

cpdef void testing(np.int32_t[:, :] board) nogil:
	with gil:
		print(np.array_str(board))
		print("ayy")
		print("HEYY")
		raw_input()

cpdef int score_board(np.int32_t[:, :] board, np.int32_t[:] pos_values) nogil:
	cdef:
		int i, j, total

	total = 0
	for i in range(8):
		for j in range(8):
				total += pos_values[board[i][j]]
	with gil:
		print total
	return total



	
