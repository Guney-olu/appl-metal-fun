import ctypes
import numpy as np
import os
cwd = os.getcwd()

lib = cwd + "/matmul_package/PyMetalBridge/.build/release/libPyMetalBridge.dylib"

swift_fun = ctypes.CDLL(lib)
swift_fun.swift_matrix_multiplication_on_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

def swift_matrix_multiplication_on_gpu(A, B, M, N, K):
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result = np.zeros((M, N), dtype=np.float32)
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    swift_fun.swift_matrix_multiplication_on_gpu(A_ptr, B_ptr, result_ptr, M, N, K)
    return result

# Example usage
M, K, N = 4, 3, 5
A = np.random.uniform(-1, 1, (M, K)).astype("float32")
B = np.random.uniform(-1, 1, (K, N)).astype("float32")

swift_result = swift_matrix_multiplication_on_gpu(A, B, M, N, K)
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Result of A * B:\n", swift_result)
