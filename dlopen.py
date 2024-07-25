import ctypes
import numpy as np
import os
cwd = os.getcwd()

lib = cwd + "/.build/release/libPyMetalBridge.dylib"

swift_fun = ctypes.CDLL(lib)
swift_fun.swift_sigmoid_on_gpu.argtypes = [
	ctypes.POINTER(ctypes.c_float),
	ctypes.POINTER(ctypes.c_float),
	ctypes.c_int
]

def swift_sigmoid_on_gpu(input_array):
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_mutable_ptr = (ctypes.c_float * len(input_array))()
    swift_fun.swift_sigmoid_on_gpu(input_ptr, output_mutable_ptr, len(input_array))
    return np.array(output_mutable_ptr)

input_array = np.random.uniform(-1,1,100).astype("float32")
swift_res = swift_sigmoid_on_gpu(input_array)
print(swift_res)
