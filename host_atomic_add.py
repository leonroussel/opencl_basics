""" Atomic add avoid confilct of writing """

import numpy as np
import os
import pyopencl as cl

# Create context and queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Create host and device arrays
N = 10
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
d_a = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, a.nbytes)
d_b = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, b.nbytes)

# Copy "a" and "b" on the GPU
cl.enqueue_copy(queue, d_a, a)
cl.enqueue_copy(queue, d_b, b)

# Build and execute the kernel
kernel_file = os.path.join("opencl", "atomic_add.cl")
kernel_src = open(kernel_file).read()
program = cl.Program(ctx, kernel_src).build()
gridsize = (N,)
groupsize = None
program.gpu_add_float(queue, gridsize, groupsize, d_a, d_b, np.int32(N))

# Retrieve the result from GPU
res = np.zeros_like(a)
cl.enqueue_copy(queue, res, d_a)

# Display results
print("Close ? : ", np.allclose(res, a + b))

print("res :", res)
print("a + b :", a + b)
