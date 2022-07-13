""" Convolution of an image with a triangle filter """

import numpy as np
import os
import pyopencl as cl
from silx.resources import ExternalResources
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d

# Import brain phantom (an image)
utilstest = ExternalResources(
    project="nabu",
    url_base="http://www.silx.org/pub/nabu/data/",
    env_key="NABU_DATA",
    timeout=60,
)


def get_data(relative_path):
    """
    Get the numpy array strored at http://www.silx.org/pub/nabu/data/<relative_path>
    Here, relative_path should begin with 'gridrec/'
    """
    dataset_downloaded_path = utilstest.getfile(relative_path)
    return np.load(dataset_downloaded_path)


# Create context and queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Get input image
img = get_data("gridrec/brain_phantom.npz")["data"].astype(np.float32)

# Create host and device arrays
n_row, n_col = img.shape
res = np.zeros_like(img)
d_img = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, img.nbytes)
d_res = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, img.nbytes)

# Build filter of convolution
kernel_half_size = 10
# Filter 1D is a triangle
kernel_x = np.concatenate(  # separate filter
    (
        np.arange(0, kernel_half_size),
        [kernel_half_size],
        np.flip(np.arange(0, kernel_half_size)),
    )
).astype(np.float32)

# Create buffer filter
d_ker = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, kernel_x.nbytes)

# Copy on the GPU
cl.enqueue_copy(queue, d_img, img)
cl.enqueue_copy(queue, d_res, res)
cl.enqueue_copy(queue, d_ker, kernel_x)

# Build and execute the kernel
kernel_file = os.path.join("opencl", "convolution.cl")
kernel_src = open(kernel_file).read()
program = cl.Program(ctx, kernel_src).build()
gridsize = (512, 512)
groupsize = None

# Launch kernel
event = program.gpu_convolve(
    queue,
    gridsize,
    groupsize,
    d_img,
    d_res,
    d_ker,
    np.int32(kernel_half_size),
    np.int32(n_col),
    np.int32(n_row),
)
event.wait()
elapsed_time = 1e-9 * (event.profile.end - event.profile.start)
print("Execution time of test: %g s" % elapsed_time)  # warning, only computing time

# Retrieve the result from GPU
cl.enqueue_copy(queue, res, d_res)
plt.imshow(res)
plt.colorbar()
plt.title("OpenCL convolution")
plt.show()

# Display result and compare to scipy convolution
st = time.time()
ker = np.outer(kernel_x, kernel_x)
res_conv = convolve2d(img, ker)
end = time.time()
print("numpy elapsed time : ", end - st)

plt.imshow(res_conv)
plt.title("Numpy convolution")
plt.colorbar()
plt.show()
