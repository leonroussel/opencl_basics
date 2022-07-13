""" Rotation of an image """

import numpy as np
import os
import pyopencl as cl
from silx.resources import ExternalResources
import matplotlib.pyplot as plt

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
queue = cl.CommandQueue(ctx)

# Display the input image
img = get_data("gridrec/brain_phantom.npz")["data"].astype(np.float32)
plt.imshow(img)
plt.title("Input image")
plt.colorbar()
plt.show()

# Create host and device arrays
n_row, n_col = img.shape
res = np.zeros_like(img)
d_img = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, img.nbytes)
d_res = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, img.nbytes)

# Copy on the GPU
cl.enqueue_copy(queue, d_img, img)
cl.enqueue_copy(queue, d_res, res)

# Build and execute the kernel
kernel_file = os.path.join("opencl", "rotate.cl")
kernel_src = open(kernel_file).read()
program = cl.Program(ctx, kernel_src).build()
gridsize = (512, 512)
groupsize = (16, 16)  # TODO take it into account

# Set center of rotation
center_x = int(n_col // 2)
center_y = int(n_row // 2)

theta = float(input("Rotation d'angle ? (degree): ")) / 180 * np.pi

# Launch kernel
program.gpu_rotate(
    queue,
    gridsize,
    groupsize,
    d_img,
    d_res,
    np.int32(center_x),
    np.int32(center_y),
    np.float32(theta),
    np.int32(n_col),
    np.int32(n_row),
)

# Retrieve the result from GPU
cl.enqueue_copy(queue, res, d_res)
plt.imshow(res)
plt.title("Rotated brain")
plt.colorbar()
plt.show()
