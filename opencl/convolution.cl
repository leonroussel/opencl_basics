// Kernel to convolve an image with a kernel/filter 
__kernel void gpu_convolve(__global float *img, __global float *res,
                         __global float *ker, int ker_h_s, int Ncols,
                         int Nrows) {

  int xi = get_global_id(0); // point
  int yi = get_global_id(1);
  int id = yi * Ncols + xi; // global index
  if (id < Ncols * Nrows) {
    float sum = 0;
    // Weighted sum with the neighbor pixels 
    for (int i = -ker_h_s; i <= ker_h_s; i++) {
      for (int j = -ker_h_s; j <= ker_h_s; j++) { 
        int ii = xi + i;
        int jj = yi + j;
        int curr_id = jj * Ncols + ii;
        // 0 padding :
        if (!(ii < 0 || ii >= Ncols || jj < 0 || jj >= Nrows)) {
          sum += ker[i + ker_h_s] * ker[j + ker_h_s] * img[curr_id];
        }
      }
    }
    res[id] = sum;
  }
}