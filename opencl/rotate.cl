// Kerenl to rotate an image
__kernel void gpu_rotate(__global float *img, __global float *res, int center_x,
                         int center_y, float theta, int Ncols, int Nrows) {

  int xi = get_global_id(0); // point
  int yi = get_global_id(1);
  int x0 = center_x; // center
  int y0 = center_y;

  float r = sqrt(pown((float)x0 - xi, 2) +
                 pown((float)y0 - yi, 2)); // distance point to center

  float theta_old = atan2((float)yi - y0, (float)xi - x0) + theta;
  // printf("%f %f\n", theta, theta_old);
  int old_x = x0 + (int)r * cos(theta_old); // position of rotated points
  int old_y = y0 + (int)r * sin(theta_old);

  int id = yi * Ncols + xi; // old position in the array

  // Thread in the grid
  if (id < Ncols * Nrows) {
    // Rotated point in the grid
    if (old_x >= 0 && old_y >= 0 && old_x < Ncols && old_y < Nrows) {
      int old_id = old_y * Ncols + old_x; // new position in the array
      if (old_id == id) {
        // printf("idem\n");
      }
      res[id] = img[old_id];
    } else {
      // printf("%i %i\f", old_x, old_y);
    }
  }
}