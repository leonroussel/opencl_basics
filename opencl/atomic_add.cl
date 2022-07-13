#include <pyopencl-complex.h>

// Atomic add on float (not available natively)
inline void atomic_add_global_float(volatile global float *addr, float val) {
  union {
    uint u32;
    float f32;
  } next, expected, current;
  current.f32 = *addr;
  do {
    expected.f32 = current.f32;
    next.f32 = expected.f32 + val;
    current.u32 =
        atomic_cmpxchg((volatile global uint *)addr, expected.u32, next.u32);
  } while (current.u32 != expected.u32);
}

// Add with atomic_add_float (just to verify the function is running, but no conflict here)
__kernel void gpu_add_float(__global float *arr1, __global float *arr2, int N) {
  int tid = get_global_id(0); // Thread ID
  if (tid < N) {
    atomic_add_global_float(&arr1[tid], arr2[tid]); // vector summation
  }
}
