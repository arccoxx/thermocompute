#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void quartic_langevin_step_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ current,
    const scalar_t* __restrict__ noise,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t event_size,
    scalar_t j2,
    scalar_t j3,
    scalar_t j4,
    scalar_t temperature,
    scalar_t dt) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  scalar_t xi = x[i];
  scalar_t ci = current[i % event_size];
  scalar_t force = -(j2 * xi + j3 * xi * xi + j4 * xi * xi * xi - ci);
  scalar_t sigma = sqrt((scalar_t)2.0 * temperature * dt);
  out[i] = xi + dt * force + sigma * noise[i];
}

torch::Tensor quartic_langevin_step_cuda(
    torch::Tensor x,
    torch::Tensor current,
    torch::Tensor noise,
    double j2,
    double j3,
    double j4,
    double temperature,
    double dt) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(current.is_cuda(), "current must be CUDA");
  TORCH_CHECK(noise.is_cuda(), "noise must be CUDA");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(current.is_contiguous(), "current must be contiguous");
  TORCH_CHECK(noise.is_contiguous(), "noise must be contiguous");
  auto out = torch::empty_like(x);
  const int threads = 256;
  const int64_t n = x.numel();
  const int blocks = (n + threads - 1) / threads;
  const int64_t event_size = current.numel();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "quartic_langevin_step_cuda", ([&] {
    quartic_langevin_step_kernel<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        current.data_ptr<scalar_t>(),
        noise.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        n,
        event_size,
        static_cast<scalar_t>(j2),
        static_cast<scalar_t>(j3),
        static_cast<scalar_t>(j4),
        static_cast<scalar_t>(temperature),
        static_cast<scalar_t>(dt));
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quartic_langevin_step_cuda", &quartic_langevin_step_cuda, "Quartic Langevin step (CUDA)");
}
