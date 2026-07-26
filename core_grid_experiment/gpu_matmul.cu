// tuned_matmul_bench.cu — CUDA/cuBLAS counterpart to tuned_matmul_bench.py.
// Same sizes, same idea: measure achieved GFLOPS (TF32 tensor-core path AND
// plain fp32 CUDA-core path, separately) plus H2D/D2H bandwidth, so both
// sides can be normalized to "% of peak" using real vendor numbers (see the
// notes at the bottom of tuned_matmul_bench.py -- same methodology applies
// here, just look up the GPU-side peak instead of the Blackhole-side one).
//
// This script does NOT hardcode your GPU's peak TFLOPS or PCIe generation --
// it only measures what actually happened. Compute % of peak yourself with
// the numbers from your GPU's datasheet / `nvidia-smi -q` / `lspci -vv`.
//
// Build:
//   nvcc -O2 -arch=sm_80 -o tuned_matmul_bench tuned_matmul_bench.cu -lcublas
//   (adjust -arch to your GPU; TF32 tensor cores need sm_80+/Ampere or newer)
//
// Usage:
//   ./tuned_matmul_bench 1024,2048,4096,10240,20480 results.csv
//   ./tuned_matmul_bench sizes.txt results.csv
//     (sizes.txt: one size per line, '#' comments/blank lines allowed --
//      a file containing "1234\n4567\n7890\n" behaves identically to
//      passing the comma list "1234,4567,7890")

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                   \
  do {                                                                     \
    cudaError_t err__ = (expr);                                           \
    if (err__ != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                        \
      exit(1);                                                            \
    }                                                                      \
  } while (0)

#define CUBLAS_CHECK(expr)                                                 \
  do {                                                                     \
    cublasStatus_t st__ = (expr);                                         \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__, __FILE__,  \
              __LINE__);                                                  \
      exit(1);                                                            \
    }                                                                      \
  } while (0)

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Row-major C(MxN) = A(MxK) * B(KxN) via the standard cuBLAS column-major
// transpose trick (same as cuda_tf32_reference.cu).
static double timed_sgemm(cublasHandle_t handle, int M, int K, int N, const float* A_d,
                           const float* B_d, float* C_d, cublasComputeType_t compute_type,
                           int warmups) {
  const float alpha = 1.0f, beta = 0.0f;
  auto run = [&]() {
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_d, CUDA_R_32F,
                               N, A_d, CUDA_R_32F, K, &beta, C_d, CUDA_R_32F, N, compute_type,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  for (int i = 0; i < warmups; i++) {
    run();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  double t0 = now_ms();
  run();
  CUDA_CHECK(cudaDeviceSynchronize());
  double t1 = now_ms();
  return t1 - t0;
}

struct Row {
  int size;
  std::string mode;
  double warm_ms;
  double achieved_gflops;
  double h2d_gbps;
  double d2h_gbps;
};

// Writes one row immediately and flushes -- called right after each
// measurement instead of accumulating into a vector that only gets written
// out once the whole sweep finishes. A run that dies partway (OOM-killed,
// SLURM time limit, etc.) still leaves every row measured before the crash
// on disk. flush() is enough to survive the process being killed (the OS page
// cache holds the bytes independent of the process); it would not survive a
// hard power loss, which would need fsync too.
static void write_row(std::ofstream& f, const Row& r) {
  f << r.size << "," << r.mode << "," << r.warm_ms << "," << r.achieved_gflops << ","
    << r.h2d_gbps << "," << r.d2h_gbps << "\n";
  f.flush();
}

static void bench_size(int size, std::ofstream& csv) {
  int M = size, K = size, N = size;
  size_t a_n = (size_t)M * K, b_n = (size_t)K * N, c_n = (size_t)M * N;

  std::vector<float> a_h(a_n), b_h(b_n);
  srand(0);
  for (auto& v : a_h) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b_h) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;

  float *a_d, *b_d, *c_d;
  CUDA_CHECK(cudaMalloc(&a_d, a_n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, b_n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c_d, c_n * sizeof(float)));

  // ---- H2D bandwidth ----
  CUDA_CHECK(cudaMemcpy(a_d, a_h.data(), a_n * sizeof(float), cudaMemcpyHostToDevice));  // warm up
  CUDA_CHECK(cudaDeviceSynchronize());
  double t0 = now_ms();
  CUDA_CHECK(cudaMemcpy(a_d, a_h.data(), a_n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b_h.data(), b_n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  double t1 = now_ms();
  double h2d_bytes = (double)(a_n + b_n) * sizeof(float);
  double h2d_gbps = h2d_bytes / ((t1 - t0) / 1000.0) / 1e9;

  // Host input buffers aren't needed again after this point -- drop them
  // before allocating c_h below, rather than holding 2 extra N*N host copies
  // for the rest of bench_size. Mirrors the equivalent fix in
  // tuned_matmul_bench.py's use of torch.from_numpy over torch.tensor().
  std::vector<float>().swap(a_h);
  std::vector<float>().swap(b_h);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  double flops = 2.0 * (double)M * K * N;

  // ---- D2H bandwidth -- measured FIRST (moved up from the end of this
  // function), using one throwaway TF32 GEMM just to populate c_d, so
  // h2d_gbps/d2h_gbps are both already known before either real timed row is
  // written below. ----
  const float warmup_alpha = 1.0f, warmup_beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &warmup_alpha, b_d, CUDA_R_32F,
                             N, a_d, CUDA_R_32F, K, &warmup_beta, c_d, CUDA_R_32F, N,
                             CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> c_h(c_n);
  CUDA_CHECK(cudaMemcpy(c_h.data(), c_d, c_n * sizeof(float), cudaMemcpyDeviceToHost));  // warm up
  CUDA_CHECK(cudaDeviceSynchronize());
  t0 = now_ms();
  CUDA_CHECK(cudaMemcpy(c_h.data(), c_d, c_n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  t1 = now_ms();
  double d2h_bytes = (double)c_n * sizeof(float);
  double d2h_gbps = d2h_bytes / ((t1 - t0) / 1000.0) / 1e9;
  std::vector<float>().swap(c_h);  // also not needed again
  printf("  size=%6d H2D=%7.2f GB/s   D2H=%7.2f GB/s\n", size, h2d_gbps, d2h_gbps);

  double ms_tf32 = timed_sgemm(handle, M, K, N, a_d, b_d, c_d, CUBLAS_COMPUTE_32F_FAST_TF32, 2);
  double gflops_tf32 = flops / (ms_tf32 / 1000.0) / 1e9;
  write_row(csv, {size, "tf32_tensorcore", ms_tf32, gflops_tf32, h2d_gbps, d2h_gbps});
  printf("  size=%6d mode=tf32_tensorcore  warm=%9.3fms  %9.2f GFLOPS\n", size, ms_tf32, gflops_tf32);

  double ms_fp32 = timed_sgemm(handle, M, K, N, a_d, b_d, c_d, CUBLAS_COMPUTE_32F, 2);
  double gflops_fp32 = flops / (ms_fp32 / 1000.0) / 1e9;
  write_row(csv, {size, "plain_fp32", ms_fp32, gflops_fp32, h2d_gbps, d2h_gbps});
  printf("  size=%6d mode=plain_fp32       warm=%9.3fms  %9.2f GFLOPS\n", size, ms_fp32, gflops_fp32);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(a_d));
  CUDA_CHECK(cudaFree(b_d));
  CUDA_CHECK(cudaFree(c_d));
}

// Trim leading/trailing whitespace (including '\r' from files edited on Windows).
static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

static std::vector<int> parse_sizes_csv(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (!item.empty()) out.push_back(std::stoi(item));
  }
  return out;
}

// One size per line. Blank lines and lines starting with '#' are skipped. e.g.:
//   1234
//   4567
//   7890
// is equivalent to passing the comma list "1234,4567,7890".
static std::vector<int> parse_sizes_file(const std::string& path) {
  std::vector<int> out;
  std::ifstream f(path);
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    lineno++;
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    try {
      out.push_back(std::stoi(line));
    } catch (const std::exception&) {
      fprintf(stderr, "[Error] %s:%d: '%s' is not an integer size.\n", path.c_str(), lineno, line.c_str());
      exit(1);
    }
  }
  if (out.empty()) {
    fprintf(stderr, "[Error] %s contained no sizes.\n", path.c_str());
    exit(1);
  }
  return out;
}

// argv[1] can be EITHER a comma-separated list (e.g. "1024,2048,4096") OR a path
// to an existing file with one size per line -- auto-detected by trying to open
// it as a file first.
static std::vector<int> parse_sizes_arg(const std::string& arg) {
  std::ifstream test(arg);
  if (test.good()) {
    test.close();
    return parse_sizes_file(arg);
  }
  return parse_sizes_csv(arg);
}

int main(int argc, char** argv) {
  std::string sizes_arg = argc > 1 ? argv[1] : "1024,2048,4096,10240,20480";
  std::string csv_path = argc > 2 ? argv[2] : "tuned_matmul_cuda.csv";

  std::vector<int> sizes = parse_sizes_arg(sizes_arg);

  // Open once, write the header immediately, and hand the stream down so each
  // row lands on disk (flushed) as soon as it's measured -- see write_row().
  std::ofstream csv(csv_path);
  if (!csv) {
    fprintf(stderr, "[Error] Could not open %s for writing.\n", csv_path.c_str());
    return 1;
  }
  csv << "size,mode,warm_ms,achieved_gflops,h2d_gbps,d2h_gbps\n";
  csv.flush();

  int completed = 0;
  for (int size : sizes) {
    printf("\n=== size=%d ===\n", size);
    bench_size(size, csv);
    completed++;
  }

  printf("\nSaved rows for %d/%zu size(s) to %s\n", completed, sizes.size(), csv_path.c_str());
  return 0;
}