#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

#define THREADS 1024

// Error check
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
// Error check
// Copied from Kamer Kaya's homeworks.

__global__ void jaccard(int *xadj, int *adj, int *nov, float *jaccard_values,
                        int chunk_start) {
  // it probably is a better idea to do it per edge instead of per node.
  int u = blockDim.x * blockIdx.x + threadIdx.x + chunk_start;
  if (u < *nov) {
    for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++) {
      int v = adj[v_ptr]; // v is a neighbor of u
      int num_intersections = 0;

      for (int u_nbr_ptr = xadj[u]; u_nbr_ptr < xadj[u + 1]; u_nbr_ptr++) {
        // Go over all neighbors of u
        int u_nbr = adj[u_nbr_ptr];
        for (int v_nbr_ptr = xadj[v]; v_nbr_ptr < xadj[v + 1]; v_nbr_ptr++) {
          // Go over all neighbors of v
          int v_nbr = adj[v_nbr_ptr];
          if (u_nbr == v_nbr) {
            // Neighbors of u and v match. Increment the intersections
            num_intersections++;
          }
        }
      }
      int num_union = xadj[u+1]-xadj[u] + xadj[v+1] - xadj[v] - num_intersections;
      // ||X U Y|| = ||X|| + ||Y|| - ||X kesisim Y||
      jaccard_values[v_ptr] = float(num_intersections) / float(num_union);
    }
  }
}
void wrapper(int n, int nnz, int *xadj, int *adj, float *jaccard_values) {
  int *d_xadj, *d_adj, *d_nov;
  float *d_jaccard_values;
  int devices_count=1;
  // cudaGetDeviceCount(&devices_count); //truba doesn't like this.
  int max_conn=0;
  for (int i=0; i<n; i++){
    if ((xadj[i+1]-xadj[i])>max_conn)
      max_conn = xadj[i+1]-xadj[i];
  }

  int chunk_size = ceil(((n*max_conn) + (devices_count) - 1) / (devices_count));
  // the unbalanced graph will hurt.
  for (unsigned int device_id = 0; device_id < devices_count; device_id++) {
    // PUT THE DATA
    cudaSetDevice(device_id);
    cudaMalloc((void **)&d_xadj, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_adj, nnz * sizeof(int));
    cudaMalloc((void **)&d_nov, sizeof(int));
    cudaMalloc((void **)&d_jaccard_values, nnz * sizeof(float));
    cudaMemcpy(d_xadj, xadj, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj, adj, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nov, &n, sizeof(int), cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());
#ifdef DEBUG
    std::cout << "malloc copy done" << std::endl;
#endif
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  /* TODO launch n*max_conn kernels where each kernel is assigned to an edge. Each block is 1 node, place union arra
y in shared memory.*/
  for (unsigned int device_id = 0; device_id < devices_count; device_id++) {
    // DO THE COMPUTATION
    int chunk_start = device_id * chunk_size;
    cudaSetDevice(device_id);
    jaccard<<<(chunk_size + THREADS - 1) / THREADS, THREADS>>>(
  d_xadj, d_adj, d_nov, d_jaccard_values, chunk_start);
    }
    for (unsigned int device_id = 0; device_id < devices_count; device_id++) {
      // GET THE VALUES
      cudaSetDevice(device_id);
      gpuErrchk(cudaDeviceSynchronize());
    }
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    for (unsigned int device_id = 0; device_id < devices_count; device_id++) {
      cudaSetDevice(device_id);
      float *jacc_tmp = new float[nnz];
      cudaMemcpy(jacc_tmp, d_jaccard_values, nnz * sizeof(float),
                 cudaMemcpyDeviceToHost);
      for (int i = 0; i < nnz; i++) {
        // number of iterations can be reduced, start from the first edge of the
        // chunk, end with the last edge.
        if (jacc_tmp[i] > 0 && jacc_tmp[i]<1)
          jaccard_values[i] = jacc_tmp[i];
      }
    }
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU took: %f s\n", elapsedTime / 1000);
    for (unsigned int device_id = 0; device_id < devices_count; device_id++) {
      cudaSetDevice(device_id);
      cudaFree(d_xadj);
      cudaFree(d_adj);
      cudaFree(d_nov);
      cudaFree(d_jaccard_values);
    }
    return;
}

struct CSR {
  int _n;
  int _m;
  int *_xadj;
  int *_adj;
  int *_is;
  CSR(int n, int m, int *xadj, int *adj, int *is)
      : _n(n), _m(m), _xadj(xadj), _adj(adj), _is(is) {}
};

CSR create_csr_from_file(string filename);
void print_jaccards(string filename, int n, int *xadj, int *adj, float *jacc);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "Parameters: input_file output_file\n";
    return 1;
  }
  string input_file = argv[1];
  string output_file = argv[2];

  CSR graph =
      create_csr_from_file(input_file); // Creates a graph in the Compressed
                                        // Sparse Row format from the input file
  int n = graph._n, m = graph._m, *xadj = graph._xadj, *adj = graph._adj,
      *is = graph._is;
  cout << "Read the graph into CSR format" << endl;

  float *jaccard_values = new float[m];
  // jaccard_values[j] = Jaccard between vertices (a, b) where adj[j] = b and
  // adj[a] < j < adj[a+1] i.e. the edge located in index j of the adj array
  memset(jaccard_values,0,m*sizeof(float));//-0 +0 stuff happens.
  auto start = chrono::steady_clock::now();
  //////BEGIN CALCULATION CODE
  wrapper(n, m, xadj, adj, jaccard_values);
  //////END CALCULATION CODE
  auto end = chrono::steady_clock::now();
  auto diff = end - start;

  cout << "Finished calculating the Jaccards in "
       << chrono::duration<double>(diff).count() << " seconds" << endl;

  print_jaccards(output_file, n, xadj, adj, jaccard_values);
  cout << "Finished printing the Jaccards" << endl;

  return 0;
}

void print_jaccards(string output_file, int n, int *xadj, int *adj,
                    float *jacc) {
  ofstream fout(output_file);
  // Save flags/precision.
  ios_base::fmtflags oldflags = cout.flags();
  streamsize oldprecision = cout.precision();

  cout << fixed;
  for (int u = 0; u < n; u++) {
    for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++) {
      fout << "(" << u << ", " << adj[v_ptr] << "): " << fixed
           << setprecision(3) << jacc[v_ptr] << endl;
      std::cout.flags(oldflags);
      std::cout.precision(oldprecision);
    }
  }
}

CSR create_csr_from_file(string filename) {
  ifstream fin(filename);
  if (fin.fail()) {
    cout << "Failed to open graph file\n";
    throw -1;
  }
  int n = 0, m = 0, *xadj, *adj, *is;

  fin >> n >> m;
  vector<vector<int>> edge_list(n);
  int u, v;
  int read_edges = 0;
  while (fin >> u >> v) {
    if (u < 0) {
      cout << "Invalid vertex ID - negative ID found: " << u << endl;
      throw -2;
    }
    if (u >= n) {
      cout << "Invalid vertex ID - vertex ID > number of edges found. VID: "
           << u << " and n: " << n << endl;
      throw -2;
    }
    edge_list[u].push_back(v);
    read_edges += 1;
  }
  if (read_edges != m) {
    cout << "The edge list file specifies there are " << m
         << " edges but it contained " << read_edges << "instead" << endl;
    throw -3;
  }

  /////// If CSR is sorted
  for (auto &edges : edge_list) {
    sort(edges.begin(), edges.end());
  }
  ///////
  xadj = new int[n + 1];
  adj = new int[m];
  is = new int[m];
  int counter = 0;
  for (int i = 0; i < n; i++) {
    xadj[i] = counter;
    copy(edge_list[i].begin(), edge_list[i].end(), adj + counter);
    counter += edge_list[i].size();
  }
  xadj[n] = counter;
  for (int i = 0; i < n; i++) {
    for (int j = xadj[i]; j < xadj[i + 1]; j++) {
      is[j] = i;
    }
  }
  CSR graph(n, m, xadj, adj, is);
  return graph;
}
