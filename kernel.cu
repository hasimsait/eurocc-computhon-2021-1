#include <cstring>
#define THREADS 64

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

__global__ void jaccard(int *xadj, int *adj, int *nov, float *jaccard_values, int chunk_start){
    // it probably is a better idea to do it per edge instead of per node.
    // or put 1 union in shared memory and do one block per node.
    int u= blockDim.x*blockIdx.x+threadIdx.x+chunk_start;
    if (u<*nov){
        bool *uv_union = new bool[n];
        // instead of unordered set, keep an array of size n
        // it will not fit in thread private memory. I know it from the 406 project.
        memset(uv_union, false, n * sizeof(bool)); //just to be safe
        for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++){
            uv_union[adj[v_ptr]] = true;
            //set every neighbour of u to 1.
        }
        for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++){
            //for every neighbour v of u            
            int num_intersections = 0;
            int symetric_v_ptr=0;
            for (int i = xadj[v_ptr]; i < xadj[v_ptr + 1]; v_ptr++){
                //for every neighbour i of v
                if (uv_union[adj[i]])
                    num_intersections++;
            }
            int card_u = xadj[u + 1] - xadj[u];         //can be -+1 not sure.
            int card_v = xadj[v_ptr + 1] - xadj[v_ptr]; //can be -+1 not sure.
            jaccard_values[v_ptr] = float(num_intersections) / float(card_u + card_v);
        }
    }
    return;
}

void wrapper(int n,int nnz, int* xadj, int* adj, float* jaccard_values)
{
    int *d_xadj, *d_adj, *d_nov;
    float* d_jaccard_values;
    int *devices_count;
    cudaGetDeviceCount(devices_count);
    
    int chunk_size=(n+devices_count-1)/devices_count; //the unbalanced graph will hurt.
    for (unsigned int device_id = 0; device_id < devices_count; device_id++){
        // PUT THE DATA
        cudaSetDevice(device_id);
        cudaMalloc((void **)&d_xadj,(n+1)*sizeof(int));
        cudaMalloc((void **)&d_adj,nnz*sizeof(int));
        cudaMalloc((void **)&d_nov,sizeof(int));
        cudaMalloc((void **)&d_jaccard_values,nnz*sizeof(float));
        cudaMemcpy(d_xadj, xadj, (*nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
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
    for (unsigned int device_id = 0; device_id < devices_count; device_id++){
        // DO THE COMPUTATION
        int chunk_start=device_id*chunk_size;
        cudaSetDevice(device_id);
        jaccard<<<(chunk_size+THREADS-1)/THREADS, THREADS>>>(d_xadj, d_adj, d_nov, d_jaccard_values, chunk_start){
    }
    for (unsigned int device_id = 0; device_id < devices_count; device_id++){
        // GET THE VALUES
        cudaSetDevice(device_id);
        gpuErrchk(cudaDeviceSynchronize());
    }
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    for (unsigned int device_id = 0; device_id < devices_count; device_id++){
        cudaSetDevice(device_id);
        cudaMemcpy(jaccard_values, d_jaccard_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);
        //this must be a reduce, not simple copy unfortunately.
    }
    for (int i = 0; i < *nov; i++)
        printf("%d %d\n", i, ct[i]);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU took: %f s\n", elapsedTime / 1000);
    for (unsigned int device_id = 0; device_id < devices_count; device_id++){
        cudaSetDevice(device_id);
        cudaFree(d_xadj);
        cudaFree(d_adj);
        cudaFree(d_nov);
        cudaFree(d_jaccard_values);
    }    
    return;
}
