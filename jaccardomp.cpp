#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

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

  auto start = chrono::steady_clock::now();
  //////BEGIN CALCULATION CODE
#pragma omp parallel
  {
    // instead of unordered set, keep an array of size n
    bool *uv_union = new bool[n];
    // uv_union is thread private.
    memset(uv_union, false, n * sizeof(bool));
#pragma omp for schedule(dynamic)
    // TODO test if schedule(runtime) performs better with 40 threads,
    // results are within margin of error with 8 threads.

    // dynamic scheduling (chunk size changes, which thread is assigned which
    // iters changes etc.) is slower than static on ideal graphs
    // but the graphs are not balanced.
    for (int u = 0; u < n; u++) {
      for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++) {
        uv_union[adj[v_ptr]] = true;
        // set every neighbour of u to true.
      }
      for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++) {
        // for every neighbour v of u
        if (adj[v_ptr] > u) {
          // do not waste time with 3-1, 1-3 calculates that.
          int num_intersections = 0;
          int num_uncommon = 0; // V/U, so we can calculate ||U U V||
          int symetric_v_ptr = 0;
          for (int i = xadj[adj[v_ptr]]; i < xadj[adj[v_ptr] + 1]; i++) {
            // for every neighbour i of v
            if (uv_union[adj[i]]) {
              num_intersections++;
            } else {
              num_uncommon++;
              if (adj[i] == u)
                // find v-u edge
                symetric_v_ptr = i;
            }
          }
          int card_u = xadj[u + 1] - xadj[u];
          jaccard_values[v_ptr] =
              float(num_intersections) / float(card_u + num_uncommon);
          jaccard_values[symetric_v_ptr] =
              float(num_intersections) / float(card_u + num_uncommon);
        }
      }
      for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++) {
        // set every neighbour of u back to false for the next node.
        uv_union[adj[v_ptr]] = false;
      }
    }
  }
  //////END CALCULATION CODE
  auto end = chrono::steady_clock::now();
  auto diff = end - start;

  cout << "Finished calculating the Jaccards in "
       << chrono::duration<double>(diff).count() << " seconds" << endl;

  //print_jaccards(output_file, n, xadj, adj, jaccard_values);
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
