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

struct element
{
	int start_idx;
	int length;
	int real_idx;

	element(int a, int b, int c)
		:start_idx(a), length(b), real_idx(c)
	{}
};

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

int main() {

	cout << "Parameters: input_file output_file\n";


	string input_file;
	string output_file;

	cin >> input_file;
	cin >> output_file;

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

	/////// SORTING START

	vector<element> sorted; // starting idx, length, old_idx

	for (size_t i = 0; i < n ; i++)
	{
		element sample(xadj[i], xadj[i + 1] - xadj[i], i);
		sorted.push_back(sample);
	}
	auto sort1 = chrono::steady_clock::now();
	sort(sorted.begin(), sorted.end(), [](element const &a, element const &b) { return a.length < b.length; });
	auto sort2 = chrono::steady_clock::now();

	int *new_numbers, *new_xadj, *new_adj, *transform, *edge_transform;
	new_numbers = new int[n];
	new_xadj = new int[n + 1];
	new_adj = new int[m];
	transform = new int[n];
	edge_transform = new int[m];

	new_xadj[0] = 0;

	for (size_t i = 0; i < sorted.size(); i++)
	{
		new_numbers[sorted[i].real_idx] = i;
		transform[i] = sorted[i].real_idx;
		new_xadj[i + 1] = new_xadj[i] + sorted[i].length;
	}

	int idx = 0, adj_idx=0;

	for (size_t i = 0; i < sorted.size(); i++)
	{
		idx = sorted[i].start_idx;
		for (size_t j = 0; j <sorted[i].length; j++)
		{
			new_adj[adj_idx] = new_numbers[adj[idx]];
			edge_transform[adj_idx] = idx;
			adj_idx++;
			idx++;
		}
	}
	auto end_sort = chrono::steady_clock::now();
	////// SORTING END


	bool *uv_union =

		new bool[n]; // instead of unordered set, keep an array of size n
	memset(uv_union, false, n * sizeof(bool)); // just to be safe
	for (int u = 0; u < n; u++) {
		for (int v_ptr = new_xadj[u]; v_ptr < new_xadj[u + 1]; v_ptr++) {
			uv_union[new_adj[v_ptr]] = true;
			// set every neighbour of u to 1.
		}

		for (int v_ptr = new_xadj[u]; v_ptr < new_xadj[u + 1]; v_ptr++) {
			// for every neighbour v of u
			if (new_adj[v_ptr] > u) {
				// do not waste time with 3-1, 1-3 calculates that.
				int num_intersections = 0;
				int num_uncommon = 0; // V/U, so we can calculate ||U U V||
				int symetric_v_ptr = 0;
				for (int i = new_xadj[new_adj[v_ptr]]; i < new_xadj[new_adj[v_ptr] + 1]; i++) {
					// for every neighbour i of v
					if (uv_union[new_adj[i]]) {
						num_intersections++;
					}
					else {
						num_uncommon++;
						if (new_adj[i] == u)
							symetric_v_ptr = i; // find v-u edge
					}
				}
				int card_u = new_xadj[u + 1] - new_xadj[u];
				jaccard_values[v_ptr] = 
					float(num_intersections) / float(card_u + num_uncommon);
				jaccard_values[symetric_v_ptr] =
					float(num_intersections) / float(card_u + num_uncommon);
			}
		}
		for (int v_ptr = new_xadj[u]; v_ptr < new_xadj[u + 1]; v_ptr++) {
			uv_union[new_adj[v_ptr]] = false;
			// clean the array.
		}
	}
	//////END CALCULATION CODE
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "Finished sorting "
		<< chrono::duration<double>(end_sort-start).count() << " seconds" << endl;
	cout << "Real sort "
		<< chrono::duration<double>(sort2 - sort1).count() << " seconds" << endl;
	cout << "Finished calculating the Jaccards in "
		<< chrono::duration<double>(end -end_sort).count() << " seconds" << endl;
	cout << "Finished  all"
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
		throw - 1;
	}
	int n = 0, m = 0, *xadj, *adj, *is;

	fin >> n >> m;
	vector<vector<int>> edge_list(n);
	int u, v;
	int read_edges = 0;
	while (fin >> u >> v) {
		if (u < 0) {
			cout << "Invalid vertex ID - negative ID found: " << u << endl;
			throw - 2;
		}
		if (u >= n) {
			cout << "Invalid vertex ID - vertex ID > number of edges found. VID: "
				<< u << " and n: " << n << endl;
			throw - 2;
		}
		edge_list[u].push_back(v);
		read_edges += 1;
	}
	if (read_edges != m) {
		cout << "The edge list file specifies there are " << m
			<< " edges but it contained " << read_edges << "instead" << endl;
		throw - 3;
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