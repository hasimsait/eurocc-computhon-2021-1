#include <cstring>
int main()
{
    int n, *xadj, *adj;
    float *jaccard_values;

    //////BEGIN CALCULATION CODE
    bool *uv_union = new bool[n]; //instead of unordered set, keep an array of size n
    for (int u = 0; u < n; u++){
        memset(uv_union, false, n * sizeof(bool)); //just to be safe
        for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++){
            uv_union[adj[v_ptr]] = true;
            //set every neighbour of u to 1.
        }
        for (int v_ptr = xadj[u]; v_ptr < xadj[u + 1]; v_ptr++){
            //for every neighbour v of u
            if (adj[v_ptr] > u) {
            //do not waste time with 3-1, 1-3 calculates that.
                int num_intersections = 0;
                int symetric_v_ptr=0;
                for (int i = xadj[v_ptr]; i < xadj[v_ptr + 1]; v_ptr++){
                    //for every neighbour i of v
                    if (uv_union[adj[i]])
                        num_intersections++;
                    else if (adj[i]==u)
                        symetric_v_ptr = i;
                }
                int card_u = xadj[u + 1] - xadj[u];         //can be -+1 not sure.
                int card_v = xadj[v_ptr + 1] - xadj[v_ptr]; //can be -+1 not sure.
                jaccard_values[v_ptr] = float(num_intersections) / float(card_u + card_v);
                jaccard_values[symetric_v_ptr] = float(num_intersections) / float(card_u + card_v);
            }
        }
    }
    //////END CALCULATION CODE
    return 0;
}
