
#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
template <typename vectType>
void print_vect(vectType *vector, size_t dim)
{
    std::cout<<"<";
    for(int i = 0;i<dim-1;i++)
        std::cout<<vector[i]<<",";
    std::cout<<vector[dim-1]<<">";
    std::cout<<std::endl;
}


template <typename vectType>
vectType calulcate_L2(std::vector<vectType> &v1, std::vector<vectType> &v2)
{
    vectType sum = 0;
    for(int i=0;i<v1.size();i++)
        sum+= ((v1[i]-v2[i])*(v1[i]-v2[i]));
    return sqrt(sum);
}

template <typename vectType>
vectType dot_product(std::vector<vectType> &v1, std::vector<vectType> &v2)
{
    vectType sum = 0;
    for(int i=0;i<v1.size();i++)
        sum+= (v1[i]*v2[i]);
    return sum;
}

template<typename vectType>
std::vector<size_t> argpartition(const std::vector<vectType>& data, size_t k) {
    size_t n = data.size();

    // Create a vector of indices: [0, 1, 2, ..., n-1]
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Partially sort indices based on the values in data
    nth_element(indices.begin(), indices.begin() + k, indices.end(),
                [&](size_t i, size_t j) { return data[i] < data[j]; });

    // Return the first k indices
    return std::vector<size_t>(indices.begin(), indices.begin() + k);
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////

///////////  MACROS  ///////////////////////////////

#define X(N, func)                                                                                 \
    case (N):                                                                                      \
        __ret_dist_func = func<(N)>;                                                               \
        break;

// Macro for 4 cases. Used to collapse the switch statement. For a given N, expands to 4 X macros
// of 4N, 4N+1, 4N+2, 4N+3.
#define C4(X, func, N) X(4 * N, func) X(4 * N + 1, func) X(4 * N + 2, func) X(4 * N + 3, func)

// Macros for 8, 16 and 32 cases. Used to collapse the switch statement. Expands into 0-31, 0-15 or
// 0-7 cases.
#define CASES32(X, func)                                                                           \
    C4(X, func, 0)                                                                                 \
    C4(X, func, 1)                                                                                 \
    C4(X, func, 2) C4(X, func, 3) C4(X, func, 4) C4(X, func, 5) C4(X, func, 6) C4(X, func, 7)
#define CASES16(X, func) C4(X, func, 0) C4(X, func, 1) C4(X, func, 2) C4(X, func, 3)
#define CASES8(X, func)  C4(X, func, 0) C4(X, func, 1)

// Main macro. Expands into a switch statement that chooses the implementation based on the
// dimension's remainder.
// @params:
// out:     The output variable that will be set to the chosen implementation.
// dim:     The dimension.
// chunk:   The chunk size. Can be 32, 16 or 8. 32 for 16-bit elements, 16 for 32-bit elements, 8
// for 64-bit elements. func:    The templated function that we want to choose the implementation
// for.
#define CHOOSE_IMPLEMENTATION(out, dim, chunk, func)                                               \
    do {                                                                                           \
        decltype(out) __ret_dist_func;                                                             \
        switch ((dim) % (chunk)) { CASES##chunk(X, func) }                                         \
        out = __ret_dist_func;                                                                     \
    } while (0)