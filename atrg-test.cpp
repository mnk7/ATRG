#include <iostream>
#include <tuple>
#include <random>
#include <chrono>

#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#include <armadillo>

#include <atrg.h>
#include <tensor.h>
#include <svd.h>


/**
 * generates a random tensor
 */
template <typename T>
void random_Tensor(ATRG::Tensor<T> &tensor, std::mt19937_64 &generator) {

    std::normal_distribution<T> distribution(0, 1);
    std::uniform_int_distribution<decltype(tensor.get_size())> element_selector(0, tensor.get_size() - 1);

    for(uint i = 0; i < tensor.get_size() / 10; ++i) {
        tensor.set(element_selector(generator), distribution(generator));
    }
}


/**
 * give a simple tensor with an easy to compute SVD
 */
template <typename T>
void example_Tensor(ATRG::Tensor<T> &tensor) {
    tensor.reshape({4, 5});
    tensor.fill(0);
    tensor({0, 0}) = 1;
    tensor({0, 4}) = 2;
    tensor({1, 2}) = 3;
    tensor({3, 1}) = 2;
}


/**
 * Compare different SVD methods
 * ATRG::svd:    second best memory footprint (~2x std variants), as fast as dc variants; SEEMS to give worse results!
 * svd dc:       second fastest variant, memory footprint twice as large as ATRG::svd
 * svd std:      not parallelized -> very slow, lowest memory footprint
 * svd_econ dc:  fastest variant, same memory footprint as svd dc
 * svd_econ std: same as svd std, not parellelized
 */
void test_svds(ATRG::Tensor<double> &tensor) {
    std::vector<uint> null_to_dim_indices(tensor.get_order() - 1);
    std::iota(null_to_dim_indices.begin(), null_to_dim_indices.end(), 1);

    arma::SpMat<double> flat;
    tensor.flatten({0}, null_to_dim_indices, flat);

    arma::Mat<double> flat_dense(flat);

    arma::mat U;
    arma::mat V;
    arma::vec S;

    std::cout << "  compare SVD's:" << std::endl;
    std::cout << "    ATRG::svd:" << std::endl;
    auto starttime_svd = std::chrono::high_resolution_clock::now();

    std::cout << "    relative truncation error: " << ATRG::svd(flat, U, V, S) << std::endl;

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "    residual: " << ATRG::residual_svd(flat, S, U, V) << std::endl;
    std::cout << S << std::endl << " -----" << std::endl;
    //std::cout << U << std::endl << std::endl << V << std::endl;



    std::cout << "    arma::svd 'dc':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd(U, S, V, flat_dense, "dc");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "    residual: " << ATRG::residual_svd(flat_dense, S, U, V) << std::endl;
    std::cout << S << std::endl << " -----" << std::endl;



    std::cout << "    arma::svd 'std':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd(U, S, V, flat_dense, "std");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S << std::endl << " -----" << std::endl;
    //std::cout << U << std::endl << std::endl << V << std::endl;



    std::cout << "    arma::svd_econ 'dc':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd_econ(U, S, V, flat_dense, "both", "dc");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S << std::endl << " -----" << std::endl;



    std::cout << "    arma::svd_econ 'std':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd_econ(U, S, V, flat_dense, "both", "std");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S << std::endl << " -----" << std::endl;
}



/**
 * Testing the flattening and inflating functionalities of the tensor
 */
void test_flatten_and_inflate(ATRG::Tensor<double> &tensor) {
    std::cout << "  testing tensor flatten and inflate:" << std::endl;

    uint physical_dimension = tensor.get_order() / 2;

    // make lists of indices from 0 to >physical dimension< and from >physical dimension< to tensor order
    std::vector<uint> null_to_dim_indices(physical_dimension);
    std::iota(null_to_dim_indices.begin(), null_to_dim_indices.end(), 0);

    std::vector<uint> dim_to_order_indices(physical_dimension);
    std::iota(dim_to_order_indices.begin(), dim_to_order_indices.end(), physical_dimension);

    arma::SpMat<double> savepoint;
    tensor.flatten(null_to_dim_indices, dim_to_order_indices, savepoint);

    arma::SpMat<double> flat = savepoint;
    tensor.inflate(null_to_dim_indices, dim_to_order_indices, flat);

    arma::SpMat<double> endpoint;
    tensor.flatten(null_to_dim_indices, dim_to_order_indices, endpoint);

    std::cout << "    f-b flatten and inflate deviation: " << arma::norm(endpoint - savepoint, "fro") << std::endl << std::endl;



    std::cout << "    naive flattening:" << std::endl;
    // testing naive flattening and naive inflating:
    ATRG::Tensor<int> simple_tensor({2, 3, 4});
    for(uint i = 0; i < simple_tensor.get_size(); ++i) {
        simple_tensor.set(i, i + 1);
    }

    simple_tensor.print();

    arma::SpMat<int> flat_int;

    simple_tensor.flatten({1, 0}, {2}, flat_int);

    flat_int.print_dense();

    simple_tensor.inflate({1, 0}, {2}, flat_int);

    int difference = 0;
    for(uint i = 0; i < simple_tensor.get_size(); ++i) {
        difference += simple_tensor(i) - (i + 1);
    }

    std::cout << "    difference after inflation: " << difference << std::endl;



    std::cout << "    test reorder:" << std::endl;

    simple_tensor.reorder({2, 0, 1});
    simple_tensor.reorder({2, 0, 1});
    simple_tensor.reorder({2, 0, 1});

    simple_tensor.print();

    difference = 0;
    for(uint i = 0; i < simple_tensor.get_size(); ++i) {
        difference += simple_tensor(i) - (i + 1);
    }

    std::cout << "    difference after reordering: " << difference << std::endl;
}



void performance_test_flatten_inflate(std::mt19937_64 &generator) {
    std::cout << "  testing flattening and inflating performance:" << std::endl;


    ATRG::Tensor<double> tensor({20, 20, 20, 20, 20, 20});
    random_Tensor(tensor, generator);

    arma::SpMat<double> flat;


    auto starttime = std::chrono::high_resolution_clock::now();

    tensor.flatten({2, 5}, {4, 0, 3, 1}, flat);

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "      __flat__" << std::endl;


    starttime = std::chrono::high_resolution_clock::now();

    tensor.inflate({2, 5}, {4, 0, 3, 1}, flat);

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "      OOinflatedOO" << std::endl;


    std::cout << "  testing reordering performance:" << std::endl;

    starttime = std::chrono::high_resolution_clock::now();

    tensor.reorder({2, 4, 1, 0, 5, 3});

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime)
                 .count() / 1e3
              << " seconds" << std::endl;
}



int main(int argc, char **argv) {

    // get current time
    auto starttime = std::chrono::high_resolution_clock::now();
    //==============================================================================================

    std::random_device r;
    auto seed = r();
    std::mt19937_64 generator(static_cast<unsigned long>(seed));

    ATRG::Tensor<double> tensor({10, 5, 10, 5});
    random_Tensor(tensor, generator);
    //example_Tensor(tensor);


    std::cout << "  generated random tensor..." << std::endl;

    //=============================================================================================

    //test_svds(tensor);
    //test_flatten_and_inflate(tensor);
    performance_test_flatten_inflate(generator);

    //=============================================================================================

    std::cout << "  computing logZ:" << std::endl;

    // compute log(Z) on a 4x4 lattice
    auto [logZ, error_logZ, residual_error_logZ] = ATRG::compute_logZ(tensor, {4, 4}, 10, true);

    /**
     * C++11 version:
     *
     * double logZ, error_logZ, residual_error_logZ;
     * std::tie(logZ, error_logZ, residual_error_logZ) = ATRG::compute_logZ(tensor, {4, 4}, 10, true);
     */

    std::cout << "    logZ:            " << logZ << std::endl;
    std::cout << "    relative error:  " << error_logZ << std::endl;
    std::cout << "    residual error:  " << residual_error_logZ << std::endl;

    //==============================================================================================

    std::cout << std::endl << "\033[1;33mRUNTIME:\033[0m " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime)
                 .count() / 1e3
              << " seconds" << std::endl;

    return 0;
}
