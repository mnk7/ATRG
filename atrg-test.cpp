#include <iostream>
#include <tuple>
#include <random>
#include <chrono>

#include <omp.h>
#include <cblas-openblas.h>
#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#include <armadillo>

#include <atrg.h>
#include <sptensor.h>
#include <tensor.h>
#include <svd.h>


/**
 * generates a random tensor
 */
template <class TensorType>
void random_Tensor(TensorType &tensor, std::mt19937_64 &generator) {

    std::normal_distribution<decltype(tensor.max())> distribution(0, 1);
    std::uniform_int_distribution<decltype(tensor.get_size())> element_selector(0, tensor.get_size() - 1);

    for(uint i = 0; i < tensor.get_size() / 10; ++i) {
        tensor.set(element_selector(generator), distribution(generator));
    }
}


/**
 * give a simple tensor with an easy to compute SVD
 */
template <class TensorType>
void example_Tensor(TensorType &tensor) {
    tensor.reshape({4, 5});
    tensor.fill(0);
    tensor({0, 0}) = 1;
    tensor({0, 4}) = 2;
    tensor({1, 2}) = 3;
    tensor({3, 1}) = 2;
}


/**
 * fills a tensor with ascending numbers from 0 to tensor size while leaving a space of 9 null-elements
 */
template <class TensorType>
void ascending_Tensor(TensorType &tensor, double start) {
    decltype(tensor.max()) element = start;
    decltype(tensor.max()) sign = 1;

    for(decltype(tensor.get_size()) i = 0; i < tensor.get_size(); i = i + 10) {
        tensor.set(i, element);

        element *= sign;
        element += sign * 0.05;
        sign *= -1;
    }
}



template <class TensorType>
void Ising_Tensor(TensorType &tensor, TensorType &impurity, double T) {
    double beta = 1.0 / T;
    double c = sqrt(cosh(beta));
    double s = sqrt(sinh(beta));

    arma::mat W(2, 2);
    W(0, 0) = c;
    W(1, 0) = c;
    W(0, 1) = s;
    W(1, 1) = -s;

    double der[2] = {tanh(beta), 1.0 / tanh(beta)};

    tensor.reshape({2, 2, 2, 2});	// local tensor
    impurity.reshape({2, 2, 2, 2});	// two neigbouring tensors

    long idx = 0;
    for (uint yf = 0; yf <= 1; ++yf)
    for (uint xf = 0; xf <= 1; ++xf)
    for (uint yb = 0; yb <= 1; ++yb)
    for (uint xb = 0; xb <= 1; ++xb) {
        tensor.set(idx, W(0, xf) * W(0, xb) * W(0, yf) * W(0, yb) + W(1, xf) * W(1, xb) * W(1 ,yf) * W(1, yb));
        impurity.set(idx, 0.5 * tensor(idx) * (der[xb] + der[xf] + der[yb] + der[yf]));
        ++idx;
    }
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

    arma::Mat<double> flat;
    tensor.flatten({0}, null_to_dim_indices, flat);

    arma::SpMat<double> flat_sparse(flat);

    arma::mat U;
    arma::mat V;
    arma::vec S;

    std::cout << "  compare SVD's:" << std::endl;
    std::cout << "    ATRG::svd:" << std::endl;
    auto starttime_svd = std::chrono::high_resolution_clock::now();

    std::cout << "    relative truncation error: " << std::sqrt(ATRG::svd(flat_sparse, U, V, S)) << std::endl;

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "    residual: " << ATRG::residual_svd(flat_sparse, U, V, S) << std::endl;
    std::cout << S << std::endl << " -----" << std::endl;
    //std::cout << U << std::endl << std::endl << V << std::endl;



    std::cout << "    redsvd:" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    std::cout << "    relative truncation error: " << std::sqrt(ATRG::redsvd(flat, U, V, S, std::min(flat.n_cols, flat.n_rows))) << std::endl;

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "    residual: " << ATRG::residual_svd(flat, U, V, S) << std::endl;
    std::cout << S << std::endl << " -----" << std::endl;



    std::cout << "    arma::svd 'dc':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd(U, S, V, flat, "dc");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "    residual: " << ATRG::residual_svd(flat, U, V, S) << std::endl;
    std::cout << S << std::endl << " -----" << std::endl;



    std::cout << "    arma::svd 'std':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd(U, S, V, flat, "std");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S << std::endl << " -----" << std::endl;
    //std::cout << U << std::endl << std::endl << V << std::endl;



    std::cout << "    arma::svd_econ 'dc':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd_econ(U, S, V, flat, "both", "dc");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S << std::endl << " -----" << std::endl;



    std::cout << "    arma::svd_econ 'std':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd_econ(U, S, V, flat, "both", "std");

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
template <class TensorType>
void test_flatten_and_inflate(TensorType &tensor) {
    std::cout << "  testing tensor flatten and inflate:" << std::endl;

    uint physical_dimension = tensor.get_order() / 2;

    // make lists of indices from 0 to >physical dimension< and from >physical dimension< to tensor order
    std::vector<uint> null_to_dim_indices(physical_dimension);
    std::iota(null_to_dim_indices.begin(), null_to_dim_indices.end(), 0);

    std::vector<uint> dim_to_order_indices(physical_dimension);
    std::iota(dim_to_order_indices.begin(), dim_to_order_indices.end(), physical_dimension);

    decltype(tensor.data_copy()) savepoint;
    tensor.flatten(null_to_dim_indices, dim_to_order_indices, savepoint);

    decltype(tensor.data_copy()) flat = savepoint;
    tensor.inflate(null_to_dim_indices, dim_to_order_indices, flat);

    decltype(tensor.data_copy()) endpoint;
    tensor.flatten(null_to_dim_indices, dim_to_order_indices, endpoint);

    std::cout << "    f-b flatten and inflate deviation: " << arma::norm(endpoint - savepoint, "fro") << std::endl << std::endl;


    std::cout << "    single index row and other indices sorted flattening:" << std::endl;
    std::vector<uint> one_to_order_indices(tensor.get_order() - 1);
    std::iota(one_to_order_indices.begin(), one_to_order_indices.end(), 1);
    tensor.flatten({0}, one_to_order_indices, savepoint);
    flat = savepoint;
    tensor.inflate({0}, one_to_order_indices, flat);
    tensor.flatten({0}, one_to_order_indices, endpoint);

    std::cout << "      deviation: " << arma::norm(endpoint - savepoint, "fro") << std::endl << std::endl;

    std::cout << "    single index column and other indices sorted flattening:" << std::endl;
    tensor.flatten(one_to_order_indices, {0}, savepoint);
    flat = savepoint;
    tensor.inflate(one_to_order_indices, {0}, flat);
    tensor.flatten(one_to_order_indices, {0}, endpoint);

    std::cout << "      deviation: " << arma::norm(endpoint - savepoint, "fro") << std::endl << std::endl;



    std::cout << "    naive flattening:" << std::endl;
    // testing naive flattening and naive inflating:
    TensorType simple_tensor({2, 3, 4, 2});
    for(uint i = 0; i < simple_tensor.get_size(); ++i) {
        simple_tensor.set(i, i + 1);
    }

    simple_tensor.print();

    decltype(simple_tensor.data_copy()) flat_int;

    simple_tensor.flatten({1, 0, 3}, {2}, flat_int);

    flat_int.print();

    simple_tensor.inflate({1, 0, 3}, {2}, flat_int);

    int difference = 0;
    for(uint i = 0; i < simple_tensor.get_size(); ++i) {
        difference += simple_tensor(i) - (i + 1);
    }

    std::cout << "    difference after inflation: " << difference << std::endl;



    std::cout << "    test reorder:" << std::endl;

    simple_tensor.reorder({2, 3, 0, 1});
    simple_tensor.reorder({2, 3, 0, 1});

    simple_tensor.print();

    difference = 0;
    for(uint i = 0; i < simple_tensor.get_size(); ++i) {
        difference += simple_tensor(i) - (i + 1);
    }

    std::cout << "    difference after reordering: " << difference << std::endl;
}



template <class TensorType>
void performance_test_flatten_inflate(std::mt19937_64 &generator) {
    std::cout << "  testing flattening and inflating performance:" << std::endl;


    TensorType tensor({20, 20, 20, 20, 20, 20});
    random_Tensor(tensor, generator);

    decltype(tensor.data_copy()) flat;


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

    omp_set_num_threads(1);
    openblas_set_num_threads(1);

    std::random_device r;
    auto seed = r();
    std::mt19937_64 generator(static_cast<unsigned long>(seed));

    /**ATRG::Tensor<double> tensor({12, 8, 10, 12, 8, 10});
    //random_Tensor(tensor, generator);
    //example_Tensor(tensor);
    ascending_Tensor(tensor, 0.1);

    ATRG::Tensor<double> impurity({12, 8, 10, 12, 8, 10});
    ascending_Tensor(impurity, 0.1);

    //ATRG::Tensor<double> tensor_dense({10, 5, 10, 5});
    //random_Tensor(tensor_dense, generator);


    std::cout << "  generated tensor..." << std::endl;

    //=============================================================================================

    //test_svds(tensor);
    //test_flatten_and_inflate(tensor);
    //performance_test_flatten_inflate<ATRG::SpTensor<double>>(generator);
    //performance_test_flatten_inflate<ATRG::Tensor<double>>(generator);

    //=============================================================================================

    //auto [logZ, error_logZ, residual_error_logZ] = ATRG::compute_logZ(tensor, {10, 10}, 10, true, ATRG::t_blocking);
    auto [logZ, error_logZ, residual_error_logZ] = ATRG::compute_single_impurity(tensor, impurity, {3, 3, 3}, 8, true, ATRG::t_blocking);
**/
    /**
     * C++11 version:
     *
     * double logZ, error_logZ, residual_error_logZ;
     * std::tie(logZ, error_logZ, residual_error_logZ) = ATRG::compute_logZ(tensor, {4, 4}, 10, true);
     */

/**    std::cout << "      logZ:            " << logZ << std::endl;
    std::cout << "      relative error:  " << error_logZ << std::endl;
    std::cout << "      residual error:  " << residual_error_logZ << std::endl;**/


    // Ising sweep:
    std::ofstream sweep_file;
    sweep_file.open("Ising_sweeps/Ising_sweep.dat", std::ofstream::out | std::ofstream::trunc);

    uint D = 10;
    std::vector<uint> blockings = {1, 1};
    double delta = 9e-3;

    for(double T = 0.1; T <= 4.05; T += (T < 2 || T > 2.6) ? 0.1 : 0.02) {
        std::cout << "computing at T = " << T << std::endl;

        ATRG::Tensor<double> tensor;
        ATRG::Tensor<double> impurity;

        Ising_Tensor(tensor, impurity, T);
        auto [E, error_E, residual_error_E, logZ] = ATRG::compute_single_impurity(tensor, impurity, blockings, D, true, ATRG::t_blocking);

        Ising_Tensor(tensor, impurity, 1.0 / (1.0 / T + delta));
        auto [logZ_p, error_logZ_p, residual_error_p] = ATRG::compute_logZ(tensor, blockings, D, true, ATRG::t_blocking);

        Ising_Tensor(tensor, impurity, 1.0 / (1.0 / T - delta));
        auto [logZ_m, error_logZ_m, residual_error_m] = ATRG::compute_logZ(tensor, blockings, D, true, ATRG::t_blocking);

        auto E_fd = (logZ_p - logZ_m) / (2 * delta);
        auto susz_fd = 1.0 / (T * T) * (logZ_m - 2.0 * logZ + logZ_p) / (delta * delta);

        sweep_file << T << " " << -E << " " << -E_fd << " " << susz_fd << std::endl;
    }

    sweep_file.flush();
    sweep_file.close();

    //==============================================================================================

    std::cout << std::endl << "\033[1;33mRUNTIME:\033[0m " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime)
                 .count() / 1e3
              << " seconds" << std::endl;

    return 0;
}
