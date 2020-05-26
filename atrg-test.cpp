#include <iostream>
#include <tuple>
#include <random>
#include <chrono>

#define ARMA_DONT_USE_WRAPPER
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

    for(uint i = 0; i < tensor.get_size(); ++i) {
        tensor(i) = distribution(generator);
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
 * ATRG::svd:    second best memory footprint (~2x std variants), as fast as dc variants
 * svd dc:       second fastest variant, memory footprint twice as large as ATRG::svd
 * svd std:      not parallelized -> very slow, lowest memory footprint
 * svd_econ dc:  fastest variant, same memory footprint as svd dc
 * svd_econ std: same as svd std, not parellelized
 */
void test_svds(ATRG::Tensor<double> &tensor) {
    arma::mat flatk;
    tensor.flatten(0, flatk);

    arma::mat U;
    arma::mat V;
    arma::vec S;

    std::cout << "  compare SVD's:" << std::endl;
    std::cout << "    ATRG::svd:" << std::endl;
    auto starttime_svd = std::chrono::high_resolution_clock::now();

    std::cout << "    relative truncation error: " << ATRG::svd(flatk, U, V, S) << std::endl;

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "    residual: " << ATRG::residual_svd(flatk, S, U, V) << std::endl;;
    std::cout << S << std::endl << " -----" << std::endl;
    //std::cout << U << std::endl << std::endl << V << std::endl;



    std::cout << "    arma::svd 'dc':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd(U, S, V, flatk, "dc");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S<< std::endl << " -----" << std::endl;



    std::cout << "    arma::svd 'std':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd(U, S, V, flatk, "std");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S << std::endl << " -----" << std::endl;
    //std::cout << U << std::endl << std::endl << V << std::endl;



    std::cout << "    arma::svd_econ 'dc':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd_econ(U, S, V, flatk, "both", "dc");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S << std::endl << " -----" << std::endl;



    std::cout << "    arma::svd_econ 'std':" << std::endl;
    starttime_svd = std::chrono::high_resolution_clock::now();

    arma::svd_econ(U, S, V, flatk, "both", "std");

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << S << std::endl << " -----" << std::endl;
}



int main(int argc, char **argv) {

    // get current time
    auto starttime = std::chrono::high_resolution_clock::now();
    //==============================================================================================

    std::random_device r;
    auto seed = r();
    std::mt19937_64 generator(static_cast<unsigned long>(seed));

    ATRG::Tensor<double> tensor({10, 10, 10, 10});
    random_Tensor(tensor, generator);
    //example_Tensor(tensor);


    std::cout << "  generated random tensor..." << std::endl;

    //=============================================================================================

    //test_svds(tensor);

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
