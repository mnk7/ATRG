#include <iostream>
#include <random>
#include <chrono>

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <omp.h>

#include <atrg.h>
#include <tensor.h>


/**
 * generates a random tensor
 */
template <typename T>
void random_Tensor(ATRG::Tensor<T> &tensor, std::mt19937_64 &generator) {

    std::normal_distribution<T> distribution(0, 1);

    for(uint i = 0; i < tensor.size(); ++i) {
        tensor(i) = distribution(generator);
    }
}



/**
 * Compute the resiudal of a SVD decomposition
 */
template <typename T>
T residual_svd(const arma::Mat<T> &matrix, const arma::Col<T> &S, const arma::Mat<T> &U, const arma::Mat<T> &V) {
    arma::Mat<T> matrix_svd(U.n_rows, S.n_elem, arma::fill::zeros);

    for(uint j = 0; j < U.n_rows; ++j) {
        // multiplicate element-wise
        matrix_svd.row(j) = U.row(j) % S.t();
    }

    matrix_svd *= V.t();

    return arma::norm(matrix - matrix_svd, "fro") / arma::norm(matrix, "fro");
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

    ATRG::svd(flatk, U, V, S);

    std::cout << "    Runtime: " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime_svd)
                 .count() / 1e3
              << " seconds" << std::endl;

    std::cout << "    residual: " << residual_svd(flatk, S, U, V) << std::endl;;
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

    ATRG::Tensor<double> tensor({10, 5});
    random_Tensor(tensor, generator);

    std::cout << "  generated random tensor..." << std::endl;

    //=============================================================================================

    test_svds(tensor);


    //==============================================================================================

    std::cout << std::endl << "\033[1;33mRUNTIME:\033[0m " <<
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starttime)
                 .count() / 1e3
              << " seconds" << std::endl;

    return 0;
}
