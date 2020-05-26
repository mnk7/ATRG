#ifndef ATRG
#define ATRG

#include <vector>
#include <tuple>

#include <armadillo>

#include <tensor.h>
#include <svd.h>

namespace ATRG {

    /**
     * compute the log value of the partition sum for the given tensor and a lattice of given dimensions
     * returns log(Z) and an error estimate
     */
    template <typename T>
    inline std::tuple<T, T, T> compute_logZ(Tensor<T> &tensor, const std::vector<uint> lattice_dimensions, const uint D_truncated, const bool compute_residual_error) {
        T Z = 0;
        T error = 0;
        T residual_error = 0;

        if(tensor.get_order() != lattice_dimensions.size() * 2) {
            std::cerr << "  In ATRG::compute_logZ: order of tensor and lattice dimensions don't match!" << std::endl;
            exit(0);
        }

        arma::Mat<T> flat;
        tensor.flatten({0, 1}, flat);
        arma::Mat<T> U;
        arma::Mat<T> V;
        arma::Col<T> S;
        error = 1.0 - svd(flat, U, V, S, D_truncated);

        if(compute_residual_error) {
            residual_error = 1 - residual_svd(flat, S, U, V);
        }


        auto dimensions = tensor.get_dimensions();
        decltype(dimensions) forward_dimensions(dimensions.begin(), dimensions.begin() + tensor.get_order() / 2);
        decltype(dimensions) backward_dimensions(dimensions.begin() + tensor.get_order() / 2, dimensions.end());

        auto forward_dimensions_and_alpha(forward_dimensions);
        forward_dimensions_and_alpha.push_back(D_truncated);

        auto backward_dimensions_and_alpha(backward_dimensions);
        backward_dimensions_and_alpha.push_back(D_truncated);

        // create A, B, C, D tensors from our SVD results:
        Tensor<T> A(forward_dimensions_and_alpha);
        Tensor<T> B(backward_dimensions_and_alpha);
        Tensor<T> C(forward_dimensions_and_alpha);
        Tensor<T> D(backward_dimensions_and_alpha);

        arma::Mat<T> US = U_times_S(U, S);
        arma::Mat<T> SVp = U_times_S(V, S);

        A.inflate(forward_dimensions, U);
        B.inflate(backward_dimensions, SVp);
        C.inflate(forward_dimensions, US);
        D.inflate(backward_dimensions, V);


        return {std::log(Z), std::log(error), std::log(residual_error)};
    }

}

#endif // ATRG
