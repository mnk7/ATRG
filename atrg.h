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
            throw 0;
        }

        for(uint i = 0; i < tensor.get_order() / 2; ++i) {
            if(tensor.get_dimensions(i) != tensor.get_dimensions(i + tensor.get_order() / 2)) {
                std::cerr << "  In ATRG::compute_logZ: dimensions of forward and backward mode don't match!" << std::endl;
                throw 0;
            }
        }

        uint physical_dimension = tensor.get_order() / 2;

        // make lists of indices from 0 to >physical dimension< and from >physical dimension< to tensor order
        std::vector<uint> null_to_dim_indices(physical_dimension);
        std::iota(null_to_dim_indices.begin(), null_to_dim_indices.end(), 0);

        std::vector<uint> dim_to_order_indices(physical_dimension);
        std::iota(dim_to_order_indices.begin(), dim_to_order_indices.end(), physical_dimension);

        arma::Mat<T> flat;
        tensor.flatten(null_to_dim_indices, dim_to_order_indices, flat);
        arma::Mat<T> U;
        arma::Mat<T> V;
        arma::Col<T> S;
        error = 1.0 - svd(flat, U, V, S, D_truncated);

        if(compute_residual_error) {
            residual_error = 1.0 - residual_svd(flat, S, U, V);
        }


        auto dimensions = tensor.get_dimensions();
        decltype(dimensions) forward_dimensions(dimensions.begin(), dimensions.begin() + physical_dimension);
        decltype(dimensions) backward_dimensions(dimensions.begin() + physical_dimension, dimensions.end());

        auto forward_dimensions_and_alpha(forward_dimensions);
        forward_dimensions_and_alpha.push_back(D_truncated);

        auto backward_dimensions_and_alpha(backward_dimensions);
        backward_dimensions_and_alpha.push_back(D_truncated);

        // we don't need the tensor from here on, so we free the memory
        tensor.reshape({0});

        // create A, B, C, D tensors from our SVD results:
        Tensor<T> A(forward_dimensions_and_alpha);
        Tensor<T> B(backward_dimensions_and_alpha);
        Tensor<T> C(forward_dimensions_and_alpha);
        Tensor<T> D(backward_dimensions_and_alpha);

        arma::Mat<T> US = U_times_S(U, S);
        arma::Mat<T> SVp = U_times_S(V, S);

        // the columns only have alpha as an index, meaning the last index of the tensor
        A.inflate(null_to_dim_indices, {A.get_order() - 1}, U);
        B.inflate(null_to_dim_indices, {B.get_order() - 1}, SVp);
        C.inflate(null_to_dim_indices, {C.get_order() - 1}, US);
        D.inflate(null_to_dim_indices, {D.get_order() - 1}, V);


        return {std::log(Z), error, residual_error};
    }

}

#endif // ATRG
