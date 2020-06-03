#ifndef ATRG
#define ATRG

#include <vector>
#include <string>
#include <tuple>

#include <armadillo>

#include <sptensor.h>
#include <tensor.h>
#include <svd.h>

namespace ATRG {

    enum BlockingMode {
        t_blocking,
        s_blocking
    };



    template <typename T>
    inline void swap_bonds(Tensor<T> &B, Tensor<T> &C, Tensor<T> &X, Tensor<T> &Y, const uint blocking_direction,
                           T &error, T &residual_error, const bool compute_residual_error,
                           std::vector<uint> &forward_indices, const uint D_truncated, std::vector<uint> &forward_dimensions) {
        arma::Mat<T> B_flat;
        arma::Mat<T> C_flat;

        auto not_blocked_indices(forward_indices);
        not_blocked_indices.erase(not_blocked_indices.begin() + blocking_direction);

        // separate the not blocked indices
        B.flatten(not_blocked_indices, {blocking_direction, B.get_order() - 1}, B_flat);
        C.flatten(not_blocked_indices, {blocking_direction, C.get_order() - 1}, C_flat);


        arma::Mat<T> U_B;
        arma::Mat<T> V_B;
        arma::Col<T> S_B;
        error += svd(B_flat, U_B, V_B, S_B);

        uint mu_dimension = V_B.n_cols;

        if(compute_residual_error) {
            auto residual_error_b = residual_svd(B_flat, U_B, V_B, S_B);
            residual_error += residual_error_b * residual_error_b;
        }

        // make a tensor of S*V with indices: x_trunc, alpha, mu
        // forward and backward dimensions are the same!
        Tensor<T> B_S({forward_dimensions[blocking_direction], D_truncated, mu_dimension});
        V_B = U_times_S(V_B, S_B);
        B_S.inflate({0, 1}, {2}, V_B);
        B_S.flatten({0}, {1, 2}, B_flat);


        arma::Mat<T> U_C;
        arma::Mat<T> V_C;
        arma::Col<T> S_C;
        error += svd(C_flat, U_C, V_C, S_C);

        uint nu_dimension = V_C.n_cols;

        if(compute_residual_error) {
            auto residual_error_c = residual_svd(C_flat, U_C, V_C, S_C);
            residual_error += residual_error_c * residual_error_c;
        }

        // make a tensor of S*V with indices: x_trunc, beta, nu
        Tensor<T> C_S({forward_dimensions[blocking_direction], D_truncated, nu_dimension});
        V_C = U_times_S(V_C, S_C);
        C_S.inflate({0, 1}, {2}, V_C);
        C_S.flatten({0}, {1, 2}, C_flat);


        // B.t() * C with indices: alpha, mu, beta, nu
        Tensor<T> M({D_truncated, mu_dimension, D_truncated, nu_dimension});
        B_flat = B_flat.t() * C_flat;
        M.inflate({0, 1}, {2, 3}, B_flat);
        M.flatten({0, 3}, {2, 1}, B_flat);


        arma::Mat<T> U;
        arma::Mat<T> V;
        arma::Col<T> S;
        // B_flat has indices: {alpha, nu} {beta, mu}
        error += svd(B_flat, U, V, S, D_truncated);

        uint truncated_dimension = U.n_cols;

        if(compute_residual_error) {
            auto residual_error_B_t_C = residual_svd(B_flat, U, V, S);
            residual_error += residual_error_B_t_C * residual_error_B_t_C;
        }

        S.for_each([](auto &element) {element = std::sqrt(element);});


        // X has indices: alpha, nu, x_trunc (index from SVD)
        X.reshape({D_truncated, nu_dimension, truncated_dimension});
        U = U_times_S(U, S);
        X.inflate({0, 1}, {2}, U);
        X.flatten({1}, {0, 2}, B_flat);
        B_flat = U_C * B_flat;
        // new indices: all forward indices but x_trunc, alpha, x_trunc
        auto X_dimensions(forward_dimensions);
        X_dimensions.erase(X_dimensions.begin() + blocking_direction);
        X_dimensions.push_back(D_truncated);
        X_dimensions.push_back(U.n_cols);
        X.reshape(X_dimensions);
        // push the indices after x_trunc forward, so that x_trunc can be the last index
        for(decltype(not_blocked_indices.size()) i = blocking_direction; i < not_blocked_indices.size(); ++i) {
            --not_blocked_indices[i];
        }

        X.inflate(not_blocked_indices, {X.get_order() - 2, X.get_order() - 1}, B_flat);


        // Y has indices: beta, mu, x_trunc (index from SVD)
        Y.reshape({D_truncated, mu_dimension, truncated_dimension});
        V = U_times_S(V, S);
        Y.inflate({0, 1}, {2}, V);
        Y.flatten({1}, {0, 2}, B_flat);
        B_flat = U_B * B_flat;
        // new indices: all backward indices but x_trunc, beta, x_trunc
        // -> backward and forward dimensions are the same!
        Y.reshape(X_dimensions);
        Y.inflate(not_blocked_indices, {Y.get_order() - 2, Y.get_order() - 1}, B_flat);
    }



    template <typename T>
    inline void contract_bonds(Tensor<T> &A, Tensor<T> &D, Tensor<T> &X, Tensor<T> &Y, Tensor<T> &G, Tensor<T> &H, const uint blocking_direction,
                               T &error, T &residual_error, const bool compute_residual_error) {
        /*
         * indices A: all forward, alpha
         * indices D: all backward, beta
         * indices X: all forward but x_trunc, alpha, x_trunc
         * indices Y: all backward but x_trunc, beta, x_trunc
         */


    }



    /**
     * compute the log value of the partition sum for the given tensor and a lattice of given dimensions
     * returns log(Z) and an error estimate
     */
    template <typename T>
    inline std::tuple<T, T, T> compute_logZ(SpTensor<T> &tensor, const std::vector<uint> lattice_dimensions, const uint D_truncated,
                                            const bool compute_residual_error, const BlockingMode blocking_mode = t_blocking) {
        std::cout << "  computing log(Z):" << std::endl;

        T Z = 0;
        /*
         * we compute this errors with error propagation:
         * if we multiply to quantities we compute:
         *     error^2 = error1^2 + error2^2
         * and take the square root over everything in the end
         */
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
        std::vector<uint> forward_indices(physical_dimension);
        std::iota(forward_indices.begin(), forward_indices.end(), 0);

        std::vector<uint> backward_indices(physical_dimension);
        std::iota(backward_indices.begin(), backward_indices.end(), physical_dimension);

        //=============================================================================================

        arma::SpMat<T> flat;
        tensor.flatten(forward_indices, backward_indices, flat);
        arma::Mat<T> U;
        arma::Mat<T> V;
        arma::Col<T> S;
        error += svd(flat, U, V, S, D_truncated);

        if(compute_residual_error) {
            residual_error = residual_svd(flat, U, V, S);
            // keep the squared error
            residual_error *= residual_error;
        }

        flat.set_size(0);


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

        arma::Mat<T> US(U_times_S(U, S));
        arma::Mat<T> SVp(U_times_S(V, S));

        // the columns only have alpha as an index, meaning the last index of the tensor
        // B and D hold the backward modes, so their indices are relabeled
        A.inflate(forward_indices, {A.get_order() - 1}, U);
        B.inflate(forward_indices, {B.get_order() - 1}, SVp);
        C.inflate(forward_indices, {C.get_order() - 1}, US);
        D.inflate(forward_indices, {D.get_order() - 1}, V);

        US.set_size(0);
        SVp.set_size(0);

        // intermediate tensors that we will need during the blocking
        Tensor<T> X;
        Tensor<T> Y;
        Tensor<T> G;
        Tensor<T> H;

        std::cout << "    decomposed initial tensor..." << std::endl;

        //=============================================================================================

        // contract the lattice
        uint blocking_direction = physical_dimension - 1;
        if(blocking_mode == s_blocking) {
            blocking_direction = 0;
        }

        std::vector<uint> finished_blockings(lattice_dimensions.size(), 0);
        bool finished = false;

        while(!finished) {
            //=============================================================================================
            // swap the bonds, the not blocked modes between B and C and gain the tensors X and Y:
            swap_bonds(B, C, X, Y, blocking_direction,
                       error, residual_error, compute_residual_error,
                       forward_indices, D_truncated, forward_dimensions);

            // contract the double bonds of A, X in forward and B, D in backward direction
            contract_bonds(A, D, X, Y, G, H, blocking_direction,
                           error, residual_error, compute_residual_error);

            //=============================================================================================

            ++finished_blockings[blocking_direction];

            // decide the next truncation direction:
            if(blocking_mode == s_blocking) {
                if(finished_blockings[blocking_direction] == lattice_dimensions[blocking_direction]) {
                    if(blocking_direction == physical_dimension - 1) {
                        std::cout << "    finished s-blocking..." << std::endl;
                        finished = true;
                    } else {
                        std::cout << "    s-blocked direction " << blocking_direction << "..." << std::endl;
                        // block the next direction
                        ++blocking_direction;
                    }
                }
            } else {
                // t-blocking:
                if(finished_blockings[blocking_direction] == lattice_dimensions[blocking_direction]) {
                    if(blocking_direction == 0) {
                        std::cout << "    finished t-blocking..." << std::endl;
                        finished = true;
                    } else {
                        std::cout << "    t-blocked direction " << blocking_direction << "..." << std::endl;
                        // block the next direction
                        --blocking_direction;
                    }
                }
            }
        }




        return {std::log(Z), std::sqrt(error), std::sqrt(residual_error)};
    }

}

#endif // ATRG
