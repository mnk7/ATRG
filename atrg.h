#ifndef ATRG
#define ATRG

#include <vector>
#include <string>
#include <tuple>
#include <chrono>

#include <armadillo>

#include <sptensor.h>
#include <tensor.h>
#include <svd.h>

#include <sys/resource.h>
double get_usage() {
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);

	return static_cast<double>(usage.ru_maxrss) / 1e6;
}

namespace ATRG {

    enum BlockingMode {
        t_blocking,
        s_blocking
    };



    /**
     * helper function for swap_bonds; takes B_S, C_S as well as two isometries and construncts X and Y
     */
    template <typename T>
    void assemble_new_bonds(ATRG::Tensor<T> &X, ATRG::Tensor<T> &Y, arma::Mat<T> &X_flat, arma::Mat<T> &Y_flat, const arma::Mat<T> &U_B, const arma::Mat<T> &U_C, const uint blocking_direction,
                            T &error, T &residual_error, const bool compute_residual_error,
                            const std::vector<uint> &forward_indices, const std::vector<uint> &not_blocked_indices, std::vector<uint> &forward_dimensions_and_alpha, const uint truncated_dimension) {
        uint mu_dimension = U_B.n_cols;
        uint nu_dimension = U_C.n_cols;


        // X has indices: alpha, nu, x_trunc (index from SVD)
        X.reshape({forward_dimensions_and_alpha.back(), nu_dimension, truncated_dimension});
        X.inflate({0, 1}, {2}, X_flat);
        X.flatten({1}, {0, 2}, X_flat);
        X_flat = U_C * X_flat;

        // new indices: all forward indices, alpha
        auto X_dimensions = forward_dimensions_and_alpha;
        X_dimensions[blocking_direction] = truncated_dimension;
        X.reshape(X_dimensions);
        X.inflate(not_blocked_indices, {X.get_order() - 1, blocking_direction}, X_flat);


        // Y has indices: beta, mu, x_trunc (index from SVD)
        Y.reshape({forward_dimensions_and_alpha.back(), mu_dimension, truncated_dimension});
        Y.inflate({0, 1}, {2}, Y_flat);
        Y.flatten({1}, {0, 2}, Y_flat);
        Y_flat = U_B * Y_flat;
        // new indices: all backward indices, beta
        // -> backward and forward dimensions are the same!
        Y.reshape(X_dimensions);
        Y.inflate(not_blocked_indices, {Y.get_order() - 1, blocking_direction}, Y_flat);
    }



    template <typename T>
    void swap_bonds(ATRG::Tensor<T> &B, ATRG::Tensor<T> &C, ATRG::Tensor<T> &X, ATRG::Tensor<T> &Y, const uint blocking_direction,
                    T &error, T &residual_error, const bool compute_residual_error,
                    const std::vector<uint> &forward_indices, std::vector<uint> &forward_dimensions_and_alpha, const uint D_truncated,
                    arma::Mat<T> &U_B, arma::Mat<T> &U_C, arma::Mat<T> &U_M) {
        arma::Mat<T> B_flat;
        arma::Mat<T> C_flat;

        auto not_blocked_indices(forward_indices);
        not_blocked_indices.erase(not_blocked_indices.begin() + blocking_direction);

        // separate the not blocked indices
        B.flatten(not_blocked_indices, {blocking_direction, B.get_order() - 1}, B_flat);
        C.flatten(not_blocked_indices, {blocking_direction, C.get_order() - 1}, C_flat);

        // use D*D for everything higher than 2D
        auto B_C_truncation = D_truncated;
        if(forward_dimensions_and_alpha.size() > 3) {
            B_C_truncation *= D_truncated;
        }


        arma::Mat<T> V_B;
        arma::Col<T> S_B;
        error += svd(B_flat, U_B, V_B, S_B, B_C_truncation);

        if(compute_residual_error) {
            auto residual_error_b = residual_svd(B_flat, U_B, V_B, S_B);
            residual_error += residual_error_b * residual_error_b;
        }

        uint mu_dimension = U_B.n_cols;

        // make a tensor of S*V with indices: x_trunc, alpha, mu
        // forward and backward dimensions are the same!
        B.reshape({forward_dimensions_and_alpha[blocking_direction], forward_dimensions_and_alpha.back(), mu_dimension});
        B_flat = U_times_S(V_B, S_B);
        B.inflate({0, 1}, {2}, B_flat);
        B.flatten({0}, {1, 2}, B_flat);


        arma::Mat<T> V_C;
        arma::Col<T> S_C;
        error += svd(C_flat, U_C, V_C, S_C, B_C_truncation);

        if(compute_residual_error) {
            auto residual_error_c = residual_svd(C_flat, U_C, V_C, S_C);
            residual_error += residual_error_c * residual_error_c;
        }

        uint nu_dimension = U_C.n_cols;

        // make a tensor of S*V with indices: x_trunc, beta, nu
        C.reshape({forward_dimensions_and_alpha[blocking_direction], forward_dimensions_and_alpha.back(), nu_dimension});
        C_flat = U_times_S(V_C, S_C);
        C.inflate({0, 1}, {2}, C_flat);
        C.flatten({0}, {1, 2}, C_flat);


        X.reshape({forward_dimensions_and_alpha.back(), mu_dimension, forward_dimensions_and_alpha.back(), nu_dimension});

        // B.t() * C with indices: alpha, mu, beta, nu
        B_flat = B_flat.t() * C_flat;
        X.inflate({0, 1}, {2, 3}, B_flat);
        X.flatten({0, 3}, {2, 1}, B_flat);


        arma::Mat<T> V_M;
        arma::Col<T> S;
        // B_flat has indices: {alpha, nu} {beta, mu}
        error += svd(B_flat, U_M, V_M, S, D_truncated);

        uint truncated_dimension = U_M.n_cols;

        if(compute_residual_error) {
            auto residual_error_B_t_C = residual_svd(B_flat, U_M, V_M, S);
            residual_error += residual_error_B_t_C * residual_error_B_t_C;
        }

        B_flat = U_M;
        C_flat = U_times_S(V_M, S);

#ifdef DEBUG
        std::cout << "U_B:" << std::endl << U_B << std::endl;
        std::cout << "U_C:" << std::endl << U_C << std::endl;
        std::cout << "U_M:" << std::endl << U_M << std::endl;
#endif

        assemble_new_bonds(X, Y, B_flat, C_flat, U_B, U_C, blocking_direction,
                           error, residual_error, compute_residual_error,
                           forward_indices, not_blocked_indices, forward_dimensions_and_alpha, truncated_dimension);
    }



    template <typename T>
    void swap_bonds(ATRG::Tensor<T> &B, ATRG::Tensor<T> &C, ATRG::Tensor<T> &X, ATRG::Tensor<T> &Y, const uint blocking_direction,
                    T &error, T &residual_error, const bool compute_residual_error,
                    const std::vector<uint> &forward_indices, std::vector<uint> &forward_dimensions_and_alpha, const uint D_truncated) {

        arma::Mat<T> U_B;
        arma::Mat<T> U_C;
        arma::Mat<T> U_M;

        swap_bonds(B, C, X, Y, blocking_direction, error, residual_error, compute_residual_error,
                   forward_indices, forward_dimensions_and_alpha, D_truncated,
                   U_B, U_C, U_M);
    }



    template <typename T>
    void swap_impure_bonds(ATRG::Tensor<T> &B, ATRG::Tensor<T> &C, ATRG::Tensor<T> &X, ATRG::Tensor<T> &Y, const uint blocking_direction,
                           T &error, T &residual_error, const bool compute_residual_error,
                           const std::vector<uint> &forward_indices, std::vector<uint> &forward_dimensions_and_alpha,
                           const arma::Mat<T> &U_B, const arma::Mat<T> &U_C, const arma::Mat<T> &U_M) {
        arma::Mat<T> B_flat;
        arma::Mat<T> C_flat;

        auto not_blocked_indices(forward_indices);
        not_blocked_indices.erase(not_blocked_indices.begin() + blocking_direction);

        // separate the not blocked indices
        B.flatten(not_blocked_indices, {blocking_direction, B.get_order() - 1}, B_flat);
        C.flatten(not_blocked_indices, {blocking_direction, C.get_order() - 1}, C_flat);

        uint mu_dimension = U_B.n_cols;
        uint nu_dimension = U_C.n_cols;

        // make a tensor of S*V with indices: x_trunc, alpha, mu
        // forward and backward dimensions are the same!
        B.reshape({forward_dimensions_and_alpha[blocking_direction], forward_dimensions_and_alpha.back(), mu_dimension});
        B_flat = B_flat.t() * U_B;
        B.inflate({0, 1}, {2}, B_flat);
        B.flatten({0}, {1, 2}, B_flat);


        // make a tensor of S*V with indices: x_trunc, beta, nu
        C.reshape({forward_dimensions_and_alpha[blocking_direction], forward_dimensions_and_alpha.back(), nu_dimension});
        C_flat = C_flat.t() * U_C;
        C.inflate({0, 1}, {2}, C_flat);
        C.flatten({0}, {1, 2}, C_flat);


        X.reshape({forward_dimensions_and_alpha.back(), mu_dimension, forward_dimensions_and_alpha.back(), nu_dimension});

        // B.t() * C with indices: alpha, mu, beta, nu
        B_flat = B_flat.t() * C_flat;
        X.inflate({0, 1}, {2, 3}, B_flat);
        X.flatten({0, 3}, {2, 1}, B_flat);

        C_flat = B_flat.t() * U_M;
        B_flat = U_M;


        uint truncated_dimension = U_M.n_cols;

        assemble_new_bonds(X, Y, B_flat, C_flat, U_B, U_C, blocking_direction,
                           error, residual_error, compute_residual_error,
                           forward_indices, not_blocked_indices, forward_dimensions_and_alpha, truncated_dimension);
    }



    /**
     * helper function for squeeze_bonds; computes an isometry from two tensors
     */
    template <typename T>
    void isometry(ATRG::Tensor<T> &A, ATRG::Tensor<T> &X, arma::Mat<T> &U_P, const uint index,
                  T &error, T &residual_error, const bool compute_residual_error,
                  const std::vector<uint> &psi_indices, const uint D_truncated) {
        arma::Mat<T> L_mat;
        // single out "index" and alpha
        A.flatten({index, A.get_order() - 1}, psi_indices, L_mat);
        // .t() gives the complex conjugate for complex matrices
        L_mat = L_mat * L_mat.t();
        // matrix is sorted: "index", alpha, "index", alpha
        // we want the tensor to be: "index", "index", alpha, alpha
        auto A_dimensions = A.get_dimensions();
        ATRG::Tensor<T> L({A_dimensions[index], A_dimensions[index],
                           A_dimensions.back(), A_dimensions.back()});
        L.inflate({0, 2}, {1, 3}, L_mat);
        L.flatten({0, 1}, {2, 3}, L_mat);


        arma::Mat<T> W_mat;
        X.flatten({index, X.get_order() - 1}, psi_indices, W_mat);
        W_mat = W_mat * W_mat.t();
        // reuse L to save memory
        auto X_dimensions = X.get_dimensions();
        L.reshape({X_dimensions[index], X_dimensions[index],
                   X_dimensions.back(), X_dimensions.back()});
        L.inflate({0, 2}, {1, 3}, W_mat);
        L.flatten({0, 1}, {2, 3}, W_mat);


        arma::Mat<T> P = L_mat * W_mat.t();
        // reuse L to save memory
        L.reshape({A_dimensions[index], A_dimensions[index], X_dimensions[index], X_dimensions[index]});
        // order in P: index_A, index_A, index_X, index_X
        // reorder to: index_A, index_X, index_A, index_X
        L.inflate({0, 2}, {1, 3}, P);
        L.flatten({0, 1}, {2, 3}, P);

        arma::Col<T> S;
        // U should be an isometry
        // V.t() * U = 1
        // order in U/V: {index_A, index_X}, eta
        arma::Mat<T> V_P;
        error += svd(P, U_P, V_P, S, D_truncated);

        if(compute_residual_error) {
            auto residual_error_P = residual_svd(P, U_P, V_P, S);
            residual_error += residual_error_P * residual_error_P;
        }
    }



    /**
     * helper function for squeeze_bonds; applys the squeezers to A/X, Y/D
     */
    template <typename T>
    std::vector<uint> squeeze(ATRG::Tensor<T> &A, ATRG::Tensor<T> &X, ATRG::Tensor<T> &G, std::vector<arma::Mat<T>> E_i, const uint blocking_direction,
                              const std::vector<uint> &forward_indices, const std::vector<uint> &backward_indices, const uint D_truncated) {
        arma::Mat<T> A_flat;
        arma::Mat<T> X_flat;

        A.flatten(forward_indices, {A.get_order() - 1}, A_flat);
        X.flatten(forward_indices, {X.get_order() - 1}, X_flat);
        // reuse A for the product A * X
        A_flat = A_flat * X_flat.t();

        // index order will be: x1, y1, z1, ... x2, y2, z2, ...
        std::vector<uint> all_dimensions(A.get_dimensions());
        all_dimensions.resize(all_dimensions.size() - 1);
        auto X_dimensions = X.get_dimensions();
        all_dimensions.insert(all_dimensions.end(), X_dimensions.begin(), X_dimensions.end() - 1);

        auto all_indices(forward_indices);
        all_indices.insert(all_indices.end(), backward_indices.begin(), backward_indices.end());

        G.reshape(all_dimensions);
        G.inflate(forward_indices, backward_indices, A_flat);

        uint number_backward_indices = forward_indices.size();


        for(decltype(E_i.size()) i = 0; i < E_i.size(); ++i) {
            uint index = i;
            if(index >= blocking_direction) {
                ++index;
            }

            auto remaining_indices(all_indices);
            remaining_indices.erase(remaining_indices.begin() + index);
            // we already remove an index!
            remaining_indices.erase(remaining_indices.begin() + index + number_backward_indices - 1);


            // single out y1, y2 or z1, z2 etc.
            G.flatten(remaining_indices, {index, index + number_backward_indices}, A_flat);
            // remaining_indices, y_new -> the order decreases by one
            A_flat = A_flat * E_i[i];

            // we insert the new index at the position of the forward index
            all_dimensions.erase(all_dimensions.begin() + index + number_backward_indices);
            all_dimensions[index] = A_flat.n_cols;

            // we remove the backward index at forward_index + number_backward_indices -> move all indices above that
            all_indices.resize(all_indices.size() - 1);
            remaining_indices = all_indices;
            remaining_indices.erase(remaining_indices.begin() + index);

            --number_backward_indices;

            G.reshape(all_dimensions);
            G.inflate(remaining_indices, {index}, A_flat);
        }

        return all_dimensions;
    }




    template <typename T>
    void compute_squeezers(ATRG::Tensor<T> &A, ATRG::Tensor<T> &D, ATRG::Tensor<T> &X, ATRG::Tensor<T> &Y, const uint blocking_direction,
                       T &error, T &residual_error, const bool compute_residual_error,
                       const std::vector<uint> &forward_indices, const std::vector<uint> &backward_indices, std::vector<uint> &forward_dimensions_and_alpha, const uint D_truncated,
                       std::vector<arma::Mat<T>> &E_i, std::vector<arma::Mat<T>> &F_i) {
        /*
         * indices A: all forward, alpha
         * indices D: all backward, beta
         * indices X: all forward, alpha
         * indices Y: all backward, beta
         */
        // make separate squeezers for each not blocked mode
        E_i.resize(forward_indices.size() - 1);
        F_i.resize(forward_indices.size() - 1);

        for(uint index = 0; index < forward_indices.size(); ++index) {
            if(index == blocking_direction) {
                continue;
            }

            auto psi_indices(forward_indices);
            psi_indices.erase(psi_indices.begin() + index);

            // compute an isometry from A and X
            arma::Mat<T> U_P;
            isometry(A, X, U_P, index, error, residual_error, compute_residual_error, psi_indices, D_truncated);

            // repeat for Y and D
            arma::Mat<T> U_Q;
            isometry(Y, D, U_Q, index, error, residual_error, compute_residual_error, psi_indices, D_truncated);

            // insert the isometries U/V between A-X and Y-D: U_P_T - U_P = U_Q - U_Q_T
            // and remodel U_P = U_Q to get one instead of two bonds
            arma::Mat<T> N = U_P.t() * U_Q;
            arma::Col<T> S;
            arma::Mat<T> U_N;
            arma::Mat<T> V_N;
            error += svd(N, U_N, V_N, S);

            if(compute_residual_error) {
                auto residual_error_N = residual_svd(N, U_N, V_N, S);
                residual_error += residual_error_N * residual_error_N;
            }

            S.for_each([](auto &element) {element = std::sqrt(element);});

            // compute the index in E_i, F_i
            uint i = index;
            if(i > blocking_direction) {
                --i;
            }

            /*
             * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
             * ! U_P and U_Q should be complex conjugated !
             * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
             */
            // {index_A, index_X}, new index
            E_i[i] = U_P * U_times_S(U_N, S);
            F_i[i] = U_Q * U_times_S(V_N, S);

#ifdef DEBUG
            std::cout << "E_" << i << ":" << std::endl << E_i[i] << std::endl;
            std::cout << "F_" << i << ":" << std::endl << F_i[i] << std::endl;
#endif
        }
    }



    template <typename T>
    void compute_squeezers(ATRG::Tensor<T> &A, ATRG::Tensor<T> &D, ATRG::Tensor<T> &X, ATRG::Tensor<T> &Y, ATRG::Tensor<T> &G, ATRG::Tensor<T> &H, const uint blocking_direction,
                       T &error, T &residual_error, const bool compute_residual_error,
                       const std::vector<uint> &forward_indices, const std::vector<uint> &backward_indices, std::vector<uint> &forward_dimensions_and_alpha, const uint D_truncated) {
        std::vector<arma::Mat<T>> E_i;
        std::vector<arma::Mat<T>> F_i;

        compute_squeezers(A, D, X, Y, blocking_direction, error, residual_error, compute_residual_error,
                      forward_indices, backward_indices, forward_dimensions_and_alpha, D_truncated,
                      E_i, F_i);

        // squeeze the bonds to make G and H
        auto all_dimensions = squeeze(A, X, G, E_i, blocking_direction, forward_indices, backward_indices, D_truncated);

        squeeze(Y, D, H, F_i, blocking_direction, forward_indices, backward_indices, D_truncated);

        // update the bond size
        for(decltype(all_dimensions.size()) i = 0; i < all_dimensions.size(); ++i) {
            forward_dimensions_and_alpha[i] = all_dimensions[i];
        }
    }



    template <typename T>
    void contract_bond(ATRG::Tensor<T> &G, ATRG::Tensor<T> &H, ATRG::Tensor<T> &A, ATRG::Tensor<T> &B, ATRG::Tensor<T> &C, ATRG::Tensor<T> &D, const uint blocking_direction,
                       T &error, T &residual_error, const bool compute_residual_error,
                       const std::vector<uint> &forward_indices, std::vector<uint> &forward_dimensions_and_alpha,
                       arma::Mat<T> &U_G, arma::Mat<T> &U_H, arma::Mat<T> &U_K) {
        arma::Mat<T> flat;
        // order in G: all forward indices, contraction direction backward
        G.flatten(forward_indices, {G.get_order() - 1}, flat);

        arma::Col<T> S_G;
        arma::Mat<T> V_G;
        error += svd(flat, U_G, V_G, S_G);

        if(compute_residual_error) {
            auto residual_error_G = residual_svd(flat, U_G, V_G, S_G);
            residual_error += residual_error_G * residual_error_G;
        }


        // H has the bond to G and the backward index in blocking direction swapped.
        auto backward_indices_in_H(forward_indices);
        backward_indices_in_H[blocking_direction] = H.get_order() - 1;
        H.flatten(backward_indices_in_H, {blocking_direction}, flat);

        arma::Col<T> S_H;
        arma::Mat<T> V_H;
        error += svd(flat, U_H, V_H, S_H);

        if(compute_residual_error) {
            auto residual_error_H = residual_svd(flat, U_H, V_H, S_H);
            residual_error += residual_error_H * residual_error_H;
        }


        V_G = U_times_S(V_G, S_G);
        V_H = U_times_S(V_H, S_H);

        // reuse flat for K to save memory:
        flat = V_G.t() * V_H;
        arma::Col<T> S_K;
        arma::Mat<T> V_K;
        error += svd(flat, U_K, V_K, S_K);

        if(compute_residual_error) {
            auto residual_error_K = residual_svd(flat, U_K, V_K, S_K);
            residual_error += residual_error_K * residual_error_K;
        }

        // update the alpha bond
        forward_dimensions_and_alpha[forward_dimensions_and_alpha.size() - 1] = S_K.n_elem;

        flat = U_G * U_K;
        A.reshape(forward_dimensions_and_alpha);
        A.inflate(forward_indices, {A.get_order() - 1}, flat);


        arma::Mat<T> SV_K = U_times_S(V_K, S_K);
        flat = U_H * SV_K;
        B.reshape(forward_dimensions_and_alpha);
        B.inflate(forward_indices, {B.get_order() - 1}, flat);


        SV_K = U_times_S(U_K, S_K);
        flat = U_G * SV_K;
        C.reshape(forward_dimensions_and_alpha);
        C.inflate(forward_indices, {C.get_order() - 1}, flat);


        flat = U_H * V_K;
        D.reshape(forward_dimensions_and_alpha);
        D.inflate(forward_indices, {D.get_order() - 1}, flat);
    }



    template <typename T>
    void contract_bond(ATRG::Tensor<T> &G, ATRG::Tensor<T> &H, ATRG::Tensor<T> &A, ATRG::Tensor<T> &B, ATRG::Tensor<T> &C, ATRG::Tensor<T> &D, const uint blocking_direction,
                       T &error, T &residual_error, const bool compute_residual_error,
                       const std::vector<uint> &forward_indices, std::vector<uint> &forward_dimensions_and_alpha) {
        arma::Mat<T> U_G;
        arma::Mat<T> U_H;
        arma::Mat<T> U_K;

        contract_bond(G, H, A, B, C, D, blocking_direction, error, residual_error, compute_residual_error,
                      forward_indices, forward_dimensions_and_alpha, U_G, U_H, U_K);
    }



    template <typename T>
    void contract_impure_bond(ATRG::Tensor<T> &G, ATRG::Tensor<T> &H, ATRG::Tensor<T> &B, const uint blocking_direction,
                              T &error, T &residual_error, const bool compute_residual_error,
                              const std::vector<uint> &forward_indices, const std::vector<uint> &forward_dimensions_and_alpha,
                              arma::Mat<T> &U_G, arma::Mat<T> &U_H, arma::Mat<T> &U_K) {

        arma::Mat<T> flat;
        // order in G: all forward indices, contraction direction backward
        G.flatten(forward_indices, {G.get_order() - 1}, flat);
        arma::Mat<T> SV_G = flat.t() * U_G;

        // H has the bond to G and the backward index in blocking direction swapped.
        auto backward_indices_in_H(forward_indices);
        backward_indices_in_H[blocking_direction] = H.get_order() - 1;
        H.flatten(backward_indices_in_H, {blocking_direction}, flat);
        arma::Mat<T> SV_H = flat.t() * U_H;

        // K = SV_G.t() * SV_H; We want U_K.t() * K to get S_K * V_K.t(),
        // but we transpose the operation to make the flattening easier
        flat = SV_H.t() * SV_G * U_K;
        flat = U_H * flat;
        B.reshape(forward_dimensions_and_alpha);
        B.inflate(forward_indices, {B.get_order() - 1}, flat);
    }


    /**
     * compute the last trace; G and H have the forward and backward indices first
     * and then their common bond index last
     */
    template<typename T>
    T trace(const ATRG::Tensor<T> &G, const ATRG::Tensor<T> &H) {
        T result = 0;

        for(decltype(G.get_size()) i = 0; i < G.get_size(); ++i) {
            result += G(i) * H(i);
        }

        return result;
    }


    /**
     * compute the final result including the scalefactors that were removed earlier
     */
    template<typename T>
    T result_with_scalefactors(const ATRG::Tensor<T> &G, const ATRG::Tensor<T> &H, const long double logScalefactors, const int volume,
                               T &error, T &residual_error, bool print = true) {
        auto last_result = trace(G, H);

        long double result = (logScalefactors + std::log(last_result)) / volume;

#ifdef DEBUG
        std::cout << "logScalefactors: " << logScalefactors << std::endl;
        std::cout << "log(last_result): " << std::log(last_result) << std::endl;
#endif

        if(std::isnan(result) || std::isinf(result)) {
            if(print) {
                std::cerr << "    the last step went wrong; use only scaling factor..." << std::endl;

                auto error_last_step = std::log(std::abs(last_result)) * std::log(std::abs(last_result))
                                    / (logScalefactors * logScalefactors);

                error += error_last_step;
                residual_error += error_last_step;
            }

            result = logScalefactors / volume;
        }

        return result;
    }



    /**
     * compute the log value of the partition sum for the given tensor and a lattice of given dimensions
     * returns log(Z) and an error estimate
     */
    template <typename T>
    std::tuple<T, T, T> compute_logZ(ATRG::Tensor<T> &tensor, const std::vector<uint> lattice_dimensions, const uint D_truncated,
                                     const bool compute_residual_error, const BlockingMode blocking_mode = t_blocking) {
        std::cout << "  computing log(Z):" << std::endl;
        auto starttime = std::chrono::high_resolution_clock::now();
        auto splittime = starttime;

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

        auto lattice_sizes(lattice_dimensions);
        std::for_each(lattice_sizes.begin(), lattice_sizes.end(), [](auto &element) {element = std::pow(2, element);});
        auto volume = std::accumulate(lattice_sizes.begin(), lattice_sizes.end(), 1, std::multiplies<T>());
        auto remaining_volume = volume;
        /*
         * we compute this errors with error propagation:
         * if we multiply to quantities we compute:
         *     error^2 = error1^2 + error2^2
         * and take the square root over everything in the end
         */
        T error = 0;
        T residual_error = 0;

        long double logScalefactors = 0;

        uint physical_dimension = tensor.get_order() / 2;

        // make lists of indices from 0 to >physical dimension< and from >physical dimension< to tensor order
        std::vector<uint> forward_indices(physical_dimension);
        std::iota(forward_indices.begin(), forward_indices.end(), 0);

        std::vector<uint> backward_indices(physical_dimension);
        std::iota(backward_indices.begin(), backward_indices.end(), physical_dimension);

        //=============================================================================================

        arma::Mat<T> flat;
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
        forward_dimensions_and_alpha.push_back(U.n_cols);

        auto backward_dimensions_and_alpha(backward_dimensions);
        backward_dimensions_and_alpha.push_back(U.n_cols);

        // we don't need the tensor from here on, so we free the memory
        tensor.reshape({0});

        // create A, B, C, D tensors from our SVD results:
        ATRG::Tensor<T> A(forward_dimensions_and_alpha);
        ATRG::Tensor<T> B(backward_dimensions_and_alpha);
        ATRG::Tensor<T> C(forward_dimensions_and_alpha);
        ATRG::Tensor<T> D(backward_dimensions_and_alpha);

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
        ATRG::Tensor<T> X;
        ATRG::Tensor<T> Y;
        ATRG::Tensor<T> G;
        ATRG::Tensor<T> H;

        std::cout << "    decomposed initial tensor...  " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - splittime)
                     .count() / 1e3
                  << " seconds" << std::endl;
        splittime = std::chrono::high_resolution_clock::now();

        std::cout << "      memory footprint: " << get_usage() << " GB" << std::endl;

        //=============================================================================================

        // contract the lattice
        uint blocking_direction = physical_dimension - 1;
        if(blocking_mode == s_blocking) {
            blocking_direction = 0;
        }

        std::vector<uint> finished_blockings(lattice_dimensions.size(), 0);
        bool finished = false;

        while(!finished) {
            remaining_volume /= 2;

            //=============================================================================================
            // swap the bonds, the not blocked modes between B and C and gain the tensors X and Y:
            // !!! after this step only forward_dimensions_and_alpha holds the correct bond sizes !!!
            swap_bonds(B, C, X, Y, blocking_direction,
                       error, residual_error, compute_residual_error,
                       forward_indices, forward_dimensions_and_alpha, D_truncated);

            std::cout << "X:" << std::endl << X << std::endl;
            std::cout << "Y:" << std::endl << Y << std::endl;

            auto Y_scale = std::max(std::abs(Y.max()), std::abs(Y.min()));
            Y.rescale(1.0 / Y_scale);

            logScalefactors += remaining_volume * std::log(Y_scale);

#ifdef DEBUG
            std::cout << "Y_scale: " << Y_scale << std::endl;
#endif

            // contract the double bonds of A, X in forward and B, D in backward direction
            compute_squeezers(A, D, X, Y, G, H, blocking_direction,
                          error, residual_error, compute_residual_error,
                          forward_indices, backward_indices, forward_dimensions_and_alpha, D_truncated);

            std::cout << "G:" << std::endl << G << std::endl;
            std::cout << "H:" << std::endl << H << std::endl;

            // rescale G and H
            auto G_scale = std::max(std::abs(G.max()), std::abs(G.min()));
            G.rescale(1.0 / G_scale);
            auto H_scale = std::max(std::abs(H.max()), std::abs(H.min()));
            H.rescale(1.0 / H_scale);

            logScalefactors += remaining_volume * (std::log(G_scale) + std::log(H_scale));

#ifdef DEBUG
            std::cout << "G_scale: " << G_scale << std::endl;
            std::cout << "H_scale: " << H_scale << std::endl;
#endif


            //=============================================================================================

            ++finished_blockings[blocking_direction];

            auto old_blocking_direction = blocking_direction;

            // decide the next truncation direction:
            switch(blocking_mode) {
            case s_blocking:
                if(finished_blockings[blocking_direction] >= lattice_dimensions[blocking_direction]) {
                    if(blocking_direction == physical_dimension - 1) {
                        std::cout << "    finished s-blocking..." << std::endl;

                        finished = true;
                        continue;
                    } else {
                        std::cout << "    s-blocked direction " << blocking_direction << "...  " <<
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::high_resolution_clock::now() - splittime)
                                     .count() / 1e3
                                  << " seconds" << std::endl;
                        splittime = std::chrono::high_resolution_clock::now();

                        // block the next direction
                        ++blocking_direction;

                        std::cout << "      memory footprint: " << get_usage() << " GB" << std::endl;
                    }
                }

                break;
            default:
                // t-blocking:
                if(finished_blockings[blocking_direction] >= lattice_dimensions[blocking_direction]) {
                    if(blocking_direction == 0) {
                        std::cout << "    finished t-blocking..." << std::endl;

                        finished = true;
                        continue;
                    } else {
                        std::cout << "    t-blocked direction " << blocking_direction << "...  " <<
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::high_resolution_clock::now() - splittime)
                                     .count() / 1e3
                                  << " seconds" << std::endl;
                        splittime = std::chrono::high_resolution_clock::now();

                        // block the next direction
                        --blocking_direction;

                        std::cout << "      memory footprint: " << get_usage() << " GB" << std::endl;
                    }
                }
            }

            std::cout << "      estimated result: " << logScalefactors / volume << std::endl;

            // make new A, B, C, D from G and H
            contract_bond(G, H, A, B, C, D, old_blocking_direction,
                          error, residual_error, compute_residual_error,
                          forward_indices, forward_dimensions_and_alpha);
        }

        std::cout << "      memory footprint: " << get_usage() << " GB" << std::endl;


        auto logZ = result_with_scalefactors(G, H, logScalefactors, volume, error, residual_error);


        std::cout << std::endl << "\033[1;33m    Runtime:\033[0m " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - starttime)
                     .count() / 1e3
                  << " seconds" << std::endl;

        return {logZ, std::sqrt(error), std::sqrt(residual_error)};
    }



    /**
     * compute the log value of the partition sum for the given tensor, impurity pair and a lattice of given dimensions
     * returns log(Z_impure) and an error estimate
     */
    template <typename T>
    std::tuple<T, T, T, T> compute_single_impurity(ATRG::Tensor<T> &tensor, ATRG::Tensor<T> &impurity,
                                                const std::vector<uint> lattice_dimensions, const uint D_truncated,
                                                const bool compute_residual_error, const BlockingMode blocking_mode = t_blocking) {
        std::cout << "  computing log(Z) with one impurity:" << std::endl;
        auto starttime = std::chrono::high_resolution_clock::now();
        auto splittime = starttime;

        if(tensor.get_order() != lattice_dimensions.size() * 2
           || impurity.get_order() != lattice_dimensions.size() * 2) {
            std::cerr << "  In ATRG::compute_single_impurity: order of tensor and lattice dimensions don't match!" << std::endl;
            throw 0;
        }

        for(uint i = 0; i < tensor.get_order() / 2; ++i) {
            if(tensor.get_dimensions(i) != tensor.get_dimensions(i + tensor.get_order() / 2)
               || impurity.get_dimensions(i) != impurity.get_dimensions(i + impurity.get_order() / 2)) {
                std::cerr << "  In ATRG::compute_single_impurity: dimensions of forward and backward mode don't match!" << std::endl;
                throw 0;
            }
        }

        auto lattice_sizes(lattice_dimensions);
        std::for_each(lattice_sizes.begin(), lattice_sizes.end(), [](auto &element) {element = std::pow(2, element);});
        auto volume = std::accumulate(lattice_sizes.begin(), lattice_sizes.end(), 1, std::multiplies<T>());
        auto remaining_volume = volume;

        /*
         * we compute this errors with error propagation:
         * if we multiply to quantities we compute:
         *     error^2 = error1^2 + error2^2
         * and take the square root over everything in the end
         */
        T error = 0;
        T residual_error = 0;

        long double Scalefactors = 1;
        long double last_result = 0;

        long double logScalefactors = 0;

        uint physical_dimension = tensor.get_order() / 2;

        // make lists of indices from 0 to >physical dimension< and from >physical dimension< to tensor order
        std::vector<uint> forward_indices(physical_dimension);
        std::iota(forward_indices.begin(), forward_indices.end(), 0);

        std::vector<uint> backward_indices(physical_dimension);
        std::iota(backward_indices.begin(), backward_indices.end(), physical_dimension);

        //=============================================================================================

        arma::Mat<T> flat;
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

        impurity.flatten(forward_indices, backward_indices, flat);


        auto dimensions = tensor.get_dimensions();
        decltype(dimensions) forward_dimensions(dimensions.begin(), dimensions.begin() + physical_dimension);
        decltype(dimensions) backward_dimensions(dimensions.begin() + physical_dimension, dimensions.end());

        auto forward_dimensions_and_alpha(forward_dimensions);
        forward_dimensions_and_alpha.push_back(U.n_cols);

        auto backward_dimensions_and_alpha(backward_dimensions);
        backward_dimensions_and_alpha.push_back(U.n_cols);

        tensor.reshape({0});


        // create A, B, C, D tensors from our SVD results:
        ATRG::Tensor<T> A(forward_dimensions_and_alpha);
        ATRG::Tensor<T> B(backward_dimensions_and_alpha);
        ATRG::Tensor<T> C(forward_dimensions_and_alpha);
        ATRG::Tensor<T> D(backward_dimensions_and_alpha);

        // use the same U and V for the impure tensor
        ATRG::Tensor<T> B_impure(backward_dimensions_and_alpha);
        ATRG::Tensor<T> C_impure(forward_dimensions_and_alpha);
        ATRG::Tensor<T> D_impure(backward_dimensions_and_alpha);

        arma::Mat<T> US(U_times_S(U, S));
        arma::Mat<T> SVp(U_times_S(V, S));

        // the columns only have alpha as an index, meaning the last index of the tensor
        // B and D hold the backward modes, so their indices are relabeled
        A.inflate(forward_indices, {A.get_order() - 1}, U);
        B.inflate(forward_indices, {B.get_order() - 1}, SVp);
        C.inflate(forward_indices, {C.get_order() - 1}, US);
        D.inflate(forward_indices, {D.get_order() - 1}, V);


        // multiply U.t() at flat -> {alpha, backward_indices}
        // use the transpose instead to make the inflating cheaper
        flat = flat.t() * U;
        B_impure.inflate(forward_indices, {B_impure.get_order() - 1}, flat);

        // we don't need the tensors from here on, so we free the memory
        impurity.reshape({0});
        US.set_size(0);
        SVp.set_size(0);

        // intermediate tensors that we will need during the blocking
        ATRG::Tensor<T> X;
        ATRG::Tensor<T> Y;
        ATRG::Tensor<T> G;
        ATRG::Tensor<T> H;

        ATRG::Tensor<T> X_impure;
        ATRG::Tensor<T> Y_impure;
        ATRG::Tensor<T> G_impure;
        ATRG::Tensor<T> H_impure;

        std::cout << "    decomposed initial tensor...  " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - splittime)
                     .count() / 1e3
                  << " seconds" << std::endl;
        splittime = std::chrono::high_resolution_clock::now();

        std::cout << "      memory footprint: " << get_usage() << " GB" << std::endl;

        //=============================================================================================

        // contract the lattice
        uint blocking_direction = physical_dimension - 1;
        if(blocking_mode == s_blocking) {
            blocking_direction = 0;
        }

        std::vector<uint> finished_blockings(lattice_dimensions.size(), 0);
        bool finished = false;

        while(!finished) {
            remaining_volume /= 2;

            // the C and D come from the second tensor, in this case a pure tensor
            C_impure = C;
            D_impure = D;

            //=============================================================================================
            // swap the bonds, the not blocked modes between B and C and gain the tensors X and Y:
            // !!! after this step only forward_dimensions_and_alpha holds the correct bond sizes !!!
            auto forward_dimensions_and_alpha_copy = forward_dimensions_and_alpha;
            arma::Mat<T> U_B;
            arma::Mat<T> U_C;
            arma::Mat<T> U_M;
            swap_bonds(B, C, X, Y, blocking_direction,
                       error, residual_error, compute_residual_error,
                       forward_indices, forward_dimensions_and_alpha, D_truncated,
                       U_B, U_C, U_M);

            swap_impure_bonds(B_impure, C_impure, X_impure, Y_impure, blocking_direction,
                              error, residual_error, compute_residual_error,
                              forward_indices, forward_dimensions_and_alpha_copy,
                              U_B, U_C, U_M);

            auto Y_scale = std::max(std::abs(Y.max()), std::abs(Y.min()));
            Y.rescale(1.0 / Y_scale);

            logScalefactors += remaining_volume * std::log(Y_scale);

            auto Y_scale_impure = std::max(std::abs(Y_impure.max()), std::abs(Y_impure.min()));
            Y_impure.rescale(1.0 / Y_scale_impure);

            Scalefactors *= Y_scale_impure / Y_scale;

#ifdef DEBUG
            std::cout << "Y_scale: " << Y_scale << std::endl;
            std::cout << "Y_scale_impure: " << Y_scale_impure << std::endl;
#endif

            std::cout << "X:" << std::endl << X << std::endl;
            std::cout << "Y:" << std::endl << Y << std::endl;
            std::cout << "Y_impure:" << std::endl << Y_impure << std::endl;


            std::vector<arma::Mat<T>> E_i;
            std::vector<arma::Mat<T>> F_i;

            // contract the double bonds of A, X in forward and B, D in backward direction
            compute_squeezers(A, D, X, Y, blocking_direction,
                          error, residual_error, compute_residual_error,
                          forward_indices, backward_indices, forward_dimensions_and_alpha, D_truncated,
                          E_i, F_i);

            // squeeze the bonds to make G and H
            auto all_dimensions = squeeze(A, X, G, E_i, blocking_direction, forward_indices, backward_indices, D_truncated);
            squeeze(Y, D, H, F_i, blocking_direction, forward_indices, backward_indices, D_truncated);

            squeeze(Y_impure, D_impure, H_impure, F_i, blocking_direction, forward_indices, backward_indices, D_truncated);

            // update the bond size
            for(decltype(all_dimensions.size()) i = 0; i < all_dimensions.size(); ++i) {
                forward_dimensions_and_alpha[i] = all_dimensions[i];
            }

            std::cout << "G:" << std::endl << G << std::endl;
            std::cout << "H:" << std::endl << H << std::endl;
            std::cout << "H_impure:" << std::endl << H_impure << std::endl;

            // rescale G and H
            auto G_scale = std::max(std::abs(G.max()), std::abs(G.min()));
            G.rescale(1.0 / G_scale);
            auto H_scale = std::max(std::abs(H.max()), std::abs(H.min()));
            H.rescale(1.0 / H_scale);

            logScalefactors += remaining_volume * (std::log(G_scale) + std::log(H_scale));

            G_impure = G;
            auto H_scale_impure = std::max(std::abs(H_impure.max()), std::abs(H_impure.min()));
            H_impure.rescale(1.0 / H_scale_impure);

            // rest gets cancelled
            Scalefactors *= H_scale_impure / H_scale;

#ifdef DEBUG
            std::cout << "G_scale: " << G_scale << std::endl;
            std::cout << "H_scale: " << H_scale << std::endl;
            std::cout << "H_scale_impure: " << H_scale_impure << std::endl;
            std::cout << "Scalefactor: " << Scalefactors << std::endl;
#endif

            //=============================================================================================

            ++finished_blockings[blocking_direction];

            auto old_blocking_direction = blocking_direction;

            // decide the next truncation direction:
            switch(blocking_mode) {
            case s_blocking:
                if(finished_blockings[blocking_direction] >= lattice_dimensions[blocking_direction]) {
                    if(blocking_direction == physical_dimension - 1) {
                        std::cout << "    finished s-blocking..." << std::endl;

                        finished = true;
                        continue;
                    } else {
                        std::cout << "    s-blocked direction " << blocking_direction << "...  " <<
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::high_resolution_clock::now() - splittime)
                                     .count() / 1e3
                                  << " seconds" << std::endl;
                        splittime = std::chrono::high_resolution_clock::now();

                        // block the next direction
                        ++blocking_direction;

                        std::cout << "      memory footprint: " << get_usage() << " GB" << std::endl;
                    }
                }

                break;
            default:
                // t-blocking:
                if(finished_blockings[blocking_direction] >= lattice_dimensions[blocking_direction]) {
                    if(blocking_direction == 0) {
                        std::cout << "    finished t-blocking..." << std::endl;

                        finished = true;
                        continue;
                    } else {
                        std::cout << "    t-blocked direction " << blocking_direction << "...  " <<
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::high_resolution_clock::now() - splittime)
                                     .count() / 1e3
                                  << " seconds" << std::endl;
                        splittime = std::chrono::high_resolution_clock::now();

                        // block the next direction
                        --blocking_direction;

                        std::cout << "      memory footprint: " << get_usage() << " GB" << std::endl;
                    }
                }
            }

            last_result = trace(G_impure, H_impure) / trace(G, H);
            std::cout << "      result: " << Scalefactors * last_result << std::endl;

            // reuse U_B, U_C, U_M as U_G, U_H, U_K
            // make new A, B, C, D from G and H
            contract_bond(G, H, A, B, C, D, old_blocking_direction,
                          error, residual_error, compute_residual_error,
                          forward_indices, forward_dimensions_and_alpha,
                          U_B, U_C, U_M);

            contract_impure_bond(G_impure, H_impure, B_impure, old_blocking_direction,
                                 error, residual_error, compute_residual_error,
                                 forward_indices, forward_dimensions_and_alpha,
                                 U_B, U_C, U_M);
        }

        std::cout << "      memory footprint: " << get_usage() << " GB" << std::endl;

        auto tr_impure = trace(G_impure, H_impure);
        auto tr = trace(G, H);
        auto result = tr_impure / tr;

        // numerical unreliabilites can befall single steps -> ignore them
        // also, the traces will both be almost 1 at one point because of the rescaling and blocking with pure tensors
        if(std::isnan(result) || std::isinf(result)) {
            std::cerr << "    the last step went wrong; use only scaling factor..." << std::endl;

            result = Scalefactors;

            // add the error (Scalefactors / (Scalefactors * last_result))^2
            auto error_last_step = 1.0 / (last_result * last_result);
            error += error_last_step;
            residual_error += error_last_step;
        } else {
            result *= Scalefactors;
        }


        auto logZ = result_with_scalefactors(G, H, logScalefactors, volume, error, residual_error);


        std::cout << std::endl << "\033[1;33m    Runtime:\033[0m " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - starttime)
                     .count() / 1e3
                  << " seconds" << std::endl;

        return {result, std::sqrt(error), std::sqrt(residual_error), logZ};
    }
}

#endif // ATRG
