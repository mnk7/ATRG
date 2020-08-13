#ifndef SVD_H
#define SVD_H

#include <armadillo>
#include <redsvd-h/include/RedSVD/RedSVD-h>

namespace ATRG {

    /**
     * looks at degenerate singular values and either orders them by their extremal element
     * or orders them after a given U.
     */
    template <typename T>
    inline void stabilize_SV(arma::Col<T> &S, arma::Mat<T> &U, arma::Mat<T> &V, const uint D,
                             const double SV_uncertainty, const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0)) {
        // sort singular vectors of degenerate singular values:
        uint current_SV_position = 0;
        bool start_swapping = false;
        arma::Mat<T> degenerate_Us;
        arma::Mat<T> degenerate_Vs;

        for(uint i = 1; i < S.n_elem; ++i) {
            auto difference = std::abs(1.0 - (S[i] / S[current_SV_position]));

            if(difference <= SV_uncertainty) {
                // last SV?
                if(i == S.n_elem - 1) {
                    degenerate_Us = U.cols(current_SV_position, i);
                    degenerate_Vs = V.cols(current_SV_position, i);

                    start_swapping = true;
                } else {
                    continue;
                }
            } else {
                // were the last SV degenerated?
                if(i - 1 != current_SV_position) {
                    degenerate_Us = U.cols(current_SV_position, i - 1);
                    degenerate_Vs = V.cols(current_SV_position, i - 1);

                    start_swapping = true;
                } else if(i >= D) {
                    break;
                } else {
                    current_SV_position = i;
                }
            }


            if(start_swapping) {
#ifdef DEBUG
                std::cout << "S: " << std::endl << S << std::endl;
                std::cout << "swap " << degenerate_Us.n_cols
                          << " SV between " << current_SV_position
                          << " and " << current_SV_position + degenerate_Us.n_cols - 1 << std::endl;
#endif

                if(U_reference.n_elem == 0) {
                    for(uint j = 0; j < degenerate_Us.n_cols - 1; ++j) {
                        arma::uvec order_j = arma::stable_sort_index(arma::abs(degenerate_Us.col(j)), "descend");

                        for(uint k = j + 1; k < degenerate_Us.n_cols; ++k) {
                            // sort by the position of the largest entry in the vector:
                            arma::uvec order_k = arma::stable_sort_index(arma::abs(degenerate_Us.col(k)), "descend");

                            for(uint l = 0; l < order_j.n_elem; ++l) {
                                // the extremum of k is at a lower index -> swap
                                if(order_j(l) > order_k(l)) {
                                    degenerate_Us.swap_cols(j, k);
                                    degenerate_Vs.swap_cols(j, k);
                                    S.swap_rows(current_SV_position + j, current_SV_position + k);
                                    order_j = order_k;
                                    break;
                                } else if(order_j(l) < order_k(l)) {
                                    // the extremum of k is further back -> don't swap
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    // cycle through all SV in U_reference and compute U(j) * U(k) -> find best match
                    std::vector<T> match(degenerate_Us.n_cols);

                    for(uint j = 0; j < degenerate_Us.n_cols - 1; ++j) {
                        for(uint k = j; k < degenerate_Us.n_cols; ++k) {
                            match[k] = std::abs(arma::dot(U_reference.col(j), degenerate_Us.col(k)));
                        }

                        auto best_match = std::distance(match.begin() + j, std::max_element(match.begin() + j, match.end()));

                        degenerate_Us.swap_cols(j, j + best_match);
                        degenerate_Vs.swap_cols(j, j + best_match);
                        S.swap_rows(current_SV_position + j, current_SV_position + j + best_match);

                        // orthogonalize remaining vectors with respect to the already matched vectors
                        for(uint k = j + 1; k < degenerate_Us.n_cols; ++k) {
                            for(uint m = 0; m < j; ++m) {
                                degenerate_Us.col(k) -= arma::dot(degenerate_Us.col(m), degenerate_Us.col(k))
                                                        * degenerate_Us.col(m);
                            }

                            degenerate_Us.col(k) = arma::normalise(degenerate_Us.col(k));
                        }

                        for(uint k = j + 1; k < degenerate_Vs.n_cols; ++k) {
                            for(uint m = 0; m < j; ++m) {
                                degenerate_Vs.col(k) -= arma::dot(degenerate_Vs.col(m), degenerate_Vs.col(k))
                                                        * degenerate_Vs.col(m);
                            }

                            degenerate_Vs.col(k) = arma::normalise(degenerate_Vs.col(k));
                        }
                    }
                }

                U.cols(current_SV_position, current_SV_position + degenerate_Us.n_cols - 1) = degenerate_Us;
                V.cols(current_SV_position, current_SV_position + degenerate_Vs.n_cols - 1) = degenerate_Vs;

                start_swapping = false;
                current_SV_position = i;
            }
        }
    }


    template <typename T>
    inline T redsvd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D) {
        arma::Mat<T> Q_copy = Q;
        auto Q_eigen = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(Q_copy.memptr(), Q_copy.n_rows, Q_copy.n_cols);

        RedSVD::RedSVD<decltype(Q_eigen)> redsvd;
        redsvd.compute(Q_eigen, D * D);

        auto U_eigen = redsvd.matrixU();
        auto V_eigen = redsvd.matrixV();
        auto S_eigen = redsvd.singularValues();

        U = arma::Mat<T>(U_eigen.data(), U_eigen.rows(), U_eigen.cols(), false, true);
        V = arma::Mat<T>(V_eigen.data(), V_eigen.rows(), V_eigen.cols(), false, true);
        S = arma::Col<T>(S_eigen.data(), S_eigen.rows(), false, true);

        // compute the error from the singular values
        arma::Col<T> cumulative_sum = arma::cumsum(S * S);

        S.resize(D);
        U.resize(U.n_rows, S.n_elem);
        V.resize(V.n_rows, S.n_elem);

        // sum of all squared singular values - sum of kept squared singular vectors
        //      / sum of all squared singular vectors
        return (cumulative_sum[cumulative_sum.n_elem - 1] - cumulative_sum[D - 1])
                / cumulative_sum[cumulative_sum.n_elem - 1];
    }


    /**
     * compute SVD of a sparse matrix by means of the eigenvalues of Q or with armadillos own svd function
     * return the squared error
     */
    template <typename T>
    inline T svd(const arma::SpMat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14) {
        if(!arma::svds(U, S, V, Q, std::min(Q.n_cols, Q.n_rows))) {
            std::cerr << "  could not perform sparse SVD!" << std::endl;
            throw 0;
        }

        // compute the error from the singular values
        arma::Col<T> cumulative_sum = arma::cumsum(S * S);

        S.resize(D);
        U.resize(U.n_rows, S.n_elem);
        V.resize(V.n_rows, S.n_elem);

        // sum of all squared singular values - sum of kept squared singular vectors
        //      / sum of all squared singular vectors
        return (cumulative_sum[cumulative_sum.n_elem - 1] - cumulative_sum[D - 1])
                / cumulative_sum[cumulative_sum.n_elem - 1];
    }


    /**
     * compute SVD of a dense matrix by means of the eigenvalues of Q or with armadillos own svd function
     * return the squared error
     */
    template <typename T>
    inline T svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14) {
        if(!arma::svd(U, S, V, Q)) {
            std::cerr << "  could not perform SVD!" << std::endl;

            std::cout << "  trying redsvd:" << std::endl;
            return redsvd(Q, U, V, S, D);
        }

        // compute the error from the singular values
        arma::Col<T> cumulative_sum = arma::cumsum(S * S);


#ifdef DEBUG
        double diff_to_unitary_U = arma::norm(arma::eye(U.n_cols, U.n_cols) - U.t() * U, "fro");
        double diff_to_unitary_V = arma::norm(arma::eye(V.n_cols, V.n_cols) - V.t() * V, "fro");
        std::cout << "difference to unitarity U: " << diff_to_unitary_U << std::endl
                  << "                        V: " << diff_to_unitary_V << std::endl;
#endif


        if(SV_uncertainty >= 0) {
            stabilize_SV(S, U, V, D, SV_uncertainty, U_reference);
        }

        if(D <= S.n_elem) {
            S.resize(D);
            U.resize(U.n_rows, S.n_elem);
            V.resize(V.n_rows, S.n_elem);
        } else {
            std::cerr << "  in svd: could not cut S (" << S.n_elem << " elements) to " << D << " elements!" << std::endl;
        }

        if(cutoff > 0) {
            U.for_each([&cutoff](auto &element) {if(std::abs(element) < cutoff){element = 0;}});
            V.for_each([&cutoff](auto &element) {if(std::abs(element) < cutoff){element = 0;}});
            S.for_each([&S, &cutoff](auto &element) {if(std::abs(element / S(0)) < cutoff){element = 0;}});
        }

        // sum of all squared singular values - sum of kept squared singular vectors
        //      / sum of all squared singular vectors
        return (cumulative_sum[cumulative_sum.n_elem - 1] - cumulative_sum[D - 1])
                / cumulative_sum[cumulative_sum.n_elem - 1];
    }


    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V, const uint D,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0)) {
        arma::Col<T> S;

        return svd(Q, U, V, S, D, U_reference);
    }


    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0)) {
        return svd(Q, U, V, S, std::min(Q.n_cols, Q.n_rows), U_reference);
    }


    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0)) {
        return svd(Q, U, V, std::min(Q.n_cols, Q.n_rows), U_reference);
    }



    /**
     * compute the product of U*S or S*U'. S is a diagonal matrix represented as a vector
     */
    template <typename T>
    inline arma::Mat<T> U_times_S(const arma::Mat<T> &U, const arma::Col<T> &S) {
        arma::Mat<T> US(U.n_rows, S.n_elem, arma::fill::zeros);

        for(uint j = 0; j < U.n_rows; ++j) {
            // multiplicate element-wise
            US.row(j) = U.row(j) % S.t();
        }

        return US;
    }


    /**
     * compute the resiudal of a SVD decomposition
     */
    template <class MatrixType, typename T>
    inline T residual_svd(const MatrixType &matrix, const arma::Mat<T> &U, const arma::Mat<T> &V, const arma::Col<T> &S) {
        arma::Mat<T> matrix_svd = U_times_S(U, S);

        matrix_svd *= V.t();

        return arma::norm(matrix - matrix_svd, "fro") / arma::norm(matrix, "fro");
    }

}


#endif // SVD_H
