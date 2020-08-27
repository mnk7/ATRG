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
                             const double SV_uncertainty,
                             const arma::Mat<T> &U_reference_ext = arma::zeros<arma::Mat<T>>(0),
                             const bool only_U = false) {
        // sort singular vectors of degenerate singular values:
        uint current_SV_position = 0;
        bool start_swapping = false;
        arma::Mat<T> degenerate_Us;
        arma::Mat<T> degenerate_Vs;

        arma::Mat<T> U_reference = arma::zeros<arma::Mat<T>>(0);
        bool no_reference = true;
        // if the reference has less SV than D, we stop our comparision earlier
        uint last_checked_SV = D;

        if(U_reference_ext.n_elem > 0) {
            if(U_reference_ext.n_rows != U.n_rows) {
                std::cerr << "  In ATRG::stabilize_SV: reference has wrong size!" << std::endl;
            } else {
                U_reference = U_reference_ext;
                no_reference = false;
                last_checked_SV = std::min(D, static_cast<uint>(U_reference.n_cols));
            }
        }


        // rotate first vector
        if(no_reference) {
            uint max_index = arma::index_max(arma::abs(U.col(0)));
            if(U(max_index, 0) < 0) {
                U.col(0) *= -1;
                V.col(0) *= -1;
            }
        } else if(arma::dot(U_reference.col(0), U.col(0)) < 0) {
            U.col(0) *= -1;
            V.col(0) *= -1;
        }


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
                } else if(i > last_checked_SV) {
                    break;
                } else {
                    current_SV_position = i;
                }

                // rotate this
                if(no_reference) {
                    uint max_index = arma::index_max(arma::abs(U.col(i)));
                    if(U(max_index, i) < 0) {
                        U.col(i) *= -1;
                        V.col(i) *= -1;
                    }
                } else if(arma::dot(U_reference.col(i), U.col(i)) < 0) {
                    U.col(i) *= -1;
                    V.col(i) *= -1;
                }
            }


            if(start_swapping) {
#ifdef DEBUG
                std::cout << "S: " << std::endl << S << std::endl;
                std::cout << "swap " << degenerate_Us.n_cols
                          << " SV between " << current_SV_position
                          << " and " << current_SV_position + degenerate_Us.n_cols - 1 << std::endl;
#endif

                if(no_reference) {
                    // sort the SV by their largest entries
                    for(uint j = 0; j < degenerate_Us.n_cols; ++j) {
                        arma::uvec order_j = arma::stable_sort_index(arma::abs(degenerate_Us.col(j)), "descend");
                        uint current_j = j;

                        for(uint k = j + 1; k < degenerate_Us.n_cols; ++k) {
                            // sort by the position of the largest entry in the vector:
                            arma::uvec order_k = arma::stable_sort_index(arma::abs(degenerate_Us.col(k)), "descend");

                            for(uint l = 0; l < order_j.n_elem; ++l) {
                                // the extremum of k is at a lower index -> swap
                                if(order_j(l) > order_k(l)) {
                                    current_j = k;
                                    order_j = order_k;

                                    break;
                                } else if(order_j(l) < order_k(l)) {
                                    // the extremum of k is further back -> don't swap
                                    break;
                                }
                            }
                        }

                        if(current_j != j) {
                            degenerate_Us.swap_cols(j, current_j);
                            degenerate_Vs.swap_cols(j, current_j);
                            S.swap_rows(current_SV_position + j, current_SV_position + current_j);
                        }

                        // try to turn all vectors to the forward quadrant
                        if(U(order_j(0), j) < 0) {
                            degenerate_Us.col(j) *= -1;
                            degenerate_Vs.col(j) *= -1;
                        }
                    }
                } else {
                    if(only_U) {
                        std::cout << "swap only U!" << std::endl;
                        // cycle through all SV in U_reference and compute U(j) * U(k) -> find best match
                        // then rotate the SV so that U is roughly aligned with U_reference
                        for(uint j = 0; j < std::min(degenerate_Us.n_cols, U_reference.n_cols - current_SV_position); ++j) {
                            std::vector<T> match(degenerate_Us.n_cols - j);
                            double match_norm = 0;
                            uint best_match = j;
                            double best_match_value = 0;

                            for(uint k = j; k < degenerate_Us.n_cols; ++k) {
                                match[k - j] = arma::dot(U_reference.col(j + current_SV_position), degenerate_Us.col(k));
                                match_norm += match[k - j] * match[k - j];

                                if(std::abs(match[k - j]) > std::abs(best_match_value)) {
                                    best_match = k;
                                    best_match_value = match[k - j];
                                }
                            }
                            match_norm = std::sqrt(match_norm);


                            arma::Col<T> new_U_col(degenerate_Us.n_rows, arma::fill::zeros);
                            for(uint k = j; k < degenerate_Us.n_cols; ++k) {
                                new_U_col += (match[k - j] / match_norm) * degenerate_Us.col(k);
                            }
                            degenerate_Us.col(best_match) = new_U_col;


                            // swap
                            if(best_match != j) {
                                degenerate_Us.swap_cols(j, best_match);
                                S.swap_rows(current_SV_position + j, current_SV_position + best_match);
                            }


                            // orthogonalize remaining vectors with respect to the already matched vectors
                            for(uint k = j + 1; k < degenerate_Us.n_cols; ++k) {
                                for(uint m = j; m < k; ++m) {
                                    degenerate_Us.col(k) -= arma::dot(degenerate_Us.col(m), degenerate_Us.col(k))
                                                            * degenerate_Us.col(m);
                                }

                                degenerate_Us.col(k) = arma::normalise(degenerate_Us.col(k));
                            }
                        }
                    } else {
                        std::cout << "swap U and V" << std::endl;
                        // cycle through all SV in U_reference and compute U(j) * U(k) -> find best match
                        // then swap the SV
                        for(uint j = 0; j < std::min(degenerate_Us.n_cols, U_reference.n_cols - current_SV_position); ++j) {
                            std::vector<T> match(degenerate_Us.n_cols - j);
                            uint best_match = j;
                            double best_match_value = 0;

                            for(uint k = j; k < degenerate_Us.n_cols; ++k) {
                                match[k - j] = arma::dot(U_reference.col(j + current_SV_position), degenerate_Us.col(k));

                                if(std::abs(match[k - j]) > std::abs(best_match_value)) {
                                    best_match = k;
                                    best_match_value = match[k - j];
                                }
                            }

                            if(best_match_value < 0) {
                                degenerate_Us.col(best_match) *= -1;
                                degenerate_Vs.col(best_match) *= -1;
                            }

                            // swap
                            if(best_match != j) {
                                degenerate_Us.swap_cols(j, best_match);
                                degenerate_Vs.swap_cols(j, best_match);
                                S.swap_rows(current_SV_position + j, current_SV_position + best_match);
                            }
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
    inline void redsvd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D) {
        arma::Mat<T> Q_copy = Q;
        auto Q_eigen = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(Q_copy.memptr(), Q_copy.n_rows, Q_copy.n_cols);

        RedSVD::RedSVD<decltype(Q_eigen)> redsvd;
        redsvd.compute(Q_eigen, 2 * D);

        auto U_eigen = redsvd.matrixU();
        auto V_eigen = redsvd.matrixV();
        auto S_eigen = redsvd.singularValues();

        U = arma::Mat<T>(U_eigen.data(), U_eigen.rows(), U_eigen.cols(), false, true);
        V = arma::Mat<T>(V_eigen.data(), V_eigen.rows(), V_eigen.cols(), false, true);
        S = arma::Col<T>(S_eigen.data(), S_eigen.rows(), false, true);
    }



    /**
     * compute SVD of a dense matrix by means of the eigenvalues of Q or with armadillos own svd function
     * return the squared error
     */
    template <typename T>
    inline T svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D, const bool use_redsvd = false,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14, const bool only_U = false) {
        if(use_redsvd) {
            redsvd(Q, U, V, S, D);
        } else {
            if(!arma::svd(U, S, V, Q)) {
                std::cerr << "  arma::svd failed! using redsvd:" << std::endl;

                redsvd(Q, U, V, S, D);
            }
        }

        // compute the error from the singular values
        arma::Col<T> cumulative_sum = arma::cumsum(S * S);

        uint new_size = S.n_elem;
        if(D <= S.n_elem) {
            new_size = D;
        }

        // cut off singular values that are 0
        for(uint i = new_size; i > 0; --i) {
            if(arma::norm(U.col(i - 1)) > 0.8) {
                new_size = i;
                break;
            }
        }

        S.resize(new_size);
        U.resize(U.n_rows, new_size);
        V.resize(V.n_rows, new_size);


        if(SV_uncertainty >= 0) {
            stabilize_SV(S, U, V, D, SV_uncertainty, U_reference, only_U);
        }

#ifdef DEBUG
        double diff_to_unitary_U = arma::norm(arma::eye(U.n_cols, U.n_cols) - U.t() * U, "fro");
        double diff_to_unitary_V = arma::norm(arma::eye(V.n_cols, V.n_cols) - V.t() * V, "fro");
        std::cout << "difference to unitarity U: " << diff_to_unitary_U << std::endl
                  << "                        V: " << diff_to_unitary_V << std::endl;

        if(diff_to_unitary_U > 1e-3 || diff_to_unitary_V > 1e-3) {
            std::cerr << "  In ATRG::svd: isometry not unitary!" << std::endl;
            std::cout << "U:" << std::endl << U;
            std::cout << "V:" << std::endl << V;
            throw 0;
        }
#endif

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



    /**
     * compute SVD of a sparse matrix by means of the eigenvalues of Q or with armadillos own svd function
     * return the squared error
     */
    template <typename T>
    inline T svd(const arma::SpMat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D, const bool use_redsvd,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14, const bool only_U = false) {
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



    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V, const uint D, const bool use_redsvd = false,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14) {
        arma::Col<T> S;

        return svd(Q, U, V, S, D, use_redsvd, U_reference, SV_uncertainty, cutoff);
    }



    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const bool use_redsvd = false,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14) {
        return svd(Q, U, V, S, std::min(Q.n_cols, Q.n_rows), use_redsvd, U_reference, SV_uncertainty, cutoff);
    }



    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V, const bool use_redsvd = false,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14) {
        return svd(Q, U, V, std::min(Q.n_cols, Q.n_rows), use_redsvd, U_reference, SV_uncertainty, cutoff);
    }



    /**
     * SVD's where only the U gets computed. This allows to split Q into U and U.t() * Q
     */
    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Col<T> &S, const uint D, const bool use_redsvd = false,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14) {
        arma::Mat<T> V;

        return svd(Q, U, V, S, D, use_redsvd, U_reference, SV_uncertainty, cutoff, true);
    }


    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Col<T> &S, const bool use_redsvd = false,
                 const arma::Mat<T> &U_reference = arma::zeros<arma::Mat<T>>(0),
                 const double SV_uncertainty = 1e-3, const double cutoff = -1e-14) {
        return svd(Q, U, S, std::min(Q.n_cols, Q.n_rows), use_redsvd, U_reference, SV_uncertainty, cutoff);
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
