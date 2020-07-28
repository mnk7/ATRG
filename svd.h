#ifndef SVD_H
#define SVD_H

#include <armadillo>
#include <redsvd-h/include/RedSVD/RedSVD-h>

namespace ATRG {


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
    inline T svd(const arma::SpMat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D) {
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
    inline T svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D, const double SV_uncertainty = 1e-3) {
        if(!arma::svd(U, S, V, Q)) {
            std::cerr << "  could not perform SVD!" << std::endl;

            std::cout << "  trying redsvd:" << std::endl;
            return redsvd(Q, U, V, S, D);
        }

        // compute the error from the singular values
        arma::Col<T> cumulative_sum = arma::cumsum(S * S);

        S.resize(D);
        U.resize(U.n_rows, S.n_elem);
        V.resize(V.n_rows, S.n_elem);


        if(SV_uncertainty > 0) {
            // sort singular vectors of degenerate singular values:
            uint current_SV_position = 0;
            bool start_swapping = false;
            arma::Mat<T> degenerate_Us;
            arma::Mat<T> degenerate_Vs;

            for(uint i = 1; i < S.n_elem; ++i) {
                auto difference = std::abs(1.0 - (S[i] / S[current_SV_position]));

                if((difference > SV_uncertainty && i - 1 != current_SV_position)) {
                    degenerate_Us = U.cols(current_SV_position, i - 1);
                    degenerate_Vs = V.cols(current_SV_position, i - 1);

                    current_SV_position = i;
                    start_swapping = true;
                } else if(i == S.n_elem - 1 && difference <= SV_uncertainty) {
                    degenerate_Us = U.cols(current_SV_position, i);
                    degenerate_Vs = V.cols(current_SV_position, i);

                    start_swapping = true;
                } else {
                    current_SV_position = i;
                }

                if(start_swapping) {
                    for(uint j = 0; j < degenerate_Us.n_cols - 1; ++j) {
                        for(uint k = j + 1; k < degenerate_Us.n_cols; ++k) {
                            // look at all vector entries of the vectors at j and at k
                            for(uint l = 0; l < degenerate_Us.n_rows; ++l) {
                                // the vector at k is larger -> swap
                                if(degenerate_Us(j, l) < degenerate_Us(k, l)) {
                                    degenerate_Us.swap_cols(j, k);
                                    degenerate_Vs.swap_cols(j, k);
                                    break;
                                } else if(degenerate_Us(j, l) > degenerate_Us(k, l)) {
                                    // the vector at k is smaller -> don't swap
                                    break;
                                }
                            }
                        }
                    }

                    U.cols(current_SV_position, current_SV_position + degenerate_Us.n_cols - 1) = degenerate_Us;
                    V.cols(current_SV_position, current_SV_position + degenerate_Vs.n_cols - 1) = degenerate_Vs;

                    start_swapping = false;
                }
            }
        }

        // sum of all squared singular values - sum of kept squared singular vectors
        //      / sum of all squared singular vectors
        return (cumulative_sum[cumulative_sum.n_elem - 1] - cumulative_sum[D - 1])
                / cumulative_sum[cumulative_sum.n_elem - 1];
    }


    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V, const uint D) {
        arma::Col<T> S;

        return svd(Q, U, V, S, D);
    }


    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S) {
        return svd(Q, U, V, S, std::min(Q.n_cols, Q.n_rows));
    }


    template <class MatrixType, typename T>
    inline T svd(const MatrixType &Q, arma::Mat<T> &U, arma::Mat<T> &V) {
        return svd(Q, U, V, std::min(Q.n_cols, Q.n_rows));
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
