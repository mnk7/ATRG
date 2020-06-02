#ifndef SVD
#define SVD

#include <armadillo>

namespace ATRG {

    /**
     * compute SVD of a sparse matrix by means of the eigenvalues of Q or with armadillos own svd function
     * return the error
     */
    template <typename T>
    inline T svd(const arma::SpMat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D) {
        arma::svds(U, S, V, Q, std::min(Q.n_cols, Q.n_rows));

        // compute the error from the singular values
        arma::Col<T> cumulative_sum = arma::cumsum(S);

        S.resize(D);
        U.resize(U.n_rows, S.n_elem);
        V.resize(V.n_rows, S.n_elem);

        // sum of all singular values - sum of discarded singular vectors
        //      / sum of all singular vectors
        return (cumulative_sum[cumulative_sum.n_elem - 1] - cumulative_sum[D - 1])
                / cumulative_sum[cumulative_sum.n_elem - 1];
    }


    /**
     * compute SVD of a dense matrix by means of the eigenvalues of Q or with armadillos own svd function
     * return the error
     */
    template <typename T>
    inline T svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D) {
        arma::svd(U, S, V, Q);

        // compute the error from the singular values
        arma::Col<T> cumulative_sum = arma::cumsum(S);

        S.resize(D);
        U.resize(U.n_rows, S.n_elem);
        V.resize(V.n_rows, S.n_elem);

        // sum of all singular values - sum of discarded singular vectors
        //      / sum of all singular vectors
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


#endif // SVD
