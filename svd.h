#ifndef SVD
#define SVD

#include <armadillo>
#include <tensor.h>

namespace ATRG {

    /**
     * compute SVD by means of the eigenvalues of Q or with armadillos own svd function
     * return the error
     */
    template <typename T>
    inline T svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, const uint D, const bool armadillo = true) {
        if(armadillo) {
            arma::svd(U, S, V, Q, "dc");
        } else {
            arma::eig_sym(S, V, Q.t() * Q);
            V = arma::reverse(V, 1);

            arma::eig_sym(S, U, Q * Q.t());   // stored in ascending order in Armadillo
            U = arma::reverse(U, 1);
            S = arma::reverse(S, 0);

            S.for_each([](T &element) {element = std::sqrt(element);});
        }

        // compute the error from the squared singular values
        arma::Col<T> cumulative_sum = arma::cumsum(S);

        S.resize(D);
        U.resize(U.n_rows, S.n_elem);
        V.resize(V.n_rows, S.n_elem);

        // sum of all squared singular values - sum of discarded squared singular vectors
        //      / sum of all squared singular vectors
        return (cumulative_sum[cumulative_sum.n_elem - 1] - cumulative_sum[D - 1])
                / cumulative_sum[cumulative_sum.n_elem - 1];
    }


    template <typename T>
    inline T svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, const uint D) {
        arma::Col<T> S;

        return svd(Q, U, V, S, D);
    }


    template <typename T>
    inline T svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S) {
        return svd(Q, U, V, S, std::min(Q.n_cols, Q.n_rows));
    }


    template <typename T>
    inline T svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V) {
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
    template <typename T>
    inline T residual_svd(const arma::Mat<T> &matrix, const arma::Col<T> &S, const arma::Mat<T> &U, const arma::Mat<T> &V) {
        arma::Mat<T> matrix_svd = U_times_S(U, S);

        matrix_svd *= V.t();

        return arma::norm(matrix - matrix_svd, "fro") / arma::norm(matrix, "fro");
    }

}


#endif // SVD
