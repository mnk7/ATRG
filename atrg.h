#ifndef ATRG
#define ATRG

#include <armadillo>
#include <omp.h>

#include <tensor.h>

namespace ATRG {

    /**
     * compute SVD by means of the eigenvalues of Q
     */
    template <typename T>
    inline void svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S, uint D) {
        arma::eig_sym(S, V, Q.t() * Q);
        V = arma::reverse(V, 1);

        arma::eig_sym(S, U, Q * Q.t());   // stored in ascending order in Armadillo
        U = arma::reverse(U, 1);
        S = arma::reverse(S, 0);

        S.resize(D);
        U.resize(U.n_rows, S.n_elem);
        V.resize(V.n_rows, S.n_elem);

        for(uint i = 0; i < S.n_rows; ++i) {
            S[i] = std::sqrt(S[i]);
        }
    }


    template <typename T>
    inline void svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, uint D) {
        arma::Col<T> S;

        svd(Q, U, V, S, D);
    }


    template <typename T>
    inline void svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V, arma::Col<T> &S) {
        svd(Q, U, V, S, std::min(Q.n_cols, Q.n_rows));
    }


    template <typename T>
    inline void svd(const arma::Mat<T> &Q, arma::Mat<T> &U, arma::Mat<T> &V) {
        svd(Q, U, V, std::min(Q.n_cols, Q.n_rows));
    }

}

#endif // ATRG
