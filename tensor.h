#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <valarray>
#include <armadillo>

typedef unsigned int uint;


namespace ATRG {

    template<class T>
    class Tensor {
    public:
        Tensor() : order(0), dimtot(0) {}
        Tensor(const std::vector<uint>& dim);

        void reshape(const std::vector<uint> dim);
        void reshape(const uint k, const uint dk);

        void print();

        T* data() { return &t[0]; }
        void zero();
        T Frobnorm2() const;
        T Frobnorm() const;
        uint size() const { return dimtot; }
        uint get_order() const { return order; }
        uint get_dim(const uint k) const { return dim[k]; }
        void get_dim(std::vector<uint> &dim) { dim = this->dim; }
        uint get_base(const uint i) const { return base[i]; }
        double max() const;
        void rescale(const double s);

        uint flatindex(const std::vector<uint> index) const;
        std::vector<uint> index(const uint flatindex) const;
        void flatten(const uint k, arma::Mat<T> &flatk);
        void flatten_bf(const uint k, arma::Mat<T> &flatk);
        void inflate(const uint k, const arma::Mat<T> &flatk);

        void reduce(const uint k, const arma::Mat<T> &U, const uint D);
        void decompose_and_reduce(const uint k, const uint D);
        void decompose_and_reduce(const uint k1, const uint k2, const uint D);
        void svdkU(const uint k, arma::Mat<T> &U);


        T& operator()(const uint tot_index) {
            return t[tot_index];
        }
        const T& operator()(const uint tot_index) const {
            return t[tot_index];
        }
        T& operator()(const std::vector<uint>& index) {
            return t[flatindex(index)];
        }
        const T& operator()(const std::vector<uint>& index) const {
            return t[flatindex(index)];
        }
        std::valarray<T>& operator()(const std::slice& s) {
            return t[s];
        }
        std::valarray<T>& operator()(const std::slice& s) const {
            return t[s];
        }

        friend std::ostream &operator<<(std::ostream &out, Tensor<T> &t) {
            arma::Mat<T> flat;
            t.flatten(0, flat);
            out << flat;
            return out;
        }

    private:
        // order of tensor
        uint order;
        // number of elements in the tensor, saved in the last element of base
        uint dimtot;
        // dimensions of each index
        std::vector<uint> dim;
        // tensor content
        std::valarray<T> t;
        // base factor for each index -> hypercube for elements of the indices
        std::vector<uint> base;
    };


    /* ============================== Definition ============================== */

    /**
     * lowest index runs fastest, highest index lowest (column major for matrices)
     */
    template <class T>
    Tensor<T>::Tensor(const std::vector<uint> &dim) {
        reshape(dim);
    }


    /**
     * change the dimension of the tensor
     */
    template <class T>
    void Tensor<T>::reshape(const std::vector<uint> dim) {
        order = dim.size();
        this->dim = dim;

        // last element filled with dimtot (for efficiency of algorithms)
        base.resize(order + 1);
        base[0] = 1;
        for (decltype(order) i = 0; i < order; ++i) {
            base[i + 1] = base[i] * dim[i];
        }

        dimtot = base[order];
        t.resize(dimtot);
    }


    /**
     * change dimension of index k
     */
    template <class T>
    void Tensor<T>::reshape(const uint k, const uint dk) {
        // modify dim[k] and base from index k onwards
        dim[k] = dk;

        for (decltype(order) i = k; i < order; ++i) {
            base[i + 1] = base[i] * dim[i];
        }

        dimtot = base[order];
        t.resize(dimtot);
    }


    /**
     * print the tensor in a readable way
     */
    template<class T>
    void Tensor<T>::print() {
        for(uint i = 0; i < dimtot; ++i) {
            auto ind = index(i);

            std::cout << "[";
            for(uint j = 0; j < ind.size(); ++j) {
                std::cout << ind[j] << "\t";
            }

            std::cout << "] = " << t[i] << std::endl;
        }
    }


    /**
     * reset tensor to zero
     */
    template <class T>
    void Tensor<T>::zero() {
        t.assign(t.size(), 0);
    }


    /**
     * compute the square of the Frobenius norm of the tensor
     */
    template <class T>
    T Tensor<T>::Frobnorm2() const {
        T F = 0;

        for (decltype(t.size()) i = 0; i < t.size(); ++i) {
            // norm gives the square!
            F += std::norm(t[i]);
        }

        return F;
    }
    
    
    /**
     * ocmpute the Frobenius norm of the tensor
     */
    tempalte <class T>
    T Tensor<T>::Frobnorm() const {
        return std::sqrt(Frobnorm2());
    }


    /**
     * returns the largest element in the tensor
     */
    template <class T>
    double Tensor<T>::max() const {
         double maxt = 0;

         for (decltype(t.size()) i = 0; i < t.size(); ++i) {
             maxt = std::max(maxt, static_cast<double>(abs(t[i])));
         }

         return maxt;
    }


    /**
     * rescale every element of the tensor with the given factor
     */
    template<class T>
    void Tensor<T>::rescale(const double s) {
        for (decltype(t.size()) i = 0; i < t.size(); ++i) {
            t[i] /= s;
        }

    #ifdef DEBUG
        std::cout << "rescaled" << std::endl;
    #endif
    }


    /**
     * compute flat index (index in t) given all tensor indices as I = sum_i index(i)*base(i)
     */
    template <class T>
    uint Tensor<T>::flatindex(const std::vector<uint> index) const {
        uint tot_index = 0;
        for (decltype(order) i = 0; i < order; ++i) {
            tot_index += index[i] * base[i];
        }

        return tot_index;
    }


    /**
     * compute the index vector from a flatindex
     */
    template<class T>
    inline std::vector<uint> Tensor<T>::index(uint flatindex) const {
        std::vector<uint> index(order);
        auto rest = flatindex;

        for(uint i = order - 1; i >= 1; --i) {
            index[i] = rest / base[i];
            rest %= base[i];
        }

        index[0] = rest;

        return index;
    }


    /**
     * flatten tensor into a k-flattened matrix
     */
    template <class T>
    void Tensor<T>::flatten(const uint k, arma::Mat<T> &flatk) {
        // use one index as first matrix index and the combination of all other indices as the second
        flatk.set_size(dim[k], dimtot / dim[k]);

        // reshape complete tensor to a k-flattened matrix
        uint basecol = 0;
        // step-size for index k
        uint block = base[k];
        // dim[k] * block
        uint blocksize = base[k + 1]; // for any k (also k == order - 1)

        // iterate through total vector, starting with row=0 and col=0
        // fill row 0 of output matrix till base[k] is reached

        for (decltype(dimtot) index = 0; index < dimtot; index += blocksize) {
            // block-copy, the rhs matrix uses the t-vector storage as its own
            flatk.cols(basecol, basecol + block - 1)
                    //             ptr to mem, rows, cols, copy mem, strict?
                    = arma::Mat<T>(&t[index], block, dim[k], false, true).t();
            basecol += block;
        }
    }


    /**
     * flatten tensor into a matrix with backward and forward taken together
     */
    template <class T>
    void Tensor<T>::flatten_bf(const uint k, arma::Mat<T> &flatk) {
        int nrows = dim[k] * dim[k + 1];
        flatk.set_size(nrows, dimtot / nrows);

        // reshape complete tensor to a k-flattened matrix
        uint basecol = 0;
        // step-size for index k
        uint block = base[k];
        // dim[k + 1] * dim[k] * block
        uint blocksize = base[k + 2]; // for any k (also k == order - 1)

        // iterate through total vector, starting with row=0 and col=0
        // fill row 0 of output matrix till base[k] is reached

        for (decltype(dimtot) index = 0; index < dimtot; index += blocksize) {
            // block-copy, the rhs matrix uses the t-vector storage as its own
            flatk.cols(basecol, basecol + block - 1)
                    = arma::Mat<T>(&t[index], block, nrows, false, true).t();
            basecol += block;
        }
    }


    /**
     * inflate k-flattened matrix into tensor
     * using the block-copy algorithm, this is just the reverse of flattening
     */
    template <class T>
    void Tensor<T>::inflate(const uint k, const arma::Mat<T> &flatk) {
        if ((flatk.n_rows > dim[k]) || (flatk.n_cols != dimtot / dim[k])) {
            std::cerr << "Inflate error: inconsistent matrix size" << std::endl;
            exit(0);
        }

        if (flatk.n_rows < dim[k]) {
            reshape(k, flatk.n_rows);
        }

        uint basecol = 0;
        uint block = base[k];
        uint blocksize = base[k + 1]; // for any k (also k==order-1)

        for (decltype(dimtot) index = 0; index < dimtot; index += blocksize) {
            // block-copy, the rhs matrix uses the t-vector storage as its own
            // delicate statement as we fill a static matrix in the storage space of our tensor
            arma::Mat<T>(&t[index], block, dim[k], false, true) = flatk.cols(basecol, basecol + block - 1).t();
            basecol += block;
        }
    }


    /**
     * reduce the tensor to a lower rank
     */
    template <class T>
    void Tensor<T>::reduce(const uint k, const arma::Mat<T> &U, const uint D) {
        arma::Mat<T> flat_S;
        arma::Mat<T> reduced_S;

        flatten(k, flat_S);
        // reduce to D singular vectors
        reduced_S = U.cols(0, D - 1).t() * flat_S;
        inflate(k, reduced_S);
    }


    /**
     * make an SVD decomposition in one dimension, and reduce in the forward and backward directions
     */
    template <class T>
    void Tensor<T>::decompose_and_reduce(const uint k, const uint D) {
        if (dim[k] > D) {
            arma::Mat<T> U;
            svdkU(k, U);    // compute SVD wrt index k

            // Reduce index k and k+1
            reduce(k, U, D);
            reduce(k + 1, U, D);
        }
    }


    template <class T>
    void Tensor<T>::decompose_and_reduce(const uint k1, const uint k2, const uint D) {
        arma::Mat<T> U1, U2;

        if (dim[k1] > D) {
            svdkU(k1, U1);  // compute SVD wrt index k
        }

        if (dim[k2] > D) {
            svdkU(k2, U2);  // compute SVD wrt index k
        }

        if (dim[k1] > D) {
            // Reduce index k and k+1
            reduce(k1, U1, D);
            reduce(k1 + 1, U1, D);
        }
        if (dim[k2] > D) {
            // Reduce index k and k+1
            reduce(k2, U2, D);
            reduce(k2 + 1, U2, D);
        }
    }


    /**
     * compute singular vectors
     */
    template <class T>
    void Tensor<T>::svdkU(const uint k, arma::Mat<T> &U) {
        // flatten input tensor t in index k to matrix flatk
        arma::Mat<T> flatk;
        flatten(k, flatk);

        arma::Col<T> s;
        // use symmetrized system
        arma::Mat<T> flatH = flatk * flatk.t();

        arma::eig_sym(s, U, flatH);   // stored in ascending order in Armadillo; eigenvalues of a symmetric matrix
        U = arma::reverse(U, 1);      // reverse elements in each row

    #ifdef DEBUG
        std::cout << s << std::endl;
    #endif
    }
    
}

#endif // TENSOR_H
