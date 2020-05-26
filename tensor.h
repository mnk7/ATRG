#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>
#include <algorithm>

#include <armadillo>
#include <omp.h>

typedef unsigned int uint;


namespace ATRG {

    template<class T>
    class Tensor {
    public:
        Tensor() : order(0), size(0) {}
        Tensor(const std::vector<uint> dimensions);

        void reshape(const std::vector<uint> dimensions);
        void reshape(const uint index, const uint dimension);

        void print(std::ofstream &os);
        void print();

        T Frobnorm2();
        T Frobnorm();

        T max();
        void fill(const T value);
        void rescale(const T s);

        uint flatindex(const std::vector<uint> index);
        std::vector<uint> multiindex(const uint flatindex);
        void flatten(const uint index, arma::Mat<T> &flat);
        void flatten(const std::vector<uint> indices, arma::Mat<T> &flat);
        void inflate(const uint index, const arma::Mat<T> &flat);
        void inflate(const std::vector<uint> indices, arma::Mat<T> &flat);

        uint get_size() const { return size; }
        uint get_order() const { return order; }
        uint get_dimensions(const uint index) const { return dimensions[index]; }
        std::vector<uint> get_dimensions() { return dimensions; }
        uint get_base(const uint i) const { return base[i]; }

        T* data() { return &t[0]; }


        T& operator()(const uint flatindex) {
            return t[flatindex];
        }
        const T& operator()(const uint flatindex) const {
            return t[flatindex];
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
        uint size;
        // dimensions of each index
        std::vector<uint> dimensions;
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
    Tensor<T>::Tensor(const std::vector<uint> dimensions) {
        reshape(dimensions);
    }


    /**
     * change the dimension of the tensor
     */
    template <class T>
    void Tensor<T>::reshape(const std::vector<uint> dimensions) {
        order = dimensions.size();
        this->dimensions = dimensions;

        // last element filled with size (for efficiency of algorithms)
        base.resize(order + 1);
        base[0] = 1;
        for (decltype(order) i = 0; i < order; ++i) {
            base[i + 1] = base[i] * dimensions[i];
        }

        size = base[order];
        t.resize(size);
    }


    /**
     * change dimension of index k
     */
    template <class T>
    void Tensor<T>::reshape(const uint k, const uint dimension) {
        // modify dim[k] and base from index k onwards
        dimensions[k] = dimension;

        for (decltype(order) i = k; i < order; ++i) {
            base[i + 1] = base[i] * dimensions[i];
        }

        size = base[order];
        t.resize(size);
    }


    /**
     * print the tensor in a readable way
     */
    template <class T>
    void Tensor<T>::print(std::ofstream &os) {
        for(uint i = 0; i < size; ++i) {
            auto ind = multiindex(i);

            os << "[";

            for(uint j = 0; j < ind.size(); ++j) {
                os << ind[j] << "\t";
            }

            os << "] = " << t[i] << std::endl;
        }
    }


    /**
     * print tensor to std::cout
     */
    template <class T>
    void Tensor<T>::print() {
        print(std::cout);
    }


    /**
     * compute the square of the Frobenius norm of the tensor
     */
    template <class T>
    inline T Tensor<T>::Frobnorm2() {
        T F = 0;

        // std::norm returns the square!
        std::for_each(std::begin(t), std::end(t), [&F](T &element) {F += std::norm(element);});

        return F;
    }
    
    
    /**
     * ocmpute the Frobenius norm of the tensor
     */
    template <class T>
    inline T Tensor<T>::Frobnorm() {
        return std::sqrt(Frobnorm2());
    }


    /**
     * returns the largest element in the tensor
     */
    template <class T>
    inline T Tensor<T>::max() {
         return std::max(std::begin(t), std::end(t), [](T &a, T &b) {return std::abs(a) > std::abs(b);});
    }


    /**
     * reset tensor to zero
     */
    template <class T>
    inline void Tensor<T>::fill(const T value) {
        std::fill(std::begin(t), std::end(t), value);
    }


    /**
     * rescale every element of the tensor with the given factor
     */
    template <class T>
    inline void Tensor<T>::rescale(const T s) {
        t *= s;
    }


    /**
     * compute flat index (index in t) given all tensor indices as I = sum_i index(i)*base(i)
     */
    template <class T>
    inline uint Tensor<T>::flatindex(const std::vector<uint> index) {
        uint flatindex = 0;

        for (decltype(order) i = 0; i < order; ++i) {
            flatindex += index[i] * base[i];
        }

        return flatindex;
    }


    /**
     * compute the index vector from a flatindex
     */
    template <class T>
    inline std::vector<uint> Tensor<T>::multiindex(uint flatindex) {
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
    inline void Tensor<T>::flatten(const uint index, arma::Mat<T> &flat) {
        // use one index as first matrix index and the combination of all other indices as the second
        flat.set_size(dimensions[index], size / dimensions[index]);

        // reshape complete tensor to a k-flattened matrix
        uint basecol = 0;
        // step-size for index k
        uint block = base[index];
        // dim[k] * block
        uint blocksize = base[index + 1]; // for any k (also k == order - 1)

        // iterate through total vector, starting with row = 0 and col = 0
        // fill row 0 of output matrix till base[k] is reached

        for (decltype(size) i = 0; i < size; i += blocksize) {
            // block-copy, the rhs matrix uses the t-vector storage as its own
            flat.cols(basecol, basecol + block - 1)
                    //             ptr to mem, rows, cols, copy mem, strict?
                    = arma::Mat<T>(&t[i], block, dimensions[index], false, true).t();
            basecol += block;
        }
    }


    /**
     * flatten the tensor with the given indices identifying the rows and the rest of the indices identifying the columns
     */
    template <class T>
    inline void Tensor<T>::flatten(std::vector<uint> indices, arma::Mat<T> &flat) {
        auto n_rows = 1;
        auto n_cols = 1;

        for(uint i = 0; i < order; ++i) {
            if(std::any_of(indices.begin(), indices.end(), [&i](uint index) {return index == i;})) {
                n_rows *= dimensions[i];
            } else {
                n_cols *= dimensions[i];
            }
        }

        flat.resize(n_rows, n_cols);
    }


    /**
     * inflate k-flattened matrix into tensor
     * using the block-copy algorithm, this is just the reverse of flattening
     */
    template <class T>
    inline void Tensor<T>::inflate(const uint index, const arma::Mat<T> &flat) {
        if ((flat.n_rows > dimensions[index]) || (flat.n_cols != size / dimensions[index])) {
            std::cerr << "  In ATRG::Tensor<T>::inflate: dimensions of flat matrix and tensor don't match!" << std::endl;
            exit(0);
        }

        if (flat.n_rows < dimensions[index]) {
            reshape(index, flat.n_rows);
        }

        uint basecol = 0;
        uint block = base[index];
        uint blocksize = base[index + 1]; // for any k (also k==order-1)

        for (decltype(size) i = 0; i < size; i += blocksize) {
            // block-copy, the rhs matrix uses the t-vector storage as its own
            // delicate statement as we fill a static matrix in the storage space of our tensor
            arma::Mat<T>(&t[i], block, dimensions[index], false, true) = flat.cols(basecol, basecol + block - 1).t();
            basecol += block;
        }
    }


    /**
     * inflate the tensor with the given indices identifying the rows and the rest of the indices identifying the columns
     */
    template <class T>
    inline void Tensor<T>::inflate(std::vector<uint> indices, arma::Mat<T> &flat) {

    }
    
}

#endif // TENSOR_H
