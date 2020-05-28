#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
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
        void reorder(const std::vector<uint> new_order_indices);

        void print(std::ostream &os);
        void print();

        T Frobnorm2();
        T Frobnorm();

        T max();
        void fill(const T value);
        void rescale(const T s);

        uint flatindex(const std::vector<uint> index, std::vector<uint> &base);
        uint flatindex(const std::vector<uint> index);
        std::vector<uint> multiindex(const uint flatindex, std::vector<uint> &base);
        std::vector<uint> multiindex(const uint flatindex);
        void flatten(const uint index, arma::Mat<T> &flat);
        void flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat);
        void inflate(const uint index, const arma::Mat<T> &flat);
        void inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat);

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
     * reorder the tensor indices
     */
    template <class T>
    void Tensor<T>::reorder(const std::vector<uint> new_order_indices) {
        auto new_order_indices_copy(new_order_indices);
        std::sort(new_order_indices_copy.begin(), new_order_indices_copy.end());
        auto last = std::unique(new_order_indices_copy.begin(), new_order_indices_copy.end());

        if(last != new_order_indices_copy.end()) {
            std::cerr << "  In ATRG::Tensor<T>::reorder: duplicate indices given!" << std::endl;
            throw 0;
        }

        if(std::any_of(new_order_indices_copy.begin(), new_order_indices_copy.end(), [this](auto &index) {return index >= order || index < 0;})) {
            std::cerr << "  In ATRG::Tensor<T>::reorder: requested non-existent index!" << std::endl;
            throw 0;
        }

        auto old_dimensions = dimensions;
        auto old_base = base;
        std::valarray<T> new_t(size);

        for(decltype(new_order_indices.size()) i = 0; i < new_order_indices.size(); ++i) {
            dimensions[i] = old_dimensions[new_order_indices[i]];
        }

        base[0] = 1;
        for (decltype(order) i = 0; i < order; ++i) {
            base[i + 1] = base[i] * dimensions[i];
        }

        #pragma omp parallel for
        for(decltype(size) i = 0; i < size; ++i) {
            auto new_indices = multiindex(i);
            decltype(new_indices) old_indices(new_indices.size());

            for(decltype(new_indices.size()) j = 0; j < new_indices.size(); ++j) {
                old_indices[new_order_indices[j]] = new_indices[j];
            }

            new_t[i] = t[flatindex(old_indices, old_base)];
        }

        t = new_t;
    }


    /**
     * print the tensor in a readable way
     */
    template <class T>
    void Tensor<T>::print(std::ostream &os) {
        std::cout << "Printing tensor:" << std::endl;

        for(uint i = 0; i < size; ++i) {
            auto ind = multiindex(i);

            os << "\t[";

            for(decltype(ind.size()) j = 0; j < ind.size() - 1; ++j) {
                os << ind[j] << "\t";
            }

            os << ind.back() << "] = " << t[i] << std::endl;
        }

        std::cout << std::endl;
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
     * compute flat index (index in t) given the indices of all modes
     */
    template <class T>
    inline uint Tensor<T>::flatindex(const std::vector<uint> index, std::vector<uint> &base) {
        uint flatindex = 0;

        // the last entry in base contains the total number of elements
        for (decltype(base.size()) i = 0; i < base.size() - 1; ++i) {
            flatindex += index[i] * base[i];
        }

        return flatindex;
    }


    template <class T>
    inline uint Tensor<T>::flatindex(const std::vector<uint> index) {
        return flatindex(index, base);
    }


    /**
     * compute the index vector from a flatindex
     * base has to contain the total number of elements in its last entry!
     */
    template <class T>
    inline std::vector<uint> Tensor<T>::multiindex(uint flatindex, std::vector<uint> &base) {
        std::vector<uint> index(base.size() - 1);
        auto rest = flatindex;

        for(uint i = base.size() - 2; i >= 1; --i) {
            index[i] = rest / base[i];
            rest %= base[i];
        }

        index[0] = rest;

        return index;
    }


    template <class T>
    inline std::vector<uint> Tensor<T>::multiindex(uint flatindex) {
        return multiindex(flatindex, base);
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
     * Flatten the tensor with the given indices identifying the rows and the rest of the indices identifying the columns.
     *
     * !!! This function may return the tensors own memory, so its integrity is not guarantied after calling it !!!
     */
    template <class T>
    inline void Tensor<T>::flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat) {
        if(indices_rows.size() + indices_columns.size() != order) {
            std::cerr << "  In ATRG::Tensor<T>::flatten: given more indices than the tensor has!" << std::endl;
            throw 0;
        }

        std::vector<uint> all_indices(indices_rows);
        all_indices.insert(all_indices.end(), indices_columns.begin(), indices_columns.end());

        // watch out for duplicates:
        auto all_indices_copy(all_indices);
        std::sort(all_indices_copy.begin(), all_indices_copy.end());
        auto last = std::unique(all_indices_copy.begin(), all_indices_copy.end());

        if(last != all_indices_copy.end()) {
            std::cerr << "  In ATRG::Tensor<T>::flatten: duplicate indices given!" << std::endl;
            throw 0;
        }

        if(std::any_of(all_indices.begin(), all_indices.end(), [this](auto &index) {return index >= order || index < 0;})) {
            std::cerr << "  In ATRG::Tensor<T>::flatten: requested non-existent index!" << std::endl;
            throw 0;
        }

        uint n_rows = 1;
        uint n_cols = 1;

        // create bases to compute the indices of the tensor elements in the flat matrix
        // the last entry contains the total number of elements
        std::vector<uint> base_rows(indices_rows.size() + 1, 1);
        std::vector<uint> base_cols(indices_columns.size() + 1, 1);

        for(decltype(indices_rows.size()) i = 0; i < indices_rows.size(); ++i) {
            n_rows *= dimensions[indices_rows[i]];
            base_rows[i + 1] = base_rows[i] * dimensions[indices_rows[i]];
        }

        for(decltype(indices_columns.size()) i = 0; i < indices_columns.size(); ++i) {
            n_cols *= dimensions[indices_columns[i]];
            base_cols[i + 1] = base_cols[i] * dimensions[indices_columns[i]];
        }

        flat.resize(n_rows, n_cols);

        /*
         * if both vectors of indices are sorted and the column vector picks up where the row vector ended,
         * we can use t as it is.
         * e.g.: {0 1 2}, {3 4}
         */
        if(std::is_sorted(all_indices.begin(), all_indices.end())) {
            flat = arma::Mat<T>(&t[0], n_rows, n_cols, false, true);
            return;
        }


        /*
         * naive case: compute the position of every tensor element in the flat matrix
         */
        #pragma omp parallel for
        for(decltype(size) i = 0; i < size; ++i) {
            auto tensor_indices = multiindex(i);

            uint row_index = 0;
            uint col_index = 0;

            /*
             * get the tensor indices in the ordering given by indices_rows and compute the position in the matrix row with the base from above
             */
            for(decltype(indices_rows.size()) j = 0; j < indices_rows.size(); ++j) {
                row_index += tensor_indices[indices_rows[j]] * base_rows[j];
            }

            for(decltype(indices_columns.size()) j = 0; j < indices_columns.size(); ++j) {
                col_index += tensor_indices[indices_columns[j]] * base_cols[j];
            }

            flat(row_index, col_index) = t[i];
        }

    }


    /**
     * inflate k-flattened matrix into tensor
     * using the block-copy algorithm, this is just the reverse of flattening
     */
    template <class T>
    inline void Tensor<T>::inflate(const uint index, const arma::Mat<T> &flat) {
        if ((flat.n_rows > dimensions[index]) || (flat.n_cols != size / dimensions[index])) {
            std::cerr << "  In ATRG::Tensor<T>::inflate: dimensions of flat matrix and tensor don't match!" << std::endl;
            throw 0;
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
    inline void Tensor<T>::inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat) {
        if(indices_rows.size() + indices_columns.size() != order) {
            std::cerr << "  In ATRG::Tensor<T>::inflate: given more indices than the tensor has!" << std::endl;
            throw 0;
        }

        std::vector<uint> all_indices(indices_rows);
        all_indices.insert(all_indices.end(), indices_columns.begin(), indices_columns.end());

        // watch out for duplicates:
        auto all_indices_copy(all_indices);
        std::sort(all_indices_copy.begin(), all_indices_copy.end());
        auto last = std::unique(all_indices_copy.begin(), all_indices_copy.end());

        if(last != all_indices_copy.end()) {
            std::cerr << "  In ATRG::Tensor<T>::inflate: duplicate indices given!" << std::endl;
            throw 0;
        }

        if(std::any_of(all_indices.begin(), all_indices.end(), [this](auto &index) {return index >= order || index < 0;})) {
            std::cerr << "  In ATRG::Tensor<T>::inflate: requested non-existent index!" << std::endl;
        }


        /*
         * if both vectors of indices are sorted and the column vector picks up where the row vector ended,
         * we can use flat as it is.
         * e.g.: {0 1 2}, {3 4}
         */
        if(std::is_sorted(all_indices.begin(), all_indices.end())) {
            arma::Mat<T>(&t[0], flat.n_rows, flat.n_cols, false, true) = flat;
            return;
        }


        /*
         * naive case: compute the position of every tensor element in the flat matrix
         */
        // create bases to compute the indices of the tensor elements in the flat matrix
        // the last entry contains the total number of elements
        std::vector<uint> base_rows(indices_rows.size() + 1, 1);
        std::vector<uint> base_cols(indices_columns.size() + 1, 1);

        for(decltype(indices_rows.size()) i = 0; i < indices_rows.size(); ++i) {
            base_rows[i + 1] = base_rows[i] * dimensions[indices_rows[i]];
        }

        for(decltype(indices_columns.size()) i = 0; i < indices_columns.size(); ++i) {
            base_cols[i + 1] = base_cols[i] * dimensions[indices_columns[i]];
        }


        #pragma omp parallel for
        for(decltype(size) i = 0; i < size; ++i) {
            auto tensor_indices = multiindex(i);

            uint row_index = 0;
            uint col_index = 0;

            /*
             * get the tensor indices in the ordering given by indices_rows and compute the position in the matrix row with the base from above
             */
            for(decltype(indices_rows.size()) j = 0; j < indices_rows.size(); ++j) {
                row_index += tensor_indices[indices_rows[j]] * base_rows[j];
            }

            for(decltype(indices_columns.size()) j = 0; j < indices_columns.size(); ++j) {
                col_index += tensor_indices[indices_columns[j]] * base_cols[j];
            }

            t[i] = flat(row_index, col_index);
        }
    }
    
}

#endif // TENSOR_H
