#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <valarray>
#include <algorithm>

#include <armadillo>

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
        T min();
        void zero();
        void rescale(const T s);

        uint flatindex(const std::vector<uint> index, std::vector<uint> &base);
        uint flatindex(const std::vector<uint> index);
        std::vector<uint> multiindex(const uint flatindex, std::vector<uint> &base);
        std::vector<uint> multiindex(const uint flatindex);
        void flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::SpMat<T> &flat);
        void flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat);
        void inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::SpMat<T> &flat);
        void inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat);

        uint get_size() const { return size; }
        uint get_order() const { return order; }
        uint get_dimensions(const uint index) const { return dimensions[index]; }
        std::vector<uint> get_dimensions() { return dimensions; }
        uint get_base(const uint i) const { return base[i]; }

        std::valarray<T>& data() { return t; }
        arma::Mat<T> data_copy() { return arma::Mat<T>(&t[0], size, 1, false, true); };


        T& operator()(const uint flatindex) {
            return t[flatindex];
        }

        const T& operator()(const uint flatindex) const {
            return t[flatindex];
        }

        void set(const uint flatindex, T value) {
            t[flatindex] = value;
        }

        T& operator()(const std::vector<uint>& index) {
            return t[flatindex(index)];
        }

        const T& operator()(const std::vector<uint>& index) const {
            return t[flatindex(index)];
        }

        void set(const std::vector<uint>& index, T value) {
            t[flatindex(index)] = value;
        }

        friend std::ostream &operator<<(std::ostream &out, Tensor<T> &t) {
            out << t.data_copy();
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

        void flatten_single_index(uint k, arma::Mat<T> &flat);
        void inflate_single_index(uint k, arma::Mat<T> &flat);
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
        zero();
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
        zero();
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


        // flatten and inflate single indices until we reach the desired order
        std::vector<uint> current_order_indices(new_order_indices);
        std::iota(current_order_indices.begin(), current_order_indices.end(), 0);

        // the left over index should be in the correct position
        for(decltype(new_order_indices.size()) i = 0; i < new_order_indices.size() - 1; ++i) {
            if(new_order_indices[i] != current_order_indices[i]) {
                // move the index at new_order_indices[i] to position i:
                // find the curren position of that index:
                uint current_position_index = current_order_indices.size();
                for(uint j = i + 1; j < current_order_indices.size(); ++j) {
                    if(current_order_indices[j] == new_order_indices[i]) {
                        current_position_index = j;
                        break;
                    }
                }

                arma::Mat<T> flat;
                flatten_single_index(current_position_index, flat);

                current_order_indices.erase(current_order_indices.begin() + current_position_index);
                current_order_indices.insert(current_order_indices.begin() + i, new_order_indices[i]);

                auto dimension = dimensions[current_position_index];
                dimensions.erase(dimensions.begin() + current_position_index);
                dimensions.insert(dimensions.begin() + i, dimension);

                for (decltype(order) k = i; k < order; ++k) {
                    base[k + 1] = base[k] * dimensions[k];
                }

                inflate_single_index(i, flat);
            }
        }
    }


    /**
     * print the tensor in a readable way
     */
    template <class T>
    void Tensor<T>::print(std::ostream &os) {
        std::cout << "Printing tensor:" << std::endl;

        for(uint i = 0; i < size; ++i) {
            auto ind = multiindex(i);

            os << "       [";

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

        std::for_each(t.begin(), t.end(), [&F](T &element) {F += std::abs(element) * std::abs(element);});

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
         return t.max();
    }


    /**
     * returns the smalles element in the tensor
     */
    template <class T>
    inline T Tensor<T>::min() {
         return t.min();
    }


    /**
     * reset tensor to zero
     */
    template <class T>
    inline void Tensor<T>::zero() {
        rescale(0);
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

        for(int i = base.size() - 2; i >= 1; --i) {
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


    template <class T>
    inline void Tensor<T>::flatten_single_index(uint k, arma::Mat<T> &flat) {
        flat.set_size(dimensions[k], size / dimensions[k]);

        uint basecol = 0;

        for(decltype(size) i = 0; i < size; i += base[k + 1]) {
            // get all elements with the same value in the index above the singled out index
            // all indices smaller than the index above the singled out index are now in the row
            // the index we want to single out is in the column index -> transpose
            flat.cols(basecol, basecol + base[k] - 1) = arma::Mat<T>(&t[i], base[k], dimensions[k], false, true).t();

            basecol += base[k];
        }
    }


    template <class T>
    inline void Tensor<T>::inflate_single_index(uint k, arma::Mat<T> &flat) {
        uint basecol = 0;

        for(decltype(size) i = 0; i < size; i += base[k + 1]) {
            arma::Mat<T>(&t[i], base[k], dimensions[k], false, true) = flat.cols(basecol, basecol + base[k] - 1).t();

            basecol += base[k];
        }
    }


    /**
     * Flatten the tensor with the given indices identifying the rows and the rest of the indices identifying the columns.
     */
    template <class T>
    inline void Tensor<T>::flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::SpMat<T> &flat) {
        arma::Mat<T> flat_dense(flat);

        flatten(indices_rows, indices_columns, flat_dense);
    }


    template <class T>
    inline void Tensor<T>::flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat) {
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

        /*
         * if both vectors of indices are sorted and the column vector picks up where the row vector ended,
         * we can use t as it is.
         * e.g.: {0 1 2}, {3 4}
         */
        if(std::is_sorted(all_indices.begin(), all_indices.end())) {
            // use the tensor's internal memory'
            flat = arma::Mat<T>(&t[0], n_rows, n_cols, false, true);
            return;
        }


        /*
         * flatten only in one direction and the other indices are sorted
         */
        if(indices_columns.size() == 1 && std::is_sorted(indices_rows.begin(), indices_rows.end())) {
            flat.set_size(n_cols, n_rows);
            flatten_single_index(indices_columns.back(), flat);
            flat = flat.t();
            return;
        }

        if(indices_rows.size() == 1 && std::is_sorted(indices_columns.begin(), indices_columns.end())) {
            flat.set_size(n_rows, n_cols);
            flatten_single_index(indices_rows.back(), flat);
            return;
        }



        auto old_dimensions = dimensions;
        auto old_t = t;

        reorder(all_indices);
        // use the tensor's internal memory'
        flat = arma::Mat<T>(&t[0], n_rows, n_cols, false, true);

        reshape(old_dimensions);
        t = old_t;
    }


    /**
     * inflate the tensor with the given indices identifying the rows and the rest of the indices identifying the columns
     */
    template <class T>
    inline void Tensor<T>::inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::SpMat<T> &flat) {
        arma::Mat<T> flat_dense(flat);

        inflate(indices_rows, indices_columns, flat_dense);
    }


    template <class T>
    inline void Tensor<T>::inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat) {
        if(flat.n_elem != size) {
            std::cerr << "  In ATRG::Tensor<T>::inflate: flat matrix has an incorrect size!" << std::endl;
            throw 0;
        }

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
            arma::Mat<T>(&t[0], size, 1, false, true) = flat;
            return;
        }



        /*
         * flatten in one direction only and the other indices are ordered
         */
        if(indices_columns.size() == 1 && std::is_sorted(indices_rows.begin(), indices_rows.end())) {
            flat = flat.t();
            inflate_single_index(indices_columns.back(), flat);
            flat.set_size(0);
            return;
        }

        if(indices_rows.size() == 1 && std::is_sorted(indices_columns.begin(), indices_columns.end())) {
            inflate_single_index(indices_rows.back(), flat);
            flat.set_size(0);
            return;
        }



        auto old_dimensions = dimensions;
        // prepare the tensor so that we can move the flat matrix to the internal memory
        auto new_dimensions = dimensions;
        // figure out how to reorder the tensor to get the original index order
        auto all_indices_new_order = all_indices;

        for(decltype(all_indices.size()) i = 0; i < all_indices.size(); ++i) {
            new_dimensions[i] = old_dimensions[all_indices[i]];

            all_indices_new_order[all_indices[i]] = i;
        }

        reshape(new_dimensions);
        arma::Mat<T>(&t[0], size, 1, false, true) = flat;

        reorder(all_indices_new_order);
    }
    
}

#endif // TENSOR_H
