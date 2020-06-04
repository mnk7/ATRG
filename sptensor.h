#ifndef SPTENSOR_H
#define SPTENSOR_H

#include <iostream>
#include <vector>
#include <algorithm>

#include <armadillo>
#include <omp.h>

typedef unsigned int uint;


namespace ATRG {

    template<class T>
    class SpTensor {
    public:
        SpTensor() : order(0), size(0) {}
        SpTensor(const std::vector<uint> dimensions);

        void reshape(const std::vector<uint> dimensions);
        void reshape(const uint index, const uint dimension);
        void reorder(const std::vector<uint> new_order_indices);

        void print(std::ostream &os);
        void print();

        T Frobnorm2();
        T Frobnorm();

        T max();
        void zero();
        void rescale(const T s);

        uint flatindex(const std::vector<uint> index, std::vector<uint> &base);
        uint flatindex(const std::vector<uint> index);
        std::vector<uint> multiindex(const uint flatindex, std::vector<uint> &base);
        std::vector<uint> multiindex(const uint flatindex);
        void flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::SpMat<T> &flat);
        void flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat);
        void inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, const arma::SpMat<T> &flat);
        void inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, const arma::Mat<T> &flat);

        uint get_size() const { return size; }
        uint get_order() const { return order; }
        uint get_dimensions(const uint index) const { return dimensions[index]; }
        std::vector<uint> get_dimensions() { return dimensions; }
        uint get_base(const uint i) const { return base[i]; }

        arma::SpMat<T>& data() { return t; }
        arma::SpMat<T> data_copy() { return t; };


        T operator()(const uint flatindex) {
            return t(flatindex);
        }

        void set(const uint flatindex, T value) {
            t(flatindex) = value;
        }

        T operator()(const std::vector<uint>& index) {
            return t(flatindex(index));
        }

        void set(const std::vector<uint>& index, T value) {
            t(flatindex(index)) = value;
        }

        friend std::ostream &operator<<(std::ostream &out, SpTensor<T> &t) {
            arma::SpMat<T> flat;
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
        arma::SpMat<T> t;
        // base factor for each index -> hypercube for elements of the indices
        std::vector<uint> base;
    };


    /* ============================== Definition ============================== */

    /**
     * lowest index runs fastest, highest index lowest (column major for matrices)
     */
    template <class T>
    SpTensor<T>::SpTensor(const std::vector<uint> dimensions) {
        reshape(dimensions);
    }


    /**
     * change the dimension of the tensor
     */
    template <class T>
    void SpTensor<T>::reshape(const std::vector<uint> dimensions) {
        order = dimensions.size();
        this->dimensions = dimensions;

        // last element filled with size (for efficiency of algorithms)
        base.resize(order + 1);
        base[0] = 1;
        for (decltype(order) i = 0; i < order; ++i) {
            base[i + 1] = base[i] * dimensions[i];
        }

        size = base[order];
        t.set_size(size);
    }


    /**
     * change dimension of index k
     */
    template <class T>
    void SpTensor<T>::reshape(const uint k, const uint dimension) {
        // modify dim[k] and base from index k onwards
        dimensions[k] = dimension;

        for (decltype(order) i = k; i < order; ++i) {
            base[i + 1] = base[i] * dimensions[i];
        }

        size = base[order];
        t.set_size(size);
    }


    /**
     * reorder the tensor indices
     */
    template <class T>
    void SpTensor<T>::reorder(const std::vector<uint> new_order_indices) {
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
        arma::SpMat<T> new_t(size, 1);

        for(decltype(new_order_indices.size()) i = 0; i < new_order_indices.size(); ++i) {
            dimensions[i] = old_dimensions[new_order_indices[i]];
        }

        base[0] = 1;
        for (decltype(order) i = 0; i < order; ++i) {
            base[i + 1] = base[i] * dimensions[i];
        }


        for(decltype(t.begin()) element = t.begin(); element != t.end(); ++element) {
            auto old_indices = multiindex(element.row(), old_base);
            decltype(old_indices) new_indices(old_indices.size());

            for(decltype(new_indices.size()) j = 0; j < new_indices.size(); ++j) {
                new_indices[j] = old_indices[new_order_indices[j]];
            }

            new_t(flatindex(new_indices)) = *element;
        }

        t = new_t;
    }


    /**
     * print the tensor in a readable way
     */
    template <class T>
    void SpTensor<T>::print(std::ostream &os) {
        std::cout << "Printing tensor:" << std::endl;

        for(uint i = 0; i < size; ++i) {
            auto ind = multiindex(i);

            os << "       [";

            for(decltype(ind.size()) j = 0; j < ind.size() - 1; ++j) {
                os << ind[j] << "\t";
            }

            os << ind.back() << "] = " << t(i) << std::endl;
        }

        std::cout << std::endl;
    }


    /**
     * print tensor to std::cout
     */
    template <class T>
    void SpTensor<T>::print() {
        print(std::cout);
    }


    /**
     * compute the square of the Frobenius norm of the tensor
     */
    template <class T>
    inline T SpTensor<T>::Frobnorm2() {
        T F = 0;

        t.for_each([&F](T &element) {F += arma::abs(element) * arma::abs(element);});

        return F;
    }
    
    
    /**
     * ocmpute the Frobenius norm of the tensor
     */
    template <class T>
    inline T SpTensor<T>::Frobnorm() {
        return std::sqrt(Frobnorm2());
    }


    /**
     * returns the largest element in the tensor
     */
    template <class T>
    inline T SpTensor<T>::max() {
         return t.max();
    }


    /**
     * reset tensor to zero
     */
    template <class T>
    inline void SpTensor<T>::zero() {
        t.clean(0);
    }


    /**
     * rescale every element of the tensor with the given factor
     */
    template <class T>
    inline void SpTensor<T>::rescale(const T s) {
        t *= s;
    }


    /**
     * compute flat index (index in t) given the indices of all modes
     */
    template <class T>
    inline uint SpTensor<T>::flatindex(const std::vector<uint> index, std::vector<uint> &base) {
        uint flatindex = 0;

        // the last entry in base contains the total number of elements
        for (decltype(base.size()) i = 0; i < base.size() - 1; ++i) {
            flatindex += index[i] * base[i];
        }

        return flatindex;
    }


    template <class T>
    inline uint SpTensor<T>::flatindex(const std::vector<uint> index) {
        return flatindex(index, base);
    }


    /**
     * compute the index vector from a flatindex
     * base has to contain the total number of elements in its last entry!
     */
    template <class T>
    inline std::vector<uint> SpTensor<T>::multiindex(uint flatindex, std::vector<uint> &base) {
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
    inline std::vector<uint> SpTensor<T>::multiindex(uint flatindex) {
        return multiindex(flatindex, base);
    }


    /**
     * Flatten the tensor with the given indices identifying the rows and the rest of the indices identifying the columns.
     */
    template <class T>
    inline void SpTensor<T>::flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::SpMat<T> &flat) {
        if(indices_rows.size() + indices_columns.size() != order) {
            std::cerr << "  In ATRG::SpTensor<T>::flatten: given more indices than the tensor has!" << std::endl;
            throw 0;
        }

        std::vector<uint> all_indices(indices_rows);
        all_indices.insert(all_indices.end(), indices_columns.begin(), indices_columns.end());

        // watch out for duplicates:
        auto all_indices_copy(all_indices);
        std::sort(all_indices_copy.begin(), all_indices_copy.end());
        auto last = std::unique(all_indices_copy.begin(), all_indices_copy.end());

        if(last != all_indices_copy.end()) {
            std::cerr << "  In ATRG::SpTensor<T>::flatten: duplicate indices given!" << std::endl;
            throw 0;
        }

        if(std::any_of(all_indices.begin(), all_indices.end(), [this](auto &index) {return index >= order || index < 0;})) {
            std::cerr << "  In ATRG::SpTensor<T>::flatten: requested non-existent index!" << std::endl;
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
            flat = t;
            flat.reshape(n_rows, n_cols);
            return;
        }

        flat.resize(n_rows, n_cols);

        /*
         * naive case: compute the position of every tensor element in the flat matrix
         */
        for(decltype(t.begin()) element = t.begin(); element != t.end(); ++element) {
            auto tensor_indices = multiindex(element.row());

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

            flat(row_index, col_index) = *element;
        }
    }


    template <class T>
    inline void SpTensor<T>::flatten(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, arma::Mat<T> &flat) {
        arma::SpMat<T> flat_sparse(flat);

        flatten(indices_rows, indices_columns, flat_sparse);
    }


    /**
     * inflate the tensor with the given indices identifying the rows and the rest of the indices identifying the columns
     */
    template <class T>
    inline void SpTensor<T>::inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, const arma::SpMat<T> &flat) {
        if(flat.n_elem != size) {
            std::cerr << "  In ATRG::SpTensor<T>::inflate: flat matrix has an incorrect size!" << std::endl;
            throw 0;
        }

        if(indices_rows.size() + indices_columns.size() != order) {
            std::cerr << "  In ATRG::SpTensor<T>::inflate: given more indices than the tensor has!" << std::endl;
            throw 0;
        }

        std::vector<uint> all_indices(indices_rows);
        all_indices.insert(all_indices.end(), indices_columns.begin(), indices_columns.end());

        // watch out for duplicates:
        auto all_indices_copy(all_indices);
        std::sort(all_indices_copy.begin(), all_indices_copy.end());
        auto last = std::unique(all_indices_copy.begin(), all_indices_copy.end());

        if(last != all_indices_copy.end()) {
            std::cerr << "  In ATRG::SpTensor<T>::inflate: duplicate indices given!" << std::endl;
            throw 0;
        }

        if(std::any_of(all_indices.begin(), all_indices.end(), [this](auto &index) {return index >= order || index < 0;})) {
            std::cerr << "  In ATRG::SpTensor<T>::inflate: requested non-existent index!" << std::endl;
        }


        /*
         * if both vectors of indices are sorted and the column vector picks up where the row vector ended,
         * we can use flat as it is.
         * e.g.: {0 1 2}, {3 4}
         */
        if(std::is_sorted(all_indices.begin(), all_indices.end())) {
            t = flat;
            t.reshape(size, 1);
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


        for(decltype(flat.begin()) element = flat.begin(); element != flat.end(); ++element) {
            std::vector<uint> tensor_indices(order);

            /*
             * get the tensor index in the ordering given by indices_rows
             */
            uint rest = element.row();
            for(int j = indices_rows.size() - 1; j >= 1; --j) {
                tensor_indices[indices_rows[j]] = rest / base_rows[j];
                rest %= base_rows[j];
            }
            tensor_indices[indices_rows[0]] = rest;

            rest = element.col();
            for(int j = indices_columns.size() - 1; j >= 1; --j) {
                tensor_indices[indices_columns[j]] = rest / base_cols[j];
                rest %= base_cols[j];
            }
            tensor_indices[indices_columns[0]] = rest;

            t(flatindex(tensor_indices)) = *element;
        }
    }


    template <class T>
    inline void SpTensor<T>::inflate(const std::vector<uint> &indices_rows, const std::vector<uint> &indices_columns, const arma::Mat<T> &flat) {
        arma::SpMat<T> flat_sparse(flat);

        inflate(indices_rows, indices_columns, flat_sparse);
    }
    
}

#endif // SPTENSOR_H
