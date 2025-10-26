#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

class CSR {
  private:
    std::vector<int> row_ptr;   // Row pointer
    std::vector<int> col_ind;   // Column indices
    std::vector<double> values; // Non-zero values
    int num_rows;
    int num_cols;
    int num_nonzeros;
  public:
    // Do not use default constructor
    // Use factory function to create CSR from file
    CSR() {}

    // Read the Matrix Market file and populate CSR format data structures
    // Matrix Market format: https://math.nist.gov/MatrixMarket/formats.html
    // Excerpt from the format:
    // %%MatrixMarket matrix coordinate real general
    // %=================================================================================
    // %
    // % This ASCII file represents a sparse MxN matrix with L 
    // % nonzeros in the following Matrix Market format:
    // %
    // % +----------------------------------------------+
    // % |%%MatrixMarket matrix coordinate real general | <--- header line
    // % |%                                             | <--+
    // % |% comments                                    |    |-- 0 or more comment lines
    // % |%                                             | <--+         
    // % |    M  N  L                                   | <--- rows, columns, entries
    // % |    I1  J1  A(I1, J1)                         | <--+
    // % |    I2  J2  A(I2, J2)                         |    |
    // % |    I3  J3  A(I3, J3)                         |    |-- L lines
    // % |        . . .                                 |    |
    // % |    IL JL  A(IL, JL)                          | <--+
    // % +----------------------------------------------+   
    // %
    // % Indices are 1-based, i.e. A(1,1) is the first element.
    // %
    // %=================================================================================
    // We do not assume that the Matrix Market file is sorted by row or column.
    static CSR CreateFromMatrixMarketFile(const std::string& filename) {
        CSR csr;
        std::cout << "Creating CSR from file: " << filename << "\n";
        // Read the file and populate csr.row_ptr, csr.col_ind, csr.values
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            std::cerr << "Error opening file: " << filename << "\n";
            assert(false);
            return csr;
        }

        // We are reading the file as follows:
        // - We read the first non-comment line to get the matrix dimensions and
        // number of non-zeros.
        // - We read subsequent lines to get the row indices, column indices,
        // and values of the non-zero elements. Since we are not assuming the
        // file is sorted, we will store the entries in an ordered_map, where
        // the key is a pair of (row, col) and the value is the matrix value.
        int64_t num_rows = 0;
        int64_t num_cols = 0;
        int64_t num_nonzeros = 0;
        // Skip comment lines and find the first non-comment line
        while (infile.peek() == '%') {
            infile.ignore(2048, '\n');
        }
        // Read matrix dimensions and number of non-zeros
        infile >> num_rows >> num_cols >> num_nonzeros;
        csr.num_rows = static_cast<int>(num_rows);
        csr.num_cols = static_cast<int>(num_cols);
        csr.num_nonzeros = static_cast<int>(num_nonzeros);
        std::cout << "Matrix dimensions: " << num_rows << " x " << num_cols
                  << " with " << num_nonzeros << " non-zeros.\n";
        // Now we read the non-zero entries
        std::map<std::pair<int, int>, double, std::less<std::pair<int, int>>> entries;
        for (uint64_t i = 0; i < num_nonzeros; ++i) {
            int row, col;
            double value;
            infile >> row >> col >> value;
            // Store the entry in the map (adjusting for 1-based indexing)
            entries[{row - 1, col - 1}] = value;
        }

        // Now we need to convert the map to CSR format
        csr.row_ptr.resize(num_rows + 1, 0);
        csr.col_ind.reserve(num_nonzeros);
        csr.values.reserve(num_nonzeros);

        for (const auto& entry : entries) {
            int row = entry.first.first;
            int col = entry.first.second;
            double value = entry.second;
            csr.col_ind.push_back(col);
            csr.values.push_back(value);
            csr.row_ptr[row + 1]++; // Increment the count of non-zeros in this row
        }

        // Compute the row_ptr array
        for (size_t i = 1; i < csr.row_ptr.size(); ++i) {
            csr.row_ptr[i] += csr.row_ptr[i - 1];
        }

        return csr;
    }

    // Get number of rows
    int GetNumRows() const {
        return num_rows;
    }

    // Get number of columns
    int GetNumCols() const {
        return num_cols;
    }

    // Get number of non-zeros
    int GetNumNonZeros() const {
        return num_nonzeros;
    }

    // Print the CSR representation (for debugging)
    void Print() const {
        for (int i = 0; i < GetNumRows(); ++i) {
            int row_start = row_ptr[i];
            int row_end = row_ptr[i + 1];
            for (int j = row_start; j < row_end; ++j) {
                std::cout << "(" << i+1 << ", " << col_ind[j]+1 << ") = " << values[j] << "\n";
                printf("Row %d: Col %d, Value %f\n", i+1, col_ind[j]+1, values[j]);
            }
        }
    }

    // Perform Sparse Matrix-Vector Multiplication (SpMV)
    std::vector<double> SpMV(const std::vector<double>& x) const {
        assert(x.size() == GetNumCols());
        std::vector<double> y(GetNumRows(), 0.0);
        #pragma omp parallel  // Enable OpenMP parallelization
        {
            #pragma omp for nowait
            for (int i = 0; i < GetNumRows(); ++i) {
                int row_start = row_ptr[i];
                int row_end = row_ptr[i + 1];
                for (int j = row_start; j < row_end; ++j) {
                    y[i] += values[j] * x[col_ind[j]];
                }
            }
            std::cout << "Thread " << omp_get_thread_num() << " completed SpMV computation.\n";
        }
        return y;
    }

};  // class CSR

int main(int argc, char** argv) {
    std::cout << "Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n";

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <matrix_file>\n";
        return 1;
    }

    std::string matrix_file = argv[1];
    CSR csr_matrix = CSR::CreateFromMatrixMarketFile(matrix_file);

    std::cout << "Num cols: " << csr_matrix.GetNumCols() << "\n";
    std::vector<double> x(csr_matrix.GetNumCols(), 1.0); // Input vector of all ones
    std::vector<double> y = csr_matrix.SpMV(x);

    std::cout << "Result of SpMV:\n";
    for (size_t i = 0; i < y.size(); ++i) {
        std::cout << "y[" << i+1 << "] = " << y[i] << "\n";
    }

    return 0;
}