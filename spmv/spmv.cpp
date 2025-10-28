#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#if ENABLE_PICKLEDEVICE==1
#pragma message("Compiling with Pickle device")
#include "pickle_device_manager.h"
#include "pickle_job.h"
#else
#pragma message("NOT compiling with Pickle device")
#endif

#if ENABLE_GEM5==1
#pragma message("Compiling with gem5 instructions")
#include <gem5/m5ops.h>
#include "m5_mmap.h"
#endif // ENABLE_GEM5

#if ENABLE_PICKLEDEVICE==1
std::unique_ptr<PickleDeviceManager> pdev(new PickleDeviceManager());
uint64_t* UCPage = NULL;
uint64_t* PerfPage = NULL;
#endif

class CSR {
  private:
    std::vector<int> row_ptr;   // Row pointer
    std::vector<int> col_ind;   // Column indices
    std::vector<double> values; // Non-zero values
    size_t num_rows;
    size_t num_cols;
    size_t num_nonzeros;
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
        size_t num_rows = 0;
        size_t num_cols = 0;
        size_t num_nonzeros = 0;
        // Skip comment lines and find the first non-comment line
        while (infile.peek() == '%') {
            infile.ignore(2048, '\n');
        }
        // Read matrix dimensions and number of non-zeros
        infile >> num_rows >> num_cols >> num_nonzeros;
        csr.num_rows = num_rows;
        csr.num_cols = num_cols;
        csr.num_nonzeros = num_nonzeros;
        std::cout << "Matrix dimensions: " << num_rows << " x " << num_cols
                  << " with " << num_nonzeros << " non-zeros.\n";
        // Now we read the non-zero entries
        std::map<std::pair<int, int>, double, std::less<std::pair<int, int>>> entries;
        for (size_t i = 0; i < num_nonzeros; ++i) {
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
    size_t GetNumRows() const {
        return num_rows;
    }

    // Get number of columns
    size_t GetNumCols() const {
        return num_cols;
    }

    // Get number of non-zeros
    size_t GetNumNonZeros() const {
        return num_nonzeros;
    }

    // Get sum of all non-zero values
    double GetSumOfValues() const {
        double sum = 0.0;
        for (const auto& val : values) {
            sum += val;
        }
        return sum;
    }

    // Print the CSR representation (for debugging)
    void Print() const {
        for (size_t i = 0; i < GetNumRows(); ++i) {
            int row_start = row_ptr[i];
            int row_end = row_ptr[i + 1];
            for (int j = row_start; j < row_end; ++j) {
                printf(
                    "(Row %d: Col %d): Value %f\n",
                    static_cast<int>(i+1), col_ind[j]+1, values[j]
                );
            }
        }
    }

    // Perform Sparse Matrix-Vector Multiplication (SpMV)
    // Params:
    //   x: the x vector in y=Ax
    //   try_using_pickle_prefetcher: if set to True, and if the hardware has Pickle device, we will
    // use if; not using the Pickle device otherwise.
    std::vector<double> SpMV(const std::vector<double>& x, bool try_using_pickle_prefetcher) const {
        assert(x.size() == GetNumCols());
        if (try_using_pickle_prefetcher) {
            // If we use pickle prefetcher, we need to do the following steps,
            // Step 1. Gather the prefetcher specs
            uint64_t use_pdev = 0;
            uint64_t prefetch_distance = 0;
#if ENABLE_PICKLEDEVICE==1
            PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
            use_pdev = specs.availability;
            prefetch_distance = specs.prefetch_distance;
#endif
            std::cout << "Use pdev: " << use_pdev << "; Prefetch distance: " << prefetch_distance << std::endl;
            // Step 2. Construct the dependency graph
            // row_ptr -> col_idx -> values and y
            // Step 3. Send the graph to the prefetcher
        }
        std::cout << "ROI Start" << std::endl;
#if ENABLE_GEM5==1
        m5_exit_addr(0); // exit 1, 3
#endif // ENABLE_GEM5
        std::vector<double> y(GetNumRows(), 0.0);
        std::cout << "Starting SpMV computation using " << omp_get_max_threads() << " threads.\n";
        #pragma omp parallel  // Enable OpenMP parallelization
        {
            #pragma omp for nowait
            for (size_t i = 0; i < GetNumRows(); ++i) {
                int row_start = row_ptr[i];
                int row_end = row_ptr[i + 1];
                for (int j = row_start; j < row_end; ++j) {
                    y[i] += values[j] * x[col_ind[j]];
                }
            }
            //std::cout << "Thread " << omp_get_thread_num() << " completed SpMV computation.\n";
        }
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 2, 4
#endif // ENABLE_GEM5
        std::cout << "ROI end" << std::endl;
        return y;
    }

};  // class CSR

// Function to compare two double values with a tolerance rate
// Returns true if they are equal within the tolerance, false otherwise
// Tolerance is defined as tol_rate * expected_value
bool ExpectEqual(double expected_value, double actual_value, double tol_rate = 1e-6) {
    const double max_diff_allowed = expected_value * tol_rate;
    if (std::abs(expected_value - actual_value) > max_diff_allowed) {
        printf(
            "tol_rate = %.8f, Expected %.8f but got %.8f\n",
            tol_rate, actual_value, expected_value
        );
        return false;
    }
    return true;
}

// Benchmarking function for SpMV
// Runs SpMV twice:
// - iter 1 is a warm-up run performing A * x_1 where x_1 is a vector of all ones
// - iter 2 is the actual benchmark performing A * x_5 where x_5 is a vector of all fives
// Params:
// - A: The CSR matrix
// Returns:
// - true if the benchmark passes validation, false otherwise
bool BenchmarkSpMV(const CSR& A) {
    const double A_element_wise_sum = A.GetSumOfValues();
    std::vector<double> x1(A.GetNumCols(), 1.0); // Input vector of all ones
    std::vector<double> x2(A.GetNumCols(), 5.0); // Input vector of all fives

    // First iteration of SpMV
    const double y1_time_start = omp_get_wtime();
    std::vector<double> y1 = A.SpMV(x1, false); // Warm-up run
    const double y1_time_end = omp_get_wtime();
    std::cout << "Warm-up SpMV time: " << (y1_time_end - y1_time_start) << " seconds.\n";

    // Second iteration of SpMV
    const double y2_time_start = omp_get_wtime();
    std::vector<double> y2 = A.SpMV(x2, true); // Run twice for benchmarking
    const double y2_time_end = omp_get_wtime();
    std::cout << "Benchmark SpMV time: " << (y2_time_end - y2_time_start) << " seconds.\n";

    // Validate results
    double y1_element_wise_sum = 0.0;
    for (const auto& val : y1) {
        y1_element_wise_sum += val;
    }
    assert(ExpectEqual(A_element_wise_sum, y1_element_wise_sum));
    double y2_element_wise_sum = 0.0;
    for (const auto& val : y2) {
        y2_element_wise_sum += val;
    }
    assert(ExpectEqual(A_element_wise_sum * 5.0, y2_element_wise_sum));
    return true;
}

int main(int argc, char** argv) {
    std::cout << "Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n";

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <matrix_file>\n";
        return 1;
    }

    std::string matrix_file = argv[1];
    CSR csr_matrix = CSR::CreateFromMatrixMarketFile(matrix_file);

    if (!BenchmarkSpMV(csr_matrix)) {
        std::cerr << "SpMV benchmark failed validation.\n";
        return 1;
    } else {
        std::cout << "SpMV benchmark passed validation.\n";
    }

    return 0;
}
