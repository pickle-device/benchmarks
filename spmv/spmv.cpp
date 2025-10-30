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
#include "pickle_device_manager.h"
#include "pickle_job.h"
#endif // ENABLE_PICKLEDEVICE

#if ENABLE_GEM5==1
#pragma message("Compiling with gem5 instructions")
#include <gem5/m5ops.h>
#include "m5_mmap.h"
#endif // ENABLE_GEM5

#if ENABLE_PICKLEDEVICE==1
std::unique_ptr<PickleDeviceManager> pdev(new PickleDeviceManager());
uint64_t* UCPage = NULL;
uint64_t* PerfPage = NULL;
#endif // ENABLE_PICKLEDEVICE

const uint64_t PERF_THREAD_START = 0;
const uint64_t PERF_THREAD_COMPLETE = 1;

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

    static CSR CreateFromCSRFile(const std::string& filename) {
        CSR csr;
        std::cout << "Creating CSR from file: " << filename << "\n";
        // Read the file and populate csr.row_ptr, csr.col_ind, csr.values
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            std::cerr << "Error opening file: " << filename << "\n";
            assert(false);
            return csr;
        }

        // Read the matrix info
        size_t num_rows = 0;
        size_t num_cols = 0;
        size_t num_nonzeros = 0;
        infile >> num_rows >> num_cols >> num_nonzeros;
        csr.num_rows = num_rows;
        csr.num_cols = num_cols;
        csr.num_nonzeros = num_nonzeros;
        std::cout << "Matrix dimensions: " << num_rows << " x " << num_cols
                  << " with " << num_nonzeros << " non-zeros.\n";
        // Allocating memory
        csr.row_ptr.reserve(num_rows+1);
        csr.col_ind.reserve(num_nonzeros);
        csr.values.reserve(num_nonzeros);
        // Read the row_ptr array
        {
            int v = 0;
            for (size_t i = 0; i < num_rows + 1; i++) {
                infile >> v;
                csr.row_ptr.push_back(v);
            }
        }
        // Read the col_ind array
        {
            int v = 0;
            for (size_t i = 0; i < num_nonzeros; i++) {
                infile >> v;
                csr.col_ind.push_back(v);
            }
        }
        // Read the values array
        {
            double v = 0;
            for (size_t i = 0; i < num_nonzeros; i++) {
                infile >> v;
                csr.values.push_back(v);
            }
        }
        infile.close();
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

#if ENABLE_PICKLEDEVICE==1
    std::shared_ptr<PickleArrayDescriptor> GetRowPtrArrayDescriptor() const {
        auto row_ptr_array_descriptor = std::make_shared<PickleArrayDescriptor>();
        const auto* begin_ptr = row_ptr.data();
        const auto* end_ptr = row_ptr.data() + row_ptr.size();
        row_ptr_array_descriptor->vaddr_start = (uint64_t)(begin_ptr);
        row_ptr_array_descriptor->vaddr_end = (uint64_t)(end_ptr);
        row_ptr_array_descriptor->element_size = sizeof(row_ptr[0]);
        return row_ptr_array_descriptor;
    }

    std::shared_ptr<PickleArrayDescriptor> GetColIndArrayDescriptor() const {
        auto col_ind_array_descriptor = std::make_shared<PickleArrayDescriptor>();
        const auto* begin_ptr = col_ind.data();
        const auto* end_ptr = col_ind.data() + col_ind.size();
        col_ind_array_descriptor->vaddr_start = (uint64_t)(begin_ptr);
        col_ind_array_descriptor->vaddr_end = (uint64_t)(end_ptr);
        col_ind_array_descriptor->element_size = sizeof(col_ind[0]);
        return col_ind_array_descriptor;
    }

    std::shared_ptr<PickleArrayDescriptor> GetValuesArrayDescriptor() const {
        auto values_array_descriptor = std::make_shared<PickleArrayDescriptor>();
        const auto* begin_ptr = values.data();
        const auto* end_ptr = values.data() + values.size();
        values_array_descriptor->vaddr_start = (uint64_t)(begin_ptr);
        values_array_descriptor->vaddr_end = (uint64_t)(end_ptr);
        values_array_descriptor->element_size = sizeof(values[0]);
        return values_array_descriptor;
    }

    template <typename T>
    std::shared_ptr<PickleArrayDescriptor> GetArrayDescriptor(const std::vector<T>& array) const {
        auto array_descriptor = std::make_shared<PickleArrayDescriptor>();
        const auto* begin_ptr = array.data();
        const auto* end_ptr = array.data() + array.size();
        array_descriptor->vaddr_start = (uint64_t)(begin_ptr);
        array_descriptor->vaddr_end = (uint64_t)(end_ptr);
        array_descriptor->element_size = sizeof(T);
        return array_descriptor;
    }
#endif // ENABLE_PICKLEDEVICE

    // Perform Sparse Matrix-Vector Multiplication (SpMV)
    // Params:
    //   x: the x vector in y=Ax
    //   y: the output vector y in y=Ax
    void SpMVInto(const std::vector<double>& x, std::vector<double>& y) const {
        assert(x.size() == GetNumCols());
        std::cout << "ROI Start" << std::endl;
#if ENABLE_GEM5==1
        m5_exit_addr(0); // exit 1
#endif // ENABLE_GEM5
#if ENABLE_PICKLEDEVICE==1
        PerfPage = (uint64_t*) pdev->getPerfPagePtr();
        std::cout << "PerfPage: 0x" << std::hex << (uint64_t)PerfPage << std::dec << std::endl;
        assert(PerfPage != nullptr);
#endif
        std::cout << "Starting SpMV computation using " << omp_get_max_threads() << " threads.\n";
        #pragma omp parallel  // Enable OpenMP parallelization
        {
            const uint64_t thread_id = (uint64_t)omp_get_thread_num();
            *PerfPage = (thread_id << 1) | PERF_THREAD_START;
            #pragma omp for nowait
            for (size_t i = 0; i < GetNumRows(); ++i) {
                int row_start = row_ptr[i];
                int row_end = row_ptr[i + 1];
                for (int j = row_start; j < row_end; ++j) {
                    y[i] += values[j] * x[col_ind[j]];
                }
            }
            *PerfPage = (thread_id << 1) | PERF_THREAD_COMPLETE;
        }
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 2
#endif // ENABLE_GEM5
        std::cout << "ROI end" << std::endl;
    }

    void SpMVWithPrefetcherInto(const std::vector<double>& x, std::vector<double>& y) const {
        assert(x.size() == GetNumCols());
#if ENABLE_PICKLEDEVICE==1
        // If we use pickle prefetcher, we need to do the following steps,
        // Step 1. Gather the prefetcher specs
        uint64_t use_pdev = 0;
        uint64_t prefetch_distance = 0;
        PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
        use_pdev = specs.availability;
        prefetch_distance = specs.prefetch_distance;
        std::cout << "Use pdev: " << use_pdev << "; Prefetch distance: " << prefetch_distance << std::endl;
        // Step 2. Construct the dependency graph
        // row_ptr -> col_ind -> values and x
        PickleJob job(/*kernel_name*/"spmv");
        // Construct the row_ptr array descriptor
        std::shared_ptr<PickleArrayDescriptor> row_ptr_array_descriptor = GetArrayDescriptor<int>(row_ptr);
        row_ptr_array_descriptor->access_type = AccessType::Ranged;
        row_ptr_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(row_ptr_array_descriptor);
        // Construct the col_ind array descriptor
        std::shared_ptr<PickleArrayDescriptor> col_ind_array_descriptor = GetArrayDescriptor<int>(col_ind);
        col_ind_array_descriptor->access_type = AccessType::SingleElement;
        col_ind_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(col_ind_array_descriptor);
        // Construct the values array descriptor
        std::shared_ptr<PickleArrayDescriptor> values_array_descriptor = GetArrayDescriptor<double>(values);
        values_array_descriptor->access_type = AccessType::SingleElement;
        values_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(values_array_descriptor);
        // Construct the y array descriptor
        std::shared_ptr<PickleArrayDescriptor> x_array_descriptor = GetArrayDescriptor<double>(x);
        x_array_descriptor->access_type = AccessType::SingleElement;
        x_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(x_array_descriptor);
        // Construct the dependency graph
        row_ptr_array_descriptor->dst_indexing_array_id = col_ind_array_descriptor->getArrayId();
        col_ind_array_descriptor->dst_indexing_array_id = x_array_descriptor->getArrayId();
        job.print();
        // Step 3. Send the graph to the prefetcher
        pdev->sendJob(job);
        std::cout << "Sent job" << std::endl;
        // Step 4. Setup the communication uncacheable page
        UCPage = (uint64_t*) pdev->getUCPagePtr(0);
        std::cout << "UCPage: 0x" << std::hex << (uint64_t)UCPage << std::dec << std::endl;
        assert(UCPage != nullptr);
#endif
        std::cout << "ROI Start" << std::endl;
#if ENABLE_GEM5==1
        m5_exit_addr(0); // exit 3
#endif // ENABLE_GEM5
        std::cout << "Starting SpMV computation using " << omp_get_max_threads() << " threads.\n";
        #pragma omp parallel  // Enable OpenMP parallelization
        {
            const uint64_t thread_id = (uint64_t)omp_get_thread_num();
            *PerfPage = (thread_id << 1) | PERF_THREAD_START;
            #pragma omp for nowait
            for (size_t i = 0; i < GetNumRows(); ++i) {
#if ENABLE_PICKLEDEVICE==1
                *UCPage = static_cast<uint64_t>(i);
#endif
                int row_start = row_ptr[i];
                int row_end = row_ptr[i + 1];
                for (int j = row_start; j < row_end; ++j) {
                    y[i] += values[j] * x[col_ind[j]];
                }
            }
            *PerfPage = (thread_id << 1) | PERF_THREAD_COMPLETE;
            //std::cout << "Thread " << omp_get_thread_num() << " completed SpMV computation.\n";
        }
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 4
#endif // ENABLE_GEM5
        std::cout << "ROI end" << std::endl;
    }

};  // class CSR

// Function to compare two double values with a tolerance rate
// Returns true if they are equal within the tolerance, false otherwise
// Tolerance is defined as tol_rate * expected_value
bool ExpectEqual(double expected_value, double actual_value, double tol_rate = 1e-6) {
    const double max_diff_allowed = std::abs(expected_value * tol_rate);
    if (std::abs(expected_value - actual_value) > max_diff_allowed) {
        printf(
            "tol_rate = %.8f, max_diff_allowed = %8.f, Expected %.8f but got %.8f\n",
            tol_rate, max_diff_allowed, actual_value, expected_value
        );
        return false;
    }
    return true;
}

// Benchmarking function for SpMV
// Runs SpMV twice:
// - iter 1 is a warm-up run performing A * x_3 where x_3 is a vector of all threes
// - iter 2 is the actual benchmark performing A * x_5 where x_5 is a vector of all fives
// Params:
// - A: The CSR matrix
// Returns:
// - true if the benchmark passes validation, false otherwise
bool BenchmarkSpMV(const CSR& A) {
    const double A_element_wise_sum = A.GetSumOfValues();
    std::vector<double> x3(A.GetNumCols(), 3.0); // Input vector of all threes
    std::vector<double> x5(A.GetNumCols(), 5.0); // Input vector of all fives

    // First iteration of SpMV
    const double y1_time_start = omp_get_wtime();
    std::vector<double> y(A.GetNumRows(), 0.0);
    A.SpMVInto(x3, y); // Warm-up run
    const double y1_time_end = omp_get_wtime();
    std::cout << "Warm-up SpMV time: " << (y1_time_end - y1_time_start) << " seconds.\n";

    // Second iteration of SpMV
    const double y2_time_start = omp_get_wtime();
    std::fill(y.begin(), y.end(), 0.0);
    A.SpMVWithPrefetcherInto(x5, y); // Run twice for benchmarking
    const double y2_time_end = omp_get_wtime();
    std::cout << "Benchmark SpMV time: " << (y2_time_end - y2_time_start) << " seconds.\n";

    // Validate results
    double y2_element_wise_sum = 0.0;
    for (const auto& val : y) {
        y2_element_wise_sum += val;
    }
    assert(ExpectEqual(A_element_wise_sum * 5.0, y2_element_wise_sum));
    return true;
}

int main(int argc, char** argv) {
    std::cout << "Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n";

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <csr_file>\n";
        std::cout << "    <csr_file>: path to a csr file containing matrix\n";
        return 1;
    }

    std::string matrix_file = argv[1];
    CSR csr_matrix = CSR::CreateFromCSRFile(matrix_file);

#if ENABLE_GEM5==1
    map_m5_mem();
#endif // ENABLE_GEM5

    if (!BenchmarkSpMV(csr_matrix)) {
        std::cerr << "SpMV benchmark failed validation.\n";
        return 1;
    } else {
        std::cout << "SpMV benchmark passed validation.\n";
    }

#if ENABLE_GEM5==1
  //unmap_m5_mem();
#endif // ENABLE_GEM5

    return 0;
}
