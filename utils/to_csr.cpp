#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
    static CSR CreateFromMatrixMarketFile(const std::string& filename, const bool is_symmetric) {
        CSR csr;
        std::cout << "Creating CSR from .mtx file: " << filename << "\n";
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
            if (value == 0) {
                continue;
            }
            // Store the entry in the map (adjusting for 1-based indexing)
            if (entries.find({row - 1, col - 1}) != entries.end()) {
                std::cout << "Repeating entry: row " << row << ", col " << col <<std::endl;
            }
            assert(entries.find({row - 1, col - 1}) == entries.end());
            entries[{row - 1, col - 1}] = value;
            if (is_symmetric && (row != col)) {
                entries[{col - 1, row - 1}] = value;
            }
        }
        num_nonzeros = entries.size();
        csr.num_nonzeros = num_nonzeros;

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

    static CSR CreateFromGRFile(const std::string& filename) {
        CSR csr;
        std::cout << "Creating CSR from .gr file: " << filename << "\n";
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
        size_t num_nonzeros = 0;
        // Skip comment lines and find the first non-comment line
        while (infile.peek() == 'c') {
            infile.ignore(2048, '\n');
        }
        // Read matrix dimensions and number of non-zeros
        std::string temp;
        std::string temp2;
        infile >> temp >> temp2 >> num_rows >> num_nonzeros;
        csr.num_rows = num_rows;
        csr.num_cols = num_rows;
        csr.num_nonzeros = num_nonzeros;
        std::cout << "Matrix dimensions: " << csr.num_rows << " x " << csr.num_cols
                  << " with " << csr.num_nonzeros << " non-zeros.\n";
        size_t num_nonzeros_read = 0;
        // Now we read the non-zero entries
        std::map<std::pair<int, int>, double, std::less<std::pair<int, int>>> entries;
        std::string line;
        while (std::getline(infile, line)) {
            if (line.size() > 0 && line[0] == 'c') {
                continue;
            }
            std::istringstream stream(line);
            int row, col;
            double value;
            if (!(stream >> temp >> row >> col >> value)) {
                // malformed data line
                std::cout << "Malformed data line: " << line;
                continue;
            }
            if (value == 0) {
                continue;
            }
            // Store the entry in the map (adjusting for 1-based indexing)
            //if (entries.find({row - 1, col - 1}) != entries.end()) {
            //    std::cout << "Repeating entry: row " << row << ", col " << col <<std::endl;
            //}
            //assert(entries.find({row - 1, col - 1}) == entries.end());
            entries[{row - 1, col - 1}] = value;
        }
        num_nonzeros = entries.size();
        csr.num_nonzeros = num_nonzeros;

        std::cout << "num_nonzeros: " << num_nonzeros << std::endl;

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

    void WriteToCSRFile(std::string filename) {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file: " << filename << "\n";
            assert(false);
        }
        // The first line contains the information about the matrix
        outfile << num_rows << " " << num_cols << " " << num_nonzeros;
        // The next (num_rows+1) lines are the content of the row_ptr array
        for (const auto v: row_ptr) {
            outfile << v << std::endl;
        }
        // The next (num_nonzeros) lines are the content of the col_ind array
        for (const auto v: col_ind) {
            outfile << v << std::endl;
        }
        // The next (num_nonzeros) lines are the content of the values array
        outfile << std::setprecision(std::numeric_limits<double>::max_digits10)
                << std::defaultfloat;
        for (const auto v: values) {
            outfile << v << std::endl;
        }
        outfile.close();
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
};  // class CSR

int main(int argc, char** argv) {
    std::cout << "Convert a file to a CSR file\n";

    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <is_symmetric> <input_file> <output_csr_file>\n";
        std::cout << "    <is_symmetric>: must be either True of False\n";
        std::cout << "    <input_file>: path to a matrix market (.mtx) or a graph (.gr) file\n";
        std::cout << "    <output_csr_file>: path to the output csr file\n";
        return 1;
    }

    const bool is_symmetric = std::string(argv[1]) == "True";
    std::string matrix_file = argv[2];
    std::filesystem::path matrix_file_path = matrix_file;
    std::string csr_file = argv[3];

    CSR csr_matrix;
    std::string file_extension = matrix_file_path.extension();
    std::cout << "File ext: " << file_extension << std::endl;
    if (file_extension == ".mtx") {
        csr_matrix = CSR::CreateFromMatrixMarketFile(matrix_file, is_symmetric);
    } else if (file_extension == ".gr") {
        csr_matrix = CSR::CreateFromGRFile(matrix_file);
    } else {
        std::cerr << "Unknown file extension " << file_extension << std::endl;
        assert(false);
    }
    csr_matrix.WriteToCSRFile(csr_file);
    std::cout << "Wrote the matrix to " << csr_file << std::endl;
    return 0;
}
