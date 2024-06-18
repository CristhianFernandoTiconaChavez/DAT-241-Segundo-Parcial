#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct {
    int* row_ptr;
    int* col_ind;
    double* values;
    int rows;
    int cols;
    int nnz;
} csr_matrix;

csr_matrix* create_csr_matrix(int rows, int cols, int nnz) {
    csr_matrix* mat = (csr_matrix*)malloc(sizeof(csr_matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->nnz = nnz;
    mat->row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    mat->col_ind = (int*)malloc(nnz * sizeof(int));
    mat->values = (double*)malloc(nnz * sizeof(double));
    return mat;
}

void free_csr_matrix(csr_matrix* mat) {
    free(mat->row_ptr);
    free(mat->col_ind);
    free(mat->values);
    free(mat);
}

void multiply_row_by_matrix(int row, csr_matrix* A, csr_matrix* B, double* C_row) {
    for (int idx = A->row_ptr[row]; idx < A->row_ptr[row + 1]; idx++) {
        int colA = A->col_ind[idx];
        double valA = A->values[idx];
        for (int j = B->row_ptr[colA]; j < B->row_ptr[colA + 1]; j++) {
            int colB = B->col_ind[j];
            double valB = B->values[j];
            C_row[colB] += valA * valB;
        }
    }
}

void parallel_multiply(csr_matrix* A, csr_matrix* B, csr_matrix* C, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < A->rows; i++) {
        double* C_row = (double*)calloc(B->cols, sizeof(double));
        multiply_row_by_matrix(i, A, B, C_row);
        store_row_in_csr(C, i, C_row);
        free(C_row);
    }
}

int main() {
    int rows = 1000, cols = 1000, nnz = 10000;
    csr_matrix* A = create_csr_matrix(rows, cols, nnz);
    csr_matrix* B = create_csr_matrix(rows, cols, nnz);
    csr_matrix* C = create_csr_matrix(rows, cols, 0); // Inicialmente sin valores



    int num_threads = 4;
    parallel_multiply(A, B, C, num_threads);

    free_csr_matrix(A);
    free_csr_matrix(B);
    free_csr_matrix(C);

    return 0;
}
