import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool

def multiply_row_by_matrix(row_idx, A_data, A_row, A_col, B_data, B_row, B_col, shape):
    A_coo = coo_matrix((A_data, (A_row, A_col)), shape=shape)
    B_coo = coo_matrix((B_data, (B_row, B_col)), shape=shape)
    result_row = A_coo.getrow(row_idx).dot(B_coo).toarray()
    return row_idx, result_row

def parallel_multiply(A, B, num_processes):
    A_coo = A.tocoo()
    B_coo = B.tocoo()
    shape = A.shape
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(multiply_row_by_matrix, [
            (i, A_coo.data, A_coo.row, A_coo.col, B_coo.data, B_coo.row, B_coo.col, shape) 
            for i in range(A.shape[0])
        ])
    
    # Crear la matriz de resultados
    sorted_results = sorted(results)
    rows = [result[1] for result in sorted_results]
    C_data = np.vstack(rows)
    return csr_matrix(C_data)

if __name__ == "__main__":
    # Crear matrices dispersas de ejemplo
    A = csr_matrix(np.random.random((1000, 1000)))
    B = csr_matrix(np.random.random((1000, 1000)))
    
    # Realizar la multiplicación en paralelo
    num_processes = 4
    C = parallel_multiply(A, B, num_processes)
    print(C)