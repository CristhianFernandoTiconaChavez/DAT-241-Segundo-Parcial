import cv2
import scipy.sparse as sp
from google.colab import drive
import os

drive.mount('/content/drive')
os.chdir("/content/drive/MyDrive/data")

imagen1 = cv2.imread("nave.png")
imagen2 = cv2.imread("espacio.png")

m_sparse1 = sp.coo_matrix(imagen1[:, :, 1])
m_sparse2 = sp.coo_matrix(imagen2[:, :, 1])

m_sparse1_csr = m_sparse1.tocsr()
m_sparse2_csr = m_sparse2.tocsr()

resultado = m_sparse1_csr.dot(m_sparse2_csr)

print(f"Matriz 1 - Shape: {m_sparse1_csr.shape}, Número de Elementos No Nulos: {m_sparse1_csr.nnz}")
print(f"Matriz 2 - Shape: {m_sparse2_csr.shape}, Número de Elementos No Nulos: {m_sparse2_csr.nnz}")
print(f"Resultado - Shape: {resultado.shape}, Número de Elementos No Nulos: {resultado.nnz}")