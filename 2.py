from google.colab import drive
import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import cv2
import scipy.sparse as sp

drive.mount('/content/drive')
os.chdir("/content/drive/MyDrive/data")
imagen1 = cv2.imread("nave.png")
imagen2 = cv2.imread("espacio.png")

m_sparce1 = sp.coo_matrix(imagen1[:,:,1])
print("-------------------Imagen 1 matriz sparce:-------------------\n")
print(m_sparce1, "\n")
m_sparce2 = sp.coo_matrix(imagen2[:,:,1])
print("-------------------Imagen 2 matriz sparce:-------------------\n")
print(m_sparce2, "\n")