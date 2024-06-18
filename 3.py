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

suma = cv2.addWeighted(imagen1, 0.5, imagen2, 0.5, 0)
print("Suma")
plt.imshow(suma)

print("Resta")
resta = cv2.subtract(imagen1, imagen2)
plt.imshow(resta)