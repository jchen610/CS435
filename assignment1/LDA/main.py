import csv
import sys
import lda as mod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


data = np.loadtxt(sys.argv[1], delimiter=",", dtype=np.float32)
precision = len(str(data[0][0]))
# split into X and y
n_samples, n_features = data.shape
n_features -= 1

X = data[:, 0:n_features]
y = data[:, n_features]



# Project the data onto the 2 primary linear discriminants
lda = mod.LDA(20)
lda.fit(X, y)
X_projected = lda.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


result = []

for row in X_projected:
    realNumbers = []
    for index in row:
        realNumbers.append(round(index.real, precision))
    result.append(realNumbers)




with open("lca_result.txt", 'w') as file:
    for row in result:
        cleanText = str(row).replace("[", "").replace("]", "").replace(",", "")
        file.write(cleanText + "\n")