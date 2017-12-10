import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

readIris = genfromtxt('Iris.csv',delimiter=',')
print(readIris)
    

        
def ecdf(data):
    # compute ECDF for a one-dimensional array
    # ECDF stands for Empirical Cumulative Distribution Function


    # Number of data points: n
    n = len(data)

    # x-data for ECDF : x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y

x_vers, y_vers = ecdf(readIris)

# Generate plot
plt.plot(x_vers,y_vers,marker='.',linestyle='none')

# Make the margins nice
plt.margins(0.02)

# Label the axes
plt.xlabel('petal length')
plt.ylabel('ECDF')

# Display the plot
plt.show()

