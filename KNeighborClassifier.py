from sklearn.neighbors import KNeighborsClassifier
# df is the database to be loaded
# Import KNeighborsClassifier from sklearn.neighbors
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create arrays for the features and the response variable
df = pd.read_csv('gapminder.csv')
y = df['party'].values
X = df.drop('party',axis=1).values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X_train,y_train)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
print(knn.score(X_test, y_test))