# Knn-w-GridSearch

In this repository, we have implemented KNN for satellites dataset using K-nearest Neighbour Classifier.

The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that is used to solve both classification and regression problems. 
It assumes that similar things exist in close proximity. In other words, similar things are near to each other.

KNN works by finding the distances between a given data point, a query and all the remaining examples in the sample, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).

## Choosing the right value for K

To select the K that’s right for our data, we run the KNN algorithm several times with different values of K and choose the K that reduces the errors we encounter while maintaining the algorithm’s ability to accurately make predictions for new data.This can be easily done with the help of GridSearch.

##  Grid Search for Hyperparameter Tuning

Grid Search is the process of performing hyperparameter tuning in order to determine the optimal values for a given model.We know that the performance of a model significantly depends on the value of hyperparameters.Since there is no way to know in advance the best values for hyperparameters so ideally, we need to try all possible values to know the optimal values. Doing this manually could take a considerable amount of time and resources and thus we use GridSearchCV to automate the tuning of hyperparameters.

We have implemented the classifier from scratch and used Euclidean distance as a measure to decide k nearest neighbours.Next, we have compared the classification accuracy of our model with sklearn's KNearestNeighbour library to evaluate the model and dataset.

## To Run

To run the model locally, fist clone the repository using following command in your terminal.

git clone https://github.com/yay14/Knn-w-GridSearch.git

Next download the dataset from https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)

After downloading the train and test files, upload them in the same folder as the python notebook.

Then open the Knn & GridSearch.ipynb file and run in Jupyter notebook or google colab.

## Dataset

 Statlog (Landsat Satellite) Data Set is a collection of multi-spectral values of pixels in 3x3 neighbourhoods in a satellite image, and the classification associated with the central pixel in each neighbourhood. The aim is to predict this classification, given the multi-spectral values. In the sample database, the class of a pixel is coded as a number.

## TSNE

t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.

To better understand the data samples in test and training set,we have used the tsne plots to see the distribution of individual classes.

```python
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

#initialize TSNE from sklearn
tsne = TSNE()

#Plot tsne for training data
X_embedded = tsne.fit_transform(x_train)

palette = sns.color_palette("bright", 6)

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_train, legend='full', palette=palette)

```

![train_tsne](/images/train_tsne.png)

```python
#Plot tsne for test data
X_embedded = tsne.fit_transform(x_test)

palette = sns.color_palette("bright", 6)

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_test, legend='full', palette=palette)

```

![test_tsne](/images/test_tsne.png)

## Grid Search to find optimal K


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
In [21]:
#create new a knn model
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 20)}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

#fit model to data
knn_gscv.fit(x_train, y_train)

#check top performing n_neighbors value
print(knn_gscv.best_params_)
print(knn_gscv.best_score_)
plt.plot(knn_gscv.cv_results_['param_n_neighbors'].data, knn_gscv.cv_results_['mean_test_score'])

```

![grid_search](/images/grid_search.png)

## K-nn from scratch


```python
from collections import Counter
from sklearn.metrics import accuracy_score

#In the case of KNN, which is a lazy algorithm, the training block reduces 
#to just memorizing the training data. 
def train(x_train, y_train):
    # do nothing 
    return

def predict(x_train, y_train, x_test, k):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(x_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - x_train[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
    distances = sorted(distances)
    

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return most common target
    return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(x_train, y_train, x_test, k):
    # train on the input data
    train(x_train, y_train)
    
    predictions = []

    # loop over all observations
    for i in range(len(x_test)):
        predictions.append(predict(x_train, y_train, x_test[i, :], k))
        
    return np.asarray(predictions)

predictions = kNearestNeighbor(x_train, y_train, x_test, 12)

accuracy = accuracy_score(y_test, predictions)
print('\nThe accuracy of our classifier is {}%'.format(accuracy*100))

```

## K-nn using Sklearn

```python

knn = KNeighborsClassifier(n_neighbors=12)

# fitting the model
knn.fit(x_train, y_train)

# predict the response
pred = knn.predict(x_test)

# evaluate accuracy
print('\nThe accuracy of the classifier is {}%'.format(accuracy_score(y_test, pred)*100))

```

## Results

Using both classifiers , we got the same accuracy of 89.65% for the satellite dataset and optimal k was found to be 12 using GridSearchCV.
