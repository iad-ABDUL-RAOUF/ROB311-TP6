import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from time import time
import matplotlib.pyplot as plt


print('Importing training data')
directory = "/home/iad/Documents/ENSTA/3A/rob_311/TP_rob311/TP6/" # data directory on iad's computer 
# directory = "./" # data directory on madeleine's computer 
# directory = "yourpath/" # prof : put your data directory here
datafileTrain = "optdigits.tra"
dataFileTest = "optdigits.tes"

# choose if you want to have a nice plot or just to print the result in the terminal
nDimPCA = 30
plotPrettyConfMatrix = True


train_data = np.empty((3823,64),dtype=int)
train_class = np.empty(3823,dtype=int)

print('Importing training data')
file = open(directory+datafileTrain, "r")
csvFile = csv.reader(file)
i = -1
for row in csvFile:
    if i > -1:
        train_class[i] = np.array(row[-1])
        train_data[i][:] = np.array([row[:-1]])
    i += 1
print('train_class.shape : ', train_class.shape)

print('Importing testing data')
test_data = np.empty((1797, 64), dtype=int)
test_class = np.empty(1797, dtype=int)

file = open(directory+dataFileTest, "r")
csvFile = csv.reader(file)
i = -1
for row in csvFile:
    if i > -1:
        test_class[i] = np.array(row[-1])
        test_data[i][:] = np.array([row[:-1]])
    i += 1
print('test_class.shape : ', test_class.shape)

# Reducting the number of components using PCA

print("Reducing dimensions of data. The number of PCA component is set to", nDimPCA)
t0PCA = time()
pca = PCA(n_components=nDimPCA)
pca.fit(train_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)
elapsedPCA = time() - t0PCA
print("PCA dimension reduction time :",elapsedPCA, "s")
print("train data shape after PCA :",train_data.shape)



kmeans = KMeans(n_clusters = 10)
print('Training classifier')
t0TrainClassifier = time()
kmeans.fit(train_data)
elapsedTrainClassifier = time() - t0TrainClassifier
print("classifier training time ",elapsedTrainClassifier, "s")


print('Using trained classifier')
t0Prediction = time()
test_predict = kmeans.predict(test_data)
elapsedPrediction = time() - t0Prediction
print("prediction time :",elapsedPrediction, "s")

print("having ", nDimPCA, " component, total time for PCA, classifier training, and class prediction :", elapsedPrediction+elapsedTrainClassifier+elapsedPCA, "s")


# compute the confusion matrix and the accuracy
conf_mat = confusion_matrix(test_class, test_predict)
nClass = conf_mat.shape[0]
accuracy = 0
for ii in range(nClass):
    accuracy += conf_mat[ii][ii]
accuracy = accuracy/test_class.size


# display results
print("accuracy = ", accuracy)
 
if plotPrettyConfMatrix:
    print("here is the confusion matrix :")
    print(conf_mat)

print("done")