'''
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
'''

import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from time import time
import matplotlib.pyplot as plt


print('Importing training data')
directory = "/home/iad/Documents/ENSTA/3A/rob_311/TP_rob311/TP4/" # data directory on iad's computer 
# directory = "./" # data directory on madeleine's computer 
# directory = "yourpath/" # prof : put your data directory here
datafileTrain = "mnist_train.csv"
dataFileTest = "mnist_test.csv"

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


'''
# Creating and training SVM Classifier
print('Training classifier')
t0TrainClassifier = time()
clf = make_pipeline(StandardScaler(), SVC(gamma = 'auto'))
clf.fit(train_data, train_class)
elapsedTrainClassifier = time() - t0TrainClassifier
print("classifier training time ",elapsedTrainClassifier, "s")

# Use the trained classifier to classify the test dataset


print('Using trained classifier')
t0Prediction = time()
test_predict = clf.predict(test_data)
elapsedPrediction = time() - t0Prediction
print("PCA dimension reduction time :",elapsedPrediction, "s")

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
 
if not plotPrettyConfMatrix:
    print("here is the confusion matrix :")
    print(conf_mat)
else:
    np.set_printoptions(precision=2)
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, test_data, test_class,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    
    plt.show()

print("done")