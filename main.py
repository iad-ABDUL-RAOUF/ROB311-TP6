import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from time import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale


print('Importing training data')
directory = "/home/iad/Documents/ENSTA/3A/rob_311/TP_rob311/TP6/" # data directory on iad's computer 
# directory = "./" # data directory on madeleine's computer 
# directory = "yourpath/" # prof : put your data directory here
datafileTrain = "optdigits.tra"
dataFileTest = "optdigits.tes"

# choose if you want to have a nice plot or just to print the result in the terminal
doPCA = True
doRescale = False
nDimPCA = 10
KMean_method = 'k-means++' # 'k-means++', 'random', 'pca_based'
n_KMean_init = 200
tolerence = 10**(-5)
max_iteration = 800

print('Importing training data')
train_data = np.empty((3823,64),dtype=int)
train_class = np.empty(3823,dtype=int)

file = open(directory+datafileTrain, "r")
csvFile = csv.reader(file)
i = 0
for row in csvFile:
    train_class[i] = np.array(row[-1])
    train_data[i][:] = np.array([row[:-1]])
    i += 1
print('train_class.shape : ', train_class.shape)

print('Importing testing data')
test_data = np.empty((1797, 64), dtype=int)
test_class = np.empty(1797, dtype=int)

file = open(directory+dataFileTest, "r")
csvFile = csv.reader(file)
i = 0
for row in csvFile:
    test_class[i] = np.array(row[-1])
    test_data[i][:] = np.array([row[:-1]])
    i += 1
print('test_class.shape : ', test_class.shape)


# Reducting the number of components using PCA
print("Reducing dimensions of data. The number of PCA component is set to", nDimPCA)
t0PCA = time()
if doPCA:
    pca = PCA(n_components=nDimPCA)
    pca.fit(train_data)
    if not KMean_method == 'pca_based':
        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)
    # Rescale each feature
    if doRescale:
        train_data = scale(train_data, axis=0)
        test_data = scale(test_data, axis=0)
elapsedPCA = time() - t0PCA
print("PCA dimension reduction time :",elapsedPCA, "s")
print("train data shape after PCA :",train_data.shape)


if KMean_method == 'pca_based':
    KMean_method = pca.components_
    print('coucou')
    
kmeans = KMeans(n_clusters = 10,init = KMean_method, n_init = n_KMean_init, tol = tolerence, max_iter = max_iteration)
print('Training classifier')
t0TrainClassifier = time()
kmeans.fit(train_data)
elapsedTrainClassifier = time() - t0TrainClassifier
print("classifier training time ",elapsedTrainClassifier, "s")

print('associate label to cluster')
# associting digits to clusters. The most represented digit in a cluster defines
# the label
t0LabelAssiation = time()
train_predict = kmeans.labels_
brut_conf_mat_train = confusion_matrix(train_class, train_predict)
cluster2HumanLabel = np.argmax(brut_conf_mat_train,0)
train_predict_human = cluster2HumanLabel[train_predict]
elapsedLabelAssociation = time() - t0LabelAssiation
print("classifier training time ",elapsedLabelAssociation, "s")


print('Using trained classifier')
t0Prediction = time()
test_predict = kmeans.predict(test_data)
elapsedPrediction = time() - t0Prediction
print("prediction time :",elapsedPrediction, "s")
test_predict_human = cluster2HumanLabel[test_predict]

print("having ", nDimPCA, " component, total time for PCA, classifier training, and class prediction :", elapsedPrediction+elapsedTrainClassifier+elapsedPCA, "s")

##########################
# compute the confusion matrix and the accuracy
print("here is the brut train confusion matrix :")
print(brut_conf_mat_train)

# compute the confusion matrix and the accuracy
brut_conf_mat_test = confusion_matrix(test_class, test_predict)
print("here is the brut test confusion matrix :")
print(brut_conf_mat_test)

#########################
# compute the confusion matrix and the accuracy
human_conf_mat_train = confusion_matrix(train_class, train_predict_human)
nClass = human_conf_mat_train.shape[0]
accuracy_train = 0
for ii in range(nClass):
    accuracy_train += human_conf_mat_train[ii][ii]
accuracy_train = accuracy_train/train_class.size

# display results
print("accuracy_train = ", accuracy_train)
print("here is the human train confusion matrix :")
print(human_conf_mat_train)

# compute the confusion matrix and the accuracy
human_conf_mat_test = confusion_matrix(test_class, test_predict_human)
nClass = human_conf_mat_test.shape[0]
accuracy_test = 0
for ii in range(nClass):
    accuracy_test += human_conf_mat_test[ii][ii]
accuracy_test = accuracy_test/test_class.size


# display results
print("accuracy_test = ", accuracy_test)
print("here is the human test confusion matrix :")
print(human_conf_mat_test)

print("done")
