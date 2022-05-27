from skimage.transform import resize
import pandas as pd
import numpy as np
from sklearn import svm
from skimage import io
from pathlib import Path
from sklearn.model_selection import train_test_split
from skimage.feature import hog

##trainnig
X_train = []
folder = Path(r"\images")
p = folder.glob('**/*')

##extracting features
for file in p:
    img = io.imread(Path.joinpath(folder, file))
    if img is not None:
        img = resize(img, (128, 64))
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
        X_train.append(fd)

X_train = pd.DataFrame(X_train)
Y_train = []

#labeling
for i in range(2000):
    if i >= 0 and i < 1000:
        Y_train.append(-1)
    else:
        Y_train.append(1)

#shuffling data
trainset = pd.DataFrame(X_train)
trainset['label'] = Y_train

trainset = trainset.sample(frac=1).reset_index(drop=True)

X_train = pd.DataFrame(trainset)
X_train.drop('label', axis=1, inplace=True)

Y_train = trainset['label']

##fitting data
poly_svm = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, Y_train)

print("Training Accuracy:")

accuracy = poly_svm.score(X_train, Y_train)
print(accuracy)

####testing
X_test = []
folder = Path(r"\test")
p = folder.glob('**/*')

##extracting features
for file in p:
    img = io.imread(Path.joinpath(folder, file))
    if img is not None:
        img = resize(img, (128, 64))
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
        X_test.append(fd)

X_test = pd.DataFrame(X_test)
Y_test = []

#labeling
for i in range(200):
    if i >= 0 and i < 100:
        Y_test.append(-1)
    else:
        Y_test.append(1)

#shuffling data
testset = pd.DataFrame(X_test)
testset['label'] = Y_test

testset = testset.sample(frac=1).reset_index(drop=True)

X_test = pd.DataFrame(testset)
X_test.drop('label', axis=1, inplace=True)

Y_test = testset['label']

##Checking accuracy
prediction = poly_svm.predict(X_test)

print("Testing Accuracy:")

accuracy = poly_svm.score(X_test, Y_test)
print(accuracy)
