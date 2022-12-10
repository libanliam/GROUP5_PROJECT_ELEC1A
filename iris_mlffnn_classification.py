import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd 
import numpy as np
data = pd.read_csv('iris.data')
# print(data.head())

#prints num of rows and cols
print(data.shape)
#prints data types
print(data.dtypes)
#looks into the ave, sd, quantiles, and other summary stats
print(data.describe())

x = data.iloc[:,:4].values

y = data.iloc[:,4:5].values

print('x values')
print(x)
print('y value')
print(y)

#preprocess data

#normalize data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)
print("normalized data \n", x)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print('ohe\n', y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import set_random_seed
set_random_seed(2022);

model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax')) # softmax function outputs probabilities

#last step is to compile entire thing
model.compile(loss='categorical_crossentropy', 
              optimizer = keras.optimizers.Adam(learning_rate=0.001), 
              metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_data = (X_test, y_test), epochs=50, batch_size=16)

#evlauating the neural net
y_pred = model.predict(X_test)
#converting prediction to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

#converting ohe test label to label

test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))


# Confusion Matrix
from sklearn import metrics
cm = metrics.confusion_matrix(test, pred)
print(cm)

# Accuracy, Precision, and F-1 Score
from sklearn.metrics import accuracy_score, precision_score, f1_score
acc = accuracy_score(test, pred)*100
print('Accuracy: ', str(round(acc, 2)))

prec = precision_score(pred, test, average='macro')*100
print('Precision: ', str(round(prec, 2)))

f1 = f1_score(pred, test, average='macro')*100
print('F1-score: ', str(round(f1, 2)))

# Plot Predictions
import matplotlib.pyplot as plt
def plot_predictions(predicted, test):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
    plt.show()
    
plot_predictions(pred, test)