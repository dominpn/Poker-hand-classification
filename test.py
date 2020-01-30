import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Our deep learning library is Keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils.np_utils import to_categorical
import numpy as np
# fixed random seed for reproducibility
np.random.seed(0)

features = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
data = pd.read_csv('poker-hand-training-true.data', names=features)

nb_classes = 10  # we have 10 classes of poker hands
cls = {}
for i in range(nb_classes):
    cls[i] = len(data[data.CLASS==i])
print(cls)

X_train = data.iloc[:,0:10].as_matrix()
y_train = data.iloc[:,10].as_matrix()

Y_train = to_categorical(y_train)

features = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
tdata = pd.read_csv('poker-hand-testing.data', names=features)

X_test = tdata.iloc[:,0:10].as_matrix()
y_test = tdata.iloc[:,10].as_matrix()
Y_test = to_categorical(y_test)

model2 = Sequential()
model2.add(Dense(50, input_shape=(10,), kernel_initializer='uniform', activation='relu'))
model2.add(Dense(50, kernel_initializer='uniform', activation='relu'))
model2.add(Dense(nb_classes, kernel_initializer='uniform', activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

h = model2.fit(
    X_train,
    Y_train,
    batch_size=32,
    epochs=200,
    shuffle=True,
    verbose=0
)

loss, accuracy = model2.evaluate(X_test, Y_test, verbose=0)
print("Test: accuracy=%f loss=%f" % (accuracy, loss))

y_pred = model2.predict_classes(X_test, verbose=1)
false_preds = [(x,y,p) for (x,y,p) in zip(X_test, y_test, y_pred) if y != p]

print(len(false_preds))