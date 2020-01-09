import pandas as pd
import numpy as np

from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from logger import AfterEpochLogger


def one_hot_encode(x: pd.Series) -> np.array:
    label = x.unique()
    label.sort()
    M = np.eye(x.nunique())
    dictionary = dict(zip(label, M))
    result = []
    for ex in x:
        result.append(dictionary[ex])
    return np.array(result)


def create_input_layer(feature: np.array) -> (Tensor, Tensor):
    input_layer = Input(shape=(len(feature[0]),))
    x = Dense(2, activation='sigmoid', use_bias=True)(input_layer)
    #model = Model(inputs=input_layer, outputs=x)
    return input_layer, x


df = pd.read_csv('poker-hand-testing.data', header=None, sep=',')

# TODO split
# Number of Instances: 25010 training, 1,000,000 testing

features = df.iloc[:, :-1].copy()
X = []
Y = one_hot_encode(df[len(df.columns) - 1])
for feature_column_number in features:
    X.append(one_hot_encode(features[feature_column_number]))

inputs = []
outputs = []

for i in range(len(X)):
    input, output = create_input_layer(X[i])
    inputs.append(input)
    outputs.append(output)

combined = concatenate(outputs)

h1 = Dense(30, activation='sigmoid', use_bias=True)(combined)
h2 = Dense(20, activation='sigmoid', use_bias=True)(h1)
h3 = Dense(10, activation='sigmoid', use_bias=True)(h2)
y = Dense(10, activation='softmax')(h2)

model = Model(inputs=inputs, outputs=y)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam())

# print(model.summary())

k_fold = KFold(n_splits=5)

loggers = []
scores = []

for train_index, test_index in k_fold.split(X[0]):
    X_train = []
    X_test = []
    for feature_index in range(len(X)):
        X_train.append(X[feature_index][train_index])
        X_test.append(X[feature_index][test_index])
    Y_train, Y_test = Y[train_index], Y[test_index]

    model = Model(inputs=inputs, outputs=y)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam())
    logger = AfterEpochLogger(1)
    # TODO nie walidowac na testowych
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1, batch_size=1024, callbacks=[logger])

    predicts_train = model.predict(X_train)
    score_train = roc_auc_score(Y_train, predicts_train)
    predicts = model.predict(X_test)
    score = roc_auc_score(Y_test, predicts) # TODO 3 fold does not work
    print("%s %.2f%%" % ('Train data AUC value for that fold: ', score_train * 100))
    print("%s %.2f%%" % ('Test data AUC value for that fold: ', score * 100))
    scores.append(score * 100)
    loggers.append(logger)

# RESULT

print("5 fold AUC (2 sigma/95%% confidence): %.2f%% (+/- %.2f%%)" % (np.mean(scores), 2*np.std(scores)))
# TODO clean code
plt.figure(figsize=(15, 20))
plt.subplot(421)
plt.ylim([0, 1])
plt.ylabel('Training loss')
plt.xlabel('Epoch')
plt.title('Training loss')
plt.plot(loggers[0].history_loss)
plt.plot(loggers[1].history_loss)
plt.plot(loggers[2].history_loss)
plt.plot(loggers[3].history_loss)
plt.plot(loggers[4].history_loss)
plt.legend(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
plt.subplot(422)
plt.title('Averaged training loss')
avg_loss = np.sum([loggers[i].history_loss for i in range(5)],axis=0)/5
plt.ylim([0, 1])
plt.plot(avg_loss)

plt.subplot(423)
plt.ylim([0, 1])
plt.ylabel('Validation loss')
plt.plot(loggers[0].val_history_loss)
plt.plot(loggers[1].val_history_loss)
plt.plot(loggers[2].val_history_loss)
plt.plot(loggers[3].val_history_loss)
plt.plot(loggers[4].val_history_loss)
plt.subplot(424)
avg_val_loss = np.sum([loggers[i].val_history_loss for i in range(5)],axis=0)/5
plt.ylim([0,1])
plt.plot(avg_val_loss)

plt.subplot(425)
plt.ylim([0, 1])
plt.ylabel('Training acc')
plt.plot(loggers[0].history_accuracy)
plt.plot(loggers[1].history_accuracy)
plt.plot(loggers[2].history_accuracy)
plt.plot(loggers[3].history_accuracy)
plt.plot(loggers[4].history_accuracy)
plt.subplot(426)
avg_acc = np.sum([loggers[i].history_accuracy for i in range(5)],axis=0)/5
plt.ylim([0, 1])
plt.plot(avg_acc)

plt.subplot(427)
plt.ylim([0, 1])
plt.ylabel('Validation acc')
plt.plot(loggers[0].val_history_accuracy)
plt.plot(loggers[1].val_history_accuracy)
plt.plot(loggers[2].val_history_accuracy)
plt.plot(loggers[3].val_history_accuracy)
plt.plot(loggers[4].val_history_accuracy)
plt.subplot(428)
avg_val_acc = np.sum([loggers[i].val_history_accuracy for i in range(5)],axis=0)/5
plt.ylim([0, 1])
plt.plot(avg_val_acc)

plt.show()
