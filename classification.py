import pandas as pd
import numpy as np

from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

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
    x = Dense(2, activation='relu', use_bias=True)(input_layer)
    return input_layer, x


df = pd.read_csv('poker-hand-testing.data', header=None, sep=',')

features = df.iloc[:, :-1].copy()
X = []
Y = one_hot_encode(df[len(df.columns) - 1])
for feature_column_number in features:
    X.append(one_hot_encode(features[feature_column_number]))

inputs = []
outputs = []

for i in range(0, len(X), 2):
    # input, output = create_input_layer(X[i])
    input_figure = Input(shape=(len(X[i][0]),))
    input_color = Input(shape=(len(X[i+1][0]),))

    figure_layer = Dense(5, activation='relu', use_bias=True)(input_figure)
    color_layer = Dense(2, activation='relu', use_bias=True)(input_color)
    # color

    card_layer = Dense(5, activation='relu', use_bias=True)(concatenate([figure_layer, color_layer]))
    inputs.append(input_figure)
    inputs.append(input_color)
    outputs.append(card_layer)

combined = concatenate(outputs)

h1 = Dense(30, activation='relu', use_bias=True)(combined)
h2 = Dense(20, activation='relu', use_bias=True)(h1)
y = Dense(10, activation='softmax')(h2)

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
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
    plot_model(model, to_file='model.png')
    logger = AfterEpochLogger(5)
    model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=256, callbacks=[logger])

    predicts_train = model.predict(X_train)
    score_train = roc_auc_score(Y_train, predicts_train, average='micro')
    predicts = model.predict(X_test)
    score = roc_auc_score(Y_test, predicts, average='micro')
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
avg_loss = np.sum([loggers[i].history_loss for i in range(5)], axis=0)/5
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
avg_val_loss = np.sum([loggers[i].val_history_loss for i in range(5)], axis=0)/5
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
avg_acc = np.sum([loggers[i].history_accuracy for i in range(5)], axis=0)/5
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
avg_val_acc = np.sum([loggers[i].val_history_accuracy for i in range(5)], axis=0)/5
plt.ylim([0, 1])
plt.plot(avg_val_acc)

plt.savefig('plot.png')
