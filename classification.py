import pandas as pd
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import utils

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


df_train = pd.read_csv('poker-hand-training-true.data', header=None, sep=',')

features = df_train.iloc[:, :-1].copy()
X_train = []
Y_train = one_hot_encode(df_train[len(df_train.columns) - 1])
for feature_column_number in features:
    X_train.append(one_hot_encode(features[feature_column_number]))


df_test = pd.read_csv('poker-hand-testing.data', header=None, sep=',')
features = df_test.iloc[:, :-1].copy()
X_test = []
Y_test = one_hot_encode(df_test[len(df_test.columns) - 1])
for feature_column_number in features:
    X_test.append(one_hot_encode(features[feature_column_number]))

inputs = []
outputs = []

for i in range(0, len(X_train), 2):
    input_figure = Input(shape=(len(X_train[i][0]),))
    input_color = Input(shape=(len(X_train[i+1][0]),))

    figure_layer = Dense(5, activation='relu', use_bias=True)(input_figure)
    color_layer = Dense(2, activation='relu', use_bias=True)(input_color)
    # color

    card_layer = Dense(5, activation='relu', use_bias=True)(concatenate([figure_layer, color_layer]))
    inputs.append(input_figure)
    inputs.append(input_color)
    outputs.append(card_layer)

combined = concatenate(outputs)

h1 = Dense(50, activation='relu', use_bias=True)(combined)
h2 = Dense(30, activation='relu', use_bias=True)(h1)
y = Dense(10, activation='softmax')(h2)

loggers = []
scores = []

k_fold = KFold(n_splits=5)

model = Model(inputs=inputs, outputs=y)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

for train_index, val_index in k_fold.split(X_train[0]):
    X_train_split = []
    X_val = []
    for feature_index in range(len(X_train)):
        X_train_split.append(X_train[feature_index][train_index])
        X_val.append(X_train[feature_index][val_index])
    Y_train_split, Y_val = Y_train[train_index], Y_train[val_index]

    logger = AfterEpochLogger(5)
    model.fit(X_train_split, Y_train_split, validation_data=(X_val, Y_val), epochs=10, batch_size=32, callbacks=[logger], shuffle=True)

    predicts_train = model.predict(X_train)
    score_train = roc_auc_score(Y_train, predicts_train, average='micro')
    predicts = model.predict(X_test)
    score = roc_auc_score(Y_test, predicts, average='micro')
    print("%s %.2f%%" % ('Train data AUC value for that fold: ', score_train * 100))
    print("%s %.2f%%" % ('Test data AUC value for that fold: ', score * 100))
    scores.append(score * 100)
    loggers.append(logger)

    pred = utils.probas_to_classes(predicts)

    false_preds = [(x, y, p) for (x, y, p) in zip(X_test, Y_test, predicts) if y.argmax(axis=-1) != p.argmax(axis=-1)]
    print(len(false_preds))

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
plt.legend(['Fold 1'])
plt.subplot(422)
plt.title('Averaged training loss')
avg_loss = loggers[0].history_loss
plt.ylim([0, 1])
plt.plot(avg_loss)

plt.subplot(423)
plt.ylim([0, 1])
plt.ylabel('Validation loss')
plt.plot(loggers[0].val_history_loss)
plt.subplot(424)
avg_val_loss = loggers[0].val_history_loss
plt.ylim([0, 1])
plt.plot(avg_val_loss)

plt.subplot(425)
plt.ylim([0, 1])
plt.ylabel('Training acc')
plt.plot(loggers[0].history_accuracy)
plt.subplot(426)
avg_acc = loggers[0].history_accuracy
plt.ylim([0, 1])
plt.plot(avg_acc)

plt.subplot(427)
plt.ylim([0, 1])
plt.ylabel('Validation acc')
plt.plot(loggers[0].val_history_accuracy)
plt.subplot(428)
avg_val_acc = loggers[0].val_history_accuracy
plt.ylim([0, 1])
plt.plot(avg_val_acc)

plt.savefig('plot.png')
