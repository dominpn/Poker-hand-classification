import pandas as pd
import numpy as np

from tensorflow import Tensor
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from sklearn.metrics import roc_auc_score

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


def group(df):
    total_list = []

    for row in df.iterrows():

        grouped_list = []

        for x in range(1, len(row[1]), 2):
            grouped_list.append(int(str(row[1][x - 1]) + str(row[1][x])))

        total_list.append(grouped_list)

    return total_list


df_train = pd.read_csv('poker-hand-training-true.data', header=None, sep=',')

X_train = group(df_train.iloc[:, :-1].copy())
Y_train = one_hot_encode(df_train[len(df_train.columns) - 1])


df_test = pd.read_csv('poker-hand-testing.data', header=None, sep=',')
X_test = group(df_test.iloc[:, :-1].copy())
Y_test = one_hot_encode(df_test[len(df_test.columns) - 1])

loggers = []
scores = []

logger = AfterEpochLogger(5)

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(5,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

history = model.fit(np.array(X_train), np.array(Y_train),
                    batch_size=32,
                    epochs=200,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=[logger])

score = model.evaluate(np.array(X_test), np.array(Y_test), verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

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
