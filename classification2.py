import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

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

X_train = df_train.iloc[:, 0:10].as_matrix()
Y_train = one_hot_encode(df_train[len(df_train.columns) - 1])


df_test = pd.read_csv('poker-hand-testing.data', header=None, sep=',')
X_test = df_test.iloc[:,0:10].as_matrix()
y_test_class = df_test[len(df_test.columns) - 1]
Y_test = one_hot_encode(y_test_class)

model = Sequential()
model.add(Dense(100, input_shape=(10,), activation='relu', use_bias=True))
model.add(Dense(200, activation='relu', use_bias=True))
model.add(Dense(100, activation='relu', use_bias=True))
model.add(Dense(10, activation='softmax'))

loggers = []
scores = []

k_fold = KFold(n_splits=5)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
cp_callback = ModelCheckpoint(filepath='poker-model.h5', save_weights_only=False, save_best_only=True, verbose=1)

for train_index, val_index in k_fold.split(X_train):
    result = next(k_fold.split(X_train), None)

    x_train = X_train[result[0]]
    x_val = X_train[result[1]]

    y_train = Y_train[result[0]]
    y_val = Y_train[result[1]]

    logger = AfterEpochLogger(5)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=32, callbacks=[logger, cp_callback], shuffle=True)

    predicts_train = model.predict(X_train)
    score_train = roc_auc_score(Y_train, predicts_train, average='micro')
    predicts = model.predict(X_test)
    score = roc_auc_score(Y_test, predicts, average='micro')
    print("%s %.2f%%" % ('Train data AUC value for that fold: ', score_train * 100))
    print("%s %.2f%%" % ('Test data AUC value for that fold: ', score * 100))
    scores.append(score * 100)
    loggers.append(logger)

# RESULT

y_pred = model.predict_classes(X_test, verbose=1)
false_preds = [(x, y, p) for (x, y, p) in zip(X_test, y_test_class, y_pred) if y != p]

print(f'False predicts: {len(false_preds)}')

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
