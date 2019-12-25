import pandas as pd
import numpy as np

from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


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
    model = Model(inputs=input_layer, outputs=x)
    return model.input, model.output


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
y = Dense(10, activation='softmax')(h3)

model = Model(inputs=inputs, outputs=y)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam())

# print(model.summary())

k_fold = KFold(n_splits=5)

for train_index, test_index in k_fold.split(X[0]):
    X_train = []
    X_test = []
    for feature_index in range(len(X)):
        X_train.append(X[feature_index][train_index])
        X_test.append(X[feature_index][test_index])
    Y_train, Y_test = Y[train_index], Y[test_index]

    model = Model(inputs=inputs, outputs=y)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam())
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=8) # callbacks

    predicts_train = model.predict(X_train)
    score_train = roc_auc_score(Y_train, predicts_train)
    predicts = model.predict(X_test)
    score = roc_auc_score(Y_test, predicts)
    print("%s %.2f%%" % ('Train data AUC value for that fold: ', score_train * 100))
    print("%s %.2f%%" % ('Test data AUC value for that fold: ', score * 100))


