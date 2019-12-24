import pandas as pd
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import StratifiedKFold


def one_hot_encode(x: pd.Series) -> np.array:
    label = x.unique()
    label.sort()
    M = np.eye(x.nunique())
    dictionary = dict(zip(label, M))
    result = []
    for ex in x:
        result.append(dictionary[ex])
    return np.array(result)


def create_input_layer(feature):
    input_layer = Input(shape=(len(feature[0]),))
    x = Dense(2, activation='sigmoid', use_bias=True)(input_layer)
    model = Model(inputs=input_layer, outputs=x)
    return model.input, model.output


df = pd.read_csv('poker-hand-testing.data', header=None, sep=',')

features = df.iloc[:, :-1].copy()
X = []
Y = df[len(df.columns) - 1]
for feature_column_number in features:
    X.append(one_hot_encode(features[feature_column_number]))
