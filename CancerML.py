import tensorflow as tf
import pandas as pd
import numpy as np
#make numpy values easier to read
np.set_printoptions(precision=3,suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

dataset = pd.read_csv("C:/Users/oguzk/Desktop/Datensaetze/breast_cancer1.csv")
dataset = dataset.drop('id', axis=1)
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#train_dataset.describe().transpose()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('diagnosis')
test_labels = test_features.pop('diagnosis')

inputs = {}

for name, column in train_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)        

#print(inputs)    

numeric_inputs = {name:input for name,input in inputs.items() if input.dtype == tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(dataset[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

print(all_numeric_inputs)