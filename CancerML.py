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

cancer_features = train_dataset.copy()
test_features = test_dataset.copy()

cancer_labels = cancer_features.pop('diagnosis')
test_labels = test_features.pop('diagnosis')

inputs = {}

for name, column in cancer_features.items():
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

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue
    lookup = layers.StringLookup(vocabulary=np.unique(cancer_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
cancer_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
#tf.keras.utils.plot_model(model = cancer_preprocessing, rankdir="LR", dpi=72, show_shapes=True)

cancer_features_dict = {name: np.array(value) for name, value in cancer_features.items()}
features_dict = {name:values[:1] for name, values in cancer_features_dict.items()}
print(cancer_preprocessing(features_dict))

def cancer_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer=tf.keras.optimzers.Adam())
    return model

cancer_model.fit(x=cancer_features_dict, y=cancer_labels, epochs=10)    