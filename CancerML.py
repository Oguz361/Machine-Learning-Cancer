import tensorflow as tf
import pandas as pd
import numpy as np

# Setzt Präzision für numpy-Ausgaben
np.set_printoptions(precision=3, suppress=True)
from tensorflow.keras import layers

# CSV-Datensatz laden und vorbereiten
dataset = pd.read_csv("C:/Users/oguzk/Desktop/Datensaetze/breast_cancer1.csv")
dataset = dataset.drop('id', axis=1)

# Datensatz in Trainings- und Testdatensätze aufteilen
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Feature- und Label-Trennung
cancer_features = train_dataset.copy()
test_features = test_dataset.copy()

# Wandelt die Diagnose-Labels in numerische Werte um (M = malign = 1, B = benign = 0)
cancer_labels = (cancer_features.pop('diagnosis') == 'M').astype(int)
test_labels = (test_features.pop('diagnosis') == 'M').astype(int)

# Tensorflow-Inputs vorbereiten
inputs = {}
for name, column in cancer_features.items():
    dtype = tf.string if column.dtype == object else tf.float32
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

# Numerische Features vorbereiten und normalisieren
numeric_inputs = {name: input for name, input in inputs.items() if input.dtype == tf.float32}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(dataset[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
preprocessed_inputs = [all_numeric_inputs]

# Kategorische Features vorbereiten (falls vorhanden)
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue
    lookup = layers.StringLookup(vocabulary=np.unique(cancer_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

# Preprocessing-Modell erstellen
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
cancer_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

# Datensätze in TensorFlow Datasets konvertieren
cancer_features_dict = {name: np.array(value) for name, value in cancer_features.items()}
train_dataset = tf.data.Dataset.from_tensor_slices((cancer_features_dict, cancer_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(cancer_labels)).batch(32)

test_features_dict = {name: np.array(value) for name, value in test_features.items()}
val_dataset = tf.data.Dataset.from_tensor_slices((test_features_dict, test_labels)).batch(32)

# Modell erstellen
def cancer_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),  
        metrics=['accuracy']
    )
    return model

cancer_model = cancer_model(cancer_preprocessing, inputs)

# Early Stopping Callback hinzufügen
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Modell trainieren
cancer_model.fit(
    train_dataset, 
    epochs=50,  # Erlaubt mehr Epochen
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

# Modell evaluieren
test_loss, test_accuracy = cancer_model.evaluate(val_dataset)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
