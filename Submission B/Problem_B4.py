# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"

    training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(
        bbc['text'], bbc['category'], train_size=training_portion, shuffle=False)

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    # Tokenizing and padding sequences
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # You can also use Tokenizer to encode your label.
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(bbc['category'])

    training_labels = label_tokenizer.texts_to_sequences(training_labels)
    validation_labels = label_tokenizer.texts_to_sequences(validation_labels)

    training_labels = tf.convert_to_tensor(training_labels)
    validation_labels = tf.convert_to_tensor(validation_labels)

    # Defining a custom callback to stop training
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_accuracy') > 0.93:
                print("\nTraining stopped as desired accuracy reached!")
                self.model.stop_training = True

    callbacks = myCallback()

    # Building the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Compiling the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Fitting the model with training data
    model.fit(training_padded, training_labels, epochs=1000,
              validation_data=(validation_padded, validation_labels), callbacks=[callbacks])
    
    return model

if __name__ == '__main__':
    # Running the solution
    model = solution_B4()
    model.save("model_B4.h5")
