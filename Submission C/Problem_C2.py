# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def solution_C2():
    #mirip soal b2
    mnist = tf.keras.datasets.mnist

    # NORMALIZE YOUR IMAGE HERE
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_images = train_images / float(255)
    test_images = test_images / float(255)
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_accuracy') > 0.93):
                print("\nStopping..")
                self.model.stop_training = True

    callbacks = myCallback()

    # DEFINE YOUR MODEL HERE
    model = tf.keras.models.Sequential([
        # End with 10 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # COMPILE MODEL HERE
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # TRAIN YOUR MODEL HERE
    model.fit(train_images, train_labels, epochs=1000, 
              validation_data=(test_images, test_labels),verbose=2, callbacks=[callbacks])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")