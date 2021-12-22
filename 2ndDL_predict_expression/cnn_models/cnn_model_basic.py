from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Flatten, Dropout

### The model structure (especially kernel sizes) should be changed depending on the data shape (or length of the sequences or bins).
### The current version has been adjusted to 20-bins with 50 TF channels

def build_cnn(data_length = 20, n_channel = 50, last_dense = 2):
    model = models.Sequential()
    model.add(layers.Conv1D(64, 15, activation='relu', input_shape=(data_length, n_channel), padding='same'))
    model.add(layers.Conv1D(64, 15, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv1D(32, 10, activation='relu', padding='same'))
    model.add(layers.Conv1D(32, 10, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
    model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
    model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.5))
    
    model.add(Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(last_dense, activation='softmax'))
    return model

