import numpy as np
import keras
from sklearn.model_selection import train_test_split
from build_model import build_model,build_model_4l
import time
import os
import json

def create_spectrogram_model_with_dataset(validation_size=0.15, test_size=0.15):

    np_data_file = "images.npy"
    np_labels_file = "labels.npy"

    # get train, validation, test splits
    print("Start loading data...")

    X = np.load(np_data_file)
    y = np.load(np_labels_file)
    y = y.astype(np.int64).reshape(-1, 1)

    print(len(X))
    print(len(y))

    #print(y)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    #  #add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(X_train.shape)
    model = build_model_4l((X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    return model, X_train, X_validation, X_test, y_train, y_validation, y_test

model, X_train, X_validation, X_test, y_train, y_validation, y_test = create_spectrogram_model_with_dataset()


# compile model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

model.summary()

# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Start measuring time
start_time = time.time()

# train model
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100)

# Stop measuring time
end_time = time.time()

# Calculate the training time in seconds
training_time_seconds = end_time - start_time

# Convert training time to hours, minutes, and seconds
training_hours, remainder = divmod(training_time_seconds, 3600)
training_minutes, training_seconds = divmod(remainder, 60)

# Print the training time in a human-readable format
print(f"Training time: {int(training_hours):02d} hours, {int(training_minutes):02d} minutes, {int(training_seconds):02d} seconds")

# Define the relative path to the models folder
models_folder = os.path.join(script_dir, '..', 'models')

# # plot accuracy/error for training and validation
# plot_history(history, os.path.join(models_folder, 'train_history_model_3l_20e'))

# evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

# Save the model to the models folder
model.save(os.path.join(models_folder, 'mfcc_model_4l_60e_04d_test.h5'))

with open('training_history_mfcc_model_4l_60e_04d_test.json', 'w') as json_file:
    json.dump(history.history, json_file)



