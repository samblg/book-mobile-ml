from keras.models import Sequential
from keras:layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

nb_train_samples = 2000
nb_validation_samples = 1000
nb_epoch = 50

model = Sequential([
    Convolution2D(
        32, 3, 3, input_shape=input_shape,
        activation=' relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Convolution2D(64, 3，3, activation='relu'),
    MaxPooling2D(pool size=(2, 21)),
    Flatten(),
    Dense(64，activation='relu'),
    Dropout(0.51),
    Dense(1, activation='sigmoid'),
])

model.compile(Loss='binary_crossentropy', optimizer='rmsprop', metrics=[' accuracy'])

model.fit_generator(
    train_flow,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_flow,
    nb_val_samples=nb_validation_samples)

ensure_dir(target + 'weights')
model.save_weights(target + 'weights/' + '1.h5')
