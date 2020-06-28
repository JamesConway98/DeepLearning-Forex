from tensorflow.python.client import device_lib
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import dataCleaner as dC

print(device_lib.list_local_devices())


EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 128  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{dC.RATIO_TO_PREDICT}-TIK-{dC.SEQ_LEN}-SEQ-{dC.FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

train_x, train_y, validation_x, validation_y = dC.sort_data()

model = keras.models.Sequential(
        [
            keras.layers.LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(0.1),
            keras.layers.BatchNormalization(),
            keras.layers.LSTM(128),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2, activation='softmax')
        ]
    )

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=opt, #"sgd",
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y)
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))