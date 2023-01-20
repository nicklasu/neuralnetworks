"""matplotlib.pyplotエラーが出している方は以下のコードを試してみて下さい。
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    以下はtensorboardのコマンドです。
    tensorboard --logdir=logs --host localhost
    tensorboard --logdir=logs --host 0.0.0.0"""
import numpy as np
import tensorflow as tf
import tensorflow_addons.optimizers
from keras import models
from keras import layers
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
import keras.applications
import keras.optimizers
import keras.regularizers
from dataset_loader import kuzushiji49_loader
from plot_figure import plot_figure
from image_writer import image_writer
from balanced_accuracy import balanced_accuracy
from preview import preview

CONVNET = "keras.applications.ResNet152V2"
EPOCHS = 1
BATCH_SIZE = 1024
OPTIMIZER = "Adam"
MIX = False
(train_ds, val_ds, test_ds), class_amount = kuzushiji49_loader(BATCH_SIZE, MIX)
# preview(train_ds)
conv_base = eval(CONVNET)(
  weights=None,
  include_top=False,
  input_shape=(32, 32, 1))

model = models.Sequential()
model.add(tf.keras.layers.Resizing(
    32,
    32,
    interpolation='bilinear',
    crop_to_aspect_ratio=False))
#model.add(layers.Dropout(0.2))
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(class_amount * 2, activation='relu'))
    #activity_regularizer=keras.regularizers.l2(l2=0.00001)))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(class_amount, activation='softmax'))

csv_logger = CSVLogger("run/log.csv", separator=';', append=True)
tensorboard_callback = TensorBoard(
    log_dir="logs",
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None)

model.compile(optimizer=OPTIMIZER,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_ds,
    epochs=EPOCHS, callbacks=[csv_logger], batch_size=BATCH_SIZE,
    validation_data=val_ds)
"""
predict = model.predict(test_ds)
y_test = np.load('kmnist-test-labels.npz')['arr_0']
y_test = tf.one_hot(y_test, class_amount)
p_test = predict # Model predictions of class index

accs = []

for data in range(10000):
    for cls in range(class_amount):
        if y_test[data][cls] == 1:
            accs.append(p_test[data][cls])

accs = np.mean(accs) # Final balanced accuracy
print(accs)
"""
# test_loss, test_acc = model.evaluate(test_ds)

TITLE = CONVNET + " kkanji no Mixup " # + str(lr)

image_writer("logs/train_data/accuracy", plot_figure("accuracy",
    history.history["accuracy"], history.history['val_accuracy'], EPOCHS, TITLE))
image_writer("logs/train_data/loss", plot_figure("loss",
    history.history["loss"], history.history['val_loss'], EPOCHS, TITLE))
    