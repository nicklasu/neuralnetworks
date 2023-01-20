"""matplotlib.pyplotでグラフを描いています。"""
import io

import matplotlib.pyplot as plt
import tensorflow as tf

def plot_figure(label, training_history, validation_history, epochs, title):
    """パラメーターはグラフのラベル「例えばaccuracy」とヒストリーからゲットした訓練と
        バリデーションヒストリーとエポックの数とグラフのタイトルです。"""
    epochs = range(1, epochs + 1)
    plt.figure(figsize=(13, 10))
    plt.plot(epochs, training_history, 'ro', markersize=3, label="training " + label)
    plt.plot(epochs, validation_history, 'b', label="validation " + label)
    plt.title(title + " highest " + label + ": " + str(round(max(validation_history), 4)) +
        " epoch " + str(validation_history.index(max(validation_history))))
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close()
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
