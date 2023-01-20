"""画像をセーブです。"""
from datetime import datetime
import tensorflow as tf

def image_writer(logdir, image):
    """パラメーターはセーブ場所ディレクトリーとセーブしたい画像です。"""
    logdir = logdir + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        tf.summary.image("Training data", image, step=0)
