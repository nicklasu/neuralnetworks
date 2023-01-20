"""https://github.com/keras-team/keras-io/blob/master/examples/vision/mixup.py"""
import tensorflow as tf

def mix_up(train_ds, val_ds, test_ds):
    def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def mix(ds_one, ds_two, alpha):
        # Unpack two datasets
        images_one, labels_one = ds_one
        images_two, labels_two = ds_two
        batch_size = tf.shape(images_one)[0]

        # Sample lambda and reshape it to do the mixup
        l = sample_beta_distribution(batch_size, alpha, alpha)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)
        return (images, labels)
    train_ds_mu = train_ds.map(
        lambda ds_one, ds_two: mix(ds_one, ds_two, alpha=0.5), num_parallel_calls=tf.data.AUTOTUNE
    )
    return train_ds_mu, val_ds, test_ds
    