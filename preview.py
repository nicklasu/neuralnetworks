"""画像を見せる関数です"""
import matplotlib.pyplot as plt

def preview(images):
    sample_images, sample_labels = next(iter(images))
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().squeeze())
        print(label.numpy().tolist())
        plt.axis("off")
        plt.show()