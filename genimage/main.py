import functools
import os
import logging

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def print_setup():
    logger = logging.getLogger(__name__)

    logger.info("TF version: ", tf.__version__)
    logger.info("TF Hub version: ", hub.__version__)
    logger.info("Eager mode enabled: ", tf.executing_eagerly())
    logger.info("GPU available: ", tf.config.list_physical_devices("GPU"))


def crop_center(image: tf.Tensor) -> tf.Tensor:
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape
    )

    return image


@functools.lru_cache(maxsize=None)
def load_image(
    image_url, image_size=(256, 256), preserve_aspect_ratio=True
) -> tf.Tensor:
    """Loads and preprocesses images."""
    # Cache image file locally.
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    image_data = tf.io.read_file(image_path)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(image_data, channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)

    return img


def show_n(images, titles=("",)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320

    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)

    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect="equal")
        plt.axis("off")
        plt.title(titles[i] if len(titles) > i else "")

    plt.show()


def load_style_image(url: str, size: tuple[int, int]):
    img = load_image(url, size)

    return tf.nn.avg_pool(img, ksize=[3, 3], strides=[1, 1], padding="SAME")


STYLIZE_HUB_HANDLE = (
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
)


def stylize(content_image: tf.Tensor, style_image: tf.Tensor) -> tf.Tensor:
    """Stylize a content image by applying styles from another image"""
    hub_module = hub.load(STYLIZE_HUB_HANDLE)

    (stylized_image,) = hub_module(tf.constant(content_image), tf.constant(style_image))

    return stylized_image


def transpose_style(
    content_image_url: str,
    content_image_size: int,
    style_image_url: str,
    style_image_size: int,
):
    """Load and transpose styles from one image to another"""
    content_image = load_image(
        content_image_url,
        (content_image_size, content_image_size),
    )

    style_image = load_style_image(
        style_image_url,
        (style_image_size, style_image_size),
    )

    stylized_image = stylize(content_image, style_image)

    return (content_image, style_image, stylized_image)


def main():
    content_image_size = 384
    style_image_size = 256

    stylized_image = transpose_style(
        "https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg",
        content_image_size,
        "https://newevolutiondesigns.com/images/freebies/4k-galaxy-wallpaper-9.jpg",
        style_image_size,
    )

    show_n(
        [content_image, style_image, stylized_image],
        ["Content image", "Style image", "Stylized Image"],
    )


if __name__ == "__main__":
    main()
