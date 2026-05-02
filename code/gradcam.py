import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMG_SIZE
from model_manager import preprocess_for_model


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name

    return None


def get_layer_output_model(model, layer_name):
    try:
        return tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    except Exception:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    return tf.keras.models.Model(
                        inputs=model.inputs,
                        outputs=[layer.get_layer(layer_name).output, model.output]
                    )
                except Exception:
                    pass

    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = get_layer_output_model(model, last_conv_layer_name)

    if grad_model is None:
        return None

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return None

    heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_gradcam(original_image_path, heatmap, alpha=0.45):
    img = Image.open(original_image_path).convert("RGB").resize(IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    heatmap_img = Image.fromarray(heatmap_color)
    return Image.blend(img, heatmap_img, alpha)


def create_text_image(text, size=(450, 450)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = (35, 35, 35)

    cv2.putText(
        img,
        text,
        (45, size[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (80, 255, 80),
        2,
        cv2.LINE_AA
    )

    return Image.fromarray(img)


def generate_gradcam(file_path, model_item):
    if model_item is None:
        return None

    model = model_item["model"]
    model_name = model_item["name"]

    img = tf.keras.utils.load_img(
        file_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    img_raw = tf.keras.utils.img_to_array(img)
    img_raw = np.expand_dims(img_raw, axis=0)

    x = preprocess_for_model(img_raw, model_name)

    last_conv = find_last_conv_layer(model)

    if last_conv is None:
        return None

    heatmap = make_gradcam_heatmap(x, model, last_conv)

    if heatmap is None:
        return None

    return overlay_gradcam(file_path, heatmap)