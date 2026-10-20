import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

model = MobileNetV2(weights="imagenet")

LAST_CONV_LAYER = "Conv_1"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())

    return heatmap.numpy()


def save_gradcam(image_path, heatmap):
    original = cv2.imread(image_path)

    if original is None:
        raise FileNotFoundError(
            f"Could not open image: {image_path}"
        )

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    output_path = (
        image_path.rsplit(".", 1)[0]
        + "_gradcam.jpg"
    )

    cv2.imwrite(
        output_path,
        overlay
    )

    return output_path


def classify_image(image_path):
    try:
        img = image.load_img(
            image_path,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(img)

        img_array = preprocess_input(
            img_array
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        predictions = model.predict(
            img_array
        )

        decoded_predictions = decode_predictions(
            predictions,
            top=3
        )[0]

        print(
            "\nTop-3 Predictions for",
            image_path
        )

        for i, (_, label, score) in enumerate(
            decoded_predictions
        ):
            print(
                f"  {i + 1}: "
                f"{label} "
                f"({score:.2f})"
            )

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            LAST_CONV_LAYER
        )

        output_path = save_gradcam(
            image_path,
            heatmap
        )

        print(
            f"\nGrad-CAM image saved as: "
            f"{output_path}"
        )

    except Exception as e:
        print(
            f"Error processing "
            f"'{image_path}': {e}"
        )


if __name__ == "__main__":
    print(
        "Image Classifier with Grad-CAM "
        "(type 'exit' to quit)\n"
    )

    while True:
        image_path = input(
            "Enter image filename: "
        ).strip()

        if image_path.lower() == "exit":
            print("Goodbye!")
            break

        classify_image(image_path)