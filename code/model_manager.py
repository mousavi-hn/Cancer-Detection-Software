import numpy as np
import tensorflow as tf

from config import IMG_SIZE, MODELS_ROOT_DIR, CANCER_TYPES, INVERT_OUTPUT_SCORE
from quantum_layer import QuantumLayer


model_vault = {ctype: [] for ctype in CANCER_TYPES}


def preprocess_for_model(img_raw, model_name):
    name = model_name.lower()

    if "resnet" in name:
        return tf.keras.applications.resnet_v2.preprocess_input(img_raw.copy())
    if "vgg16" in name or "vgg19" in name:
        return tf.keras.applications.vgg16.preprocess_input(img_raw.copy())
    if "densenet" in name:
        return tf.keras.applications.densenet.preprocess_input(img_raw.copy())
    if "efficientnet" in name:
        return tf.keras.applications.efficientnet.preprocess_input(img_raw.copy())
    if "mobilenet" in name:
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_raw.copy())
    if "xception" in name:
        return tf.keras.applications.xception.preprocess_input(img_raw.copy())
    if "inception" in name:
        return tf.keras.applications.inception_v3.preprocess_input(img_raw.copy())

    return img_raw.copy() / 255.0


def extract_score(pred):
    pred = np.array(pred)

    if pred.ndim == 2 and pred.shape[1] == 1:
        score = float(pred[0][0])
    elif pred.ndim == 2 and pred.shape[1] == 2:
        score = float(pred[0][1])
    else:
        score = float(pred.reshape(-1)[0])

    if INVERT_OUTPUT_SCORE:
        score = 1.0 - score

    return score


def load_all_models():
    print("Initializing Model Vault...")

    for ctype in CANCER_TYPES:
        target_dir = MODELS_ROOT_DIR / ctype.lower()

        if not target_dir.exists():
            print(f"Directory not found: {target_dir}. Skipping...")
            continue

        model_files = sorted(list(target_dir.rglob("*.keras")))

        for m_path in model_files:
            try:
                print(f"Loading {ctype}: {m_path.name}")

                model = tf.keras.models.load_model(
                    str(m_path),
                    custom_objects={"QuantumLayer": QuantumLayer},
                    compile=False
                )

                is_hybrid = "hybrid" in m_path.name.lower()

                if is_hybrid:
                    model.run_eagerly = True

                model_vault[ctype].append({
                    "name": m_path.name.lower(),
                    "path": str(m_path),
                    "model": model,
                    "is_hybrid": is_hybrid,
                })

            except Exception as e:
                print(f"Error loading {m_path.name}: {e}")

        print(f"Loaded {len(model_vault[ctype])} models for {ctype}")


def predict_models(file_path, cancer_type):
    current_models = model_vault[cancer_type]

    if not current_models:
        return None, [], None

    img = tf.keras.utils.load_img(
        file_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    img_raw = tf.keras.utils.img_to_array(img)
    img_raw = np.expand_dims(img_raw, axis=0)

    predictions = []

    for item in current_models:
        model = item["model"]
        model_name = item["name"]

        try:
            x = preprocess_for_model(img_raw, model_name)

            if item["is_hybrid"]:
                pred = model(x, training=False).numpy()
            else:
                pred = model.predict(x, verbose=0)

            score = extract_score(pred)

            predictions.append({
                "model": model_name,
                "score": score,
                "is_hybrid": item["is_hybrid"],
            })

            print(f"{model_name} => {score:.4f}")

        except Exception as e:
            print(f"Prediction failed for {model_name}: {e}")

    if not predictions:
        return None, [], None

    final_score = max(p["score"] for p in predictions)

    best_classical = None
    for p in sorted(predictions, key=lambda x: x["score"], reverse=True):
        if not p["is_hybrid"]:
            best_classical = p
            break

    best_classical_item = None

    if best_classical is not None:
        best_classical_item = next(
            item for item in current_models
            if item["name"] == best_classical["model"]
        )

    return final_score, predictions, best_classical_item