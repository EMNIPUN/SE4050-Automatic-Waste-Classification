import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Target classes
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)

# Base directories
APP_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
MODELS_DIR = APP_DIR / 'models'
SRC_MODELS_DIR = APP_DIR.parent / 'src' / 'models'
RESULTS_MODELS_DIR = APP_DIR.parent / 'results' / 'models'

def find_model_path(candidate_names):
    """Search for model file across common project directories."""
    search_dirs = [MODELS_DIR, SRC_MODELS_DIR, RESULTS_MODELS_DIR, APP_DIR]
    for d in search_dirs:
        for name in candidate_names:
            p = d / name
            if p.exists():
                return p
    return None

# Load available models
loaded_models = {}

print("--- Initializing Deep Learning Models ---")

# 1. Custom CNN
cnn_path = find_model_path(['custom_cnn.keras', 'custom_cnn.h5'])
if cnn_path:
    try:
        loaded_models['cnn'] = tf.keras.models.load_model(cnn_path)
        print(f"[LOADED] Custom CNN from: {cnn_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load Custom CNN: {e}")
else:
    print("[PENDING] Custom CNN weights not found. Using placeholder.")

# 2. ResNet50
resnet_path = find_model_path(['resnet50.keras', 'resnet50.h5', 'resnet50_waste_model.keras', 'resnet50_best.keras'])
if resnet_path:
    try:
        loaded_models['resnet'] = tf.keras.models.load_model(resnet_path)
        print(f"[LOADED] ResNet50 from: {resnet_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load ResNet50: {e}")
else:
    print("[PENDING] ResNet50 weights not found. Using placeholder.")

# 3. MobileNetV2 (Our Model)
mobilenet_path = find_model_path(['mobilenetv2.keras', 'mobilenetv2_finetuned_best.keras', 'mobilenetv2.h5'])
if mobilenet_path:
    try:
        loaded_models['mobilenet'] = tf.keras.models.load_model(mobilenet_path)
        print(f"[LOADED] MobileNetV2 from: {mobilenet_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load MobileNetV2: {e}")
else:
    print("[WARNING] MobileNetV2 weights not found!")

# 4. EfficientNetB0
effnet_path = find_model_path(['efficientnetb0.keras', 'efficientnetb0.h5', 'efficientnet_best.keras'])
if effnet_path:
    try:
        loaded_models['effnet'] = tf.keras.models.load_model(effnet_path)
        print(f"[LOADED] EfficientNetB0 from: {effnet_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load EfficientNetB0: {e}")
else:
    print("[PENDING] EfficientNetB0 weights not found. Using placeholder.")

# Warmup pass to eliminate initial graph compilation latency
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
for m in loaded_models.values():
    try:
        _ = m.predict(dummy, verbose=0)
    except Exception:
        pass

print("Models initialized and warmed up.")
print("-------------------------------------------\n")


def preprocess_for_model(img, model_type):
    """Preprocess a PIL image for a specific model architecture."""
    img = img.resize(IMG_SIZE)
    img_arr = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_arr, axis=0)
    
    if model_type == 'cnn':
        return img_batch / 255.0
    elif model_type == 'resnet':
        return tf.keras.applications.resnet50.preprocess_input(img_batch.copy())
    elif model_type == 'mobilenet':
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_batch.copy())
    elif model_type == 'effnet':
        return tf.keras.applications.efficientnet.preprocess_input(img_batch.copy())
    return img_batch


def predict_single_model(model_key, model_name, model_type, img, specs):
    """Runs inference on a single model or returns a placeholder."""
    if model_key in loaded_models:
        model = loaded_models[model_key]
        x = preprocess_for_model(img, model_type)
        
        t0 = time.perf_counter()
        probs = model.predict(x, verbose=0)[0]
        latency = (time.perf_counter() - t0) * 1000
        
        top_idx = int(np.argmax(probs))
        top_cls = CLASSES[top_idx]
        top_conf = probs[top_idx] * 100
        
        label_dict = {cls_name: float(p) for cls_name, p in zip(CLASSES, probs)}
        
        info_md = f"""
| Attribute | Detail |
| :--- | :--- |
| **Prediction** | **{top_cls.title()}** ({top_conf:.1f}%) |
| **Latency** | {latency:.1f} ms |
| **Accuracy** | {specs['test_acc']} |
| **Parameters** | {specs['params']} |
| **Status** | Active |
"""
        return label_dict, info_md
    else:
        label_dict = {"Awaiting Model File": 1.0}
        info_md = f"""
| Attribute | Detail |
| :--- | :--- |
| **Prediction** | *(Pending)* |
| **Latency** | N/A |
| **Accuracy** | {specs['test_acc']} |
| **Parameters** | {specs['params']} |
| **Status** | Pending File |
"""
        return label_dict, info_md


def classify_all_models(input_img):
    """Executes 4-way evaluation on the uploaded image."""
    if input_img is None:
        empty_info = "Please select or upload an image."
        return {}, empty_info, {}, empty_info, {}, empty_info, {}, empty_info
    
    if not isinstance(input_img, Image.Image):
        input_img = Image.fromarray(input_img)
    input_img = input_img.convert('RGB')
    
    # 1. Custom CNN
    cnn_specs = {'test_acc': '67.24%', 'params': '424,006'}
    cnn_label, cnn_info = predict_single_model('cnn', 'Custom CNN', 'cnn', input_img, cnn_specs)
    
    # 2. ResNet50
    resnet_specs = {'test_acc': '~87.50%', 'params': '~25.6M'}
    res_label, res_info = predict_single_model('resnet', 'ResNet50', 'resnet', input_img, resnet_specs)
    
    # 3. MobileNetV2 (Our model)
    mob_specs = {'test_acc': '88.09%', 'params': '2,270,790'}
    mob_label, mob_info = predict_single_model('mobilenet', 'MobileNetV2', 'mobilenet', input_img, mob_specs)
    
    # 4. EfficientNetB0
    eff_specs = {'test_acc': '~88.20%', 'params': '~5.3M'}
    eff_label, eff_info = predict_single_model('effnet', 'EfficientNetB0', 'effnet', input_img, eff_specs)
    
    return (
        cnn_label, cnn_info,
        res_label, res_info,
        mob_label, mob_info,
        eff_label, eff_info
    )


# Collect sample images for gallery
examples_list = []
ex_dir = APP_DIR / 'examples'
if ex_dir.exists():
    for f in sorted(ex_dir.glob('*.jpg')):
        examples_list.append([str(f)])

custom_css = """
.header-box { text-align: center; margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid var(--border-color-primary, #e2e8f0); }
.center-col { max-width: 620px; margin: 0 auto; }
.model-card { border: 1px solid var(--border-color-primary, #e2e8f0); border-radius: 8px; padding: 14px; background: var(--background-fill-secondary, transparent); }
.model-card table { width: 100%; border-collapse: collapse; margin-top: 8px; }
.model-card th, .model-card td { padding: 6px 10px; font-size: 0.9rem; border: 1px solid var(--border-color-primary, #e2e8f0); }
.model-card * { color: var(--body-text-color, inherit) !important; }
"""

with gr.Blocks(title="Automated Waste Classification System") as demo:
    gr.HTML("""
    <div class="header-box">
        <h2 style="font-size: 1.8rem; margin-bottom: 0.2rem; font-weight: 600;">
            Automated Waste Classification System
        </h2>
        <p style="font-size: 0.95rem; margin-top: 0; opacity: 0.8;">
            SE4050 Deep Learning | Multi-Model Architecture Evaluation
        </p>
    </div>
    """)
    
    # Centered Input Section
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2, elem_classes=["center-col"]):
            input_image = gr.Image(
                label="Input Image",
                type="pil",
                sources=["upload", "webcam", "clipboard"]
            )
            classify_btn = gr.Button("Classify Image", variant="primary", size="lg")
            
            if examples_list:
                gr.Examples(
                    examples=examples_list,
                    inputs=input_image,
                    label="Sample Test Images"
                )
        with gr.Column(scale=1):
            pass
            
    gr.HTML("<hr style='border: 0; border-top: 1px solid var(--border-color-primary, #e2e8f0); margin: 28px 0 20px 0;'>")
    gr.Markdown("### Comparative Model Predictions")
    
    # 4-Column Side-by-Side Model Results
    with gr.Row():
        # 1. Custom CNN
        with gr.Column(elem_classes=["model-card"]):
            gr.Markdown("#### 1. Custom CNN\n*Trained from scratch*")
            cnn_out_label = gr.Label(label="Probabilities", num_top_classes=3)
            cnn_out_info = gr.Markdown()
            
        # 2. ResNet50
        with gr.Column(elem_classes=["model-card"]):
            gr.Markdown("#### 2. ResNet50\n*Transfer Learning*")
            res_out_label = gr.Label(label="Probabilities", num_top_classes=3)
            res_out_info = gr.Markdown()
            
        # 3. MobileNetV2
        with gr.Column(elem_classes=["model-card"]):
            gr.Markdown("#### 3. MobileNetV2\n*Fine-Tuned Specialized*")
            mob_out_label = gr.Label(label="Probabilities", num_top_classes=3)
            mob_out_info = gr.Markdown()
            
        # 4. EfficientNetB0
        with gr.Column(elem_classes=["model-card"]):
            gr.Markdown("#### 4. EfficientNetB0\n*Transfer Learning*")
            eff_out_label = gr.Label(label="Probabilities", num_top_classes=3)
            eff_out_info = gr.Markdown()

    classify_btn.click(
        fn=classify_all_models,
        inputs=[input_image],
        outputs=[
            cnn_out_label, cnn_out_info,
            res_out_label, res_out_info,
            mob_out_label, mob_out_info,
            eff_out_label, eff_out_info
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css)
