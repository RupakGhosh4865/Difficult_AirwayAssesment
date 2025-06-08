import os
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ---- Paths & Model Setup ---
base_dir = r"D:\rupak try\data"
MODEL_PATH = os.path.join(base_dir, "model_intubation.pt")
CLASS_NAMES_PATH = os.path.join(base_dir, "class_names.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local airway example images (update filenames as needed)
IMG_DIR = os.path.join(os.path.dirname(__file__), "images")
EXAMPLE_NEUTRAL = os.path.join(IMG_DIR, "6.jpg")
EXAMPLE_TONGUE = os.path.join(IMG_DIR, "5.jpg")
EXAMPLE_HEADUP = os.path.join(IMG_DIR, "4.jpg")

# ---- Streamlit App Header ----
st.set_page_config(
    page_title="AI Intubation Difficulty Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Intubation Difficulty Predictor (AI-Assisted)")
st.markdown("""
AI support for preoperative and emergency airway assessment.
Upload clinical photos and get an instant risk prediction—**Easy** or **Difficult**—with confidence scores and full transparency.
""")

# ---- About Intubation Section ----
st.subheader("What is Intubation?")
st.write("""
**Intubation** is a critical medical procedure to secure a patient's airway by inserting a tube through the mouth into the trachea, ensuring proper ventilation. It is essential for:
- Major surgical procedures under general anesthesia
- Emergency situations like respiratory failure, trauma, or cardiac arrest
- Managing critically ill patients in intensive care units

**Predicting difficulty in intubation is crucial** for patient safety:
- Difficult intubation can lead to increased attempts, prolonged procedure time, or complications such as hypoxia.
- Early recognition enables preparation with advanced airway tools and expert personnel.
""")
st.write("""
**Difficult intubation** implies:
- More attempts or prolonged intubation time
- Need for special devices such as video laryngoscopes, flexible scopes, or introducers (bougies)
- Necessity of backup airway strategies and coordinated team efforts
""")

# ---- About the AI and Its Parameters ----
st.markdown("---")
st.subheader("How does the AI model work?")

st.write("""
- **Input:** Three standardized airway/face photographs per patient:
    1. Neutral position (facing camera, mouth closed)
    2. Tongue fully extended (facing camera, mouth open)
    3. Head elevated in the “sniffing” position (neck extended)
- **Model:** A deep convolutional neural network based on [ResNet18](https://arxiv.org/abs/1512.03385), trained on a large dataset of labeled airway images (easy vs difficult intubation) enhanced with data augmentation techniques:
    - RandomResizedCrop (scale 0.7–1.0)
    - RandomHorizontalFlip
    - ColorJitter
    - RandomRotation (±15 degrees)
    - Trained with batch size 8 for 10 epochs using Adam optimizer (learning rate 1e-4)
    - Standard mean and standard deviation normalization applied
- **Output:** Probability scores for “Easy” and “Difficult” intubation based on the combined image analysis
- **Purpose:** To aid education, training, and clinical awareness — not to replace comprehensive clinical airway assessment or professional judgment
""")

# ----------- Example Images Section ----------------
with st.expander("🖼️ See Example Airway Input Photos (best practice)", expanded=False):
    coln, colt, colh = st.columns(3)
    with coln:
        st.image(EXAMPLE_NEUTRAL, caption="Neutral Face (Mouth Closed, Facing Forward)", use_container_width=True)
    with colt:
        st.image(EXAMPLE_TONGUE, caption="Tongue Fully Extended (Mouth Open, Facing Forward)", use_container_width=True)
    with colh:
        st.image(EXAMPLE_HEADUP, caption="Head Up — Sniffing Position (Neck Extended)", use_container_width=True)
    st.caption(
        "Capture images in good lighting with the entire face, jaw, tongue, and neck clearly visible. Use a neutral, uncluttered medical background when possible."
    )

# ----------- AI Prediction Section ----------
st.markdown("---")
st.header("Upload Images for AI Prediction")

st.markdown("""
Upload the three clinical photos as described above (neutral, tongue-out, head-up). The AI model will analyze the combined images for anatomical risk markers and provide a risk prediction with confidence scores.

- Intended for training, skill development, and increasing airway risk awareness.
- **Not** a substitute for thorough airway examination by trained clinicians.
""")

with st.form("upload_form", clear_on_submit=False):
    form_cols = st.columns(3)
    with form_cols[0]:
        up1 = st.file_uploader("Neutral Face", type=["jpg", "jpeg", "png"], key="imgNeutral")
    with form_cols[1]:
        up2 = st.file_uploader("Tongue Out", type=["jpg", "jpeg", "png"], key="imgTongue")
    with form_cols[2]:
        up3 = st.file_uploader("Head Up", type=["jpg", "jpeg", "png"], key="imgHeadup")
    submit_button = st.form_submit_button(label='Predict Intubation Risk')

images = []
if up1: images.append(Image.open(up1).convert('RGB'))
if up2: images.append(Image.open(up2).convert('RGB'))
if up3: images.append(Image.open(up3).convert('RGB'))

@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip().capitalize() for line in f.readlines()]
    return model, class_names

model, class_names = load_model_and_classes()

if submit_button:
    if len(images) == 3:
        imgs_resized = [img.resize((224,224)) for img in images]
        collage = Image.new('RGB', (224*3, 224))
        for i, img in enumerate(imgs_resized):
            collage.paste(img, (i*224, 0))
        st.image(collage, caption="Your Combined Image Collage", use_container_width=True)

        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        input_tensor = preprocess(collage).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
            pred_idx = int(np.argmax(probs))
        st.markdown(f"""
            <div style="background:#e5fbe8;border-radius:4px;padding:1em;margin-bottom:0.6em;font-size:1.21em;">
            <span style="color:#16669c;font-weight:700">AI Prediction:</span>
            <span style="font-size:1.08em;font-weight:600;">{class_names[pred_idx]} Intubation</span>
            <br><span style="color:#289c58;font-size:1.01em;">(Confidence: {probs[pred_idx]*100:.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        st.write(f"**Easy:** {probs[0]*100:.2f}%  |  **Difficult:** {probs[1]*100:.2f}%")
        st.caption("Interpret results in context of bedside airway exams, patient history, BMI, and prior anesthesia records.")
    elif 0 < len(images) < 3:
        st.warning("Please upload all 3 photos (Neutral, Tongue Out, Head Up) to get a prediction.")

# ---------- Expanded Clinical Info Section ----------
st.markdown("---")
st.header("Clinical Background: Difficult Intubation")
st.markdown("""
- **Difficult intubation** means that securing the airway is not straightforward and requires additional time, expertise, or specialized equipment.
- **Common predictors** include limited mouth opening, large tongue, high Mallampati score, obesity, reduced neck mobility, small jaw, airway trauma, and history of difficult intubation.
- **Clinical scores** such as Mallampati classification, thyromental distance, interincisor gap, and neck mobility tests are widely used bedside tools to assess risk.

**Why anticipate difficulty?**
- To prevent serious complications like hypoxia, trauma, and procedural delays
- To prepare the appropriate equipment and team beforehand
- To improve safety, efficiency, and confidence during airway management

This AI tool aims to support clinical training and risk awareness but should never replace comprehensive clinical assessment and decision-making.
""")

# ---------- Responsible Use and Disclaimer ----------
st.markdown("---")
st.markdown(
    """
    <div style='border-radius:5px;padding:0.9em;background:#f2fafd;color:#155481;margin-bottom:1em'>
    <b>Disclaimer:</b> This app is designed for <b>educational and simulated clinical decision support only</b>.<br>
    It is <b>not</b> validated for real patient care and should never be used as the sole basis for airway management or anesthesia planning.
    </div>
    """,
    unsafe_allow_html=True
)
