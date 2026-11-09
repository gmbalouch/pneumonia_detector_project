import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/pneumonia_model.keras"
SAMPLES_PATH = "data/test_samples.npz"
LABELS = ["Normal", "Pneumonia"]
N_SAMPLES = 5

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")


# ---------- helpers ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_samples():
    """Test images (N,28,28 uint8) and labels (N,) saved by make_samples.py."""
    try:
        data = np.load(SAMPLES_PATH)
        return data["images"], data["labels"].reshape(-1).astype(int)
    except FileNotFoundError:
        return None, None


def preprocess(img: Image.Image) -> np.ndarray:
    """Same preprocessing as training: grayscale, 28x28, scaled to 0-1."""
    img = img.convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1)


def thumb(arr: np.ndarray, size: int = 140) -> Image.Image:
    return Image.fromarray(arr).resize((size, size), Image.BICUBIC)


def new_batch(labels: np.ndarray) -> list:
    """Pick 5 random test images, always a mix of normal and pneumonia."""
    rng = np.random.default_rng()
    n_normal = int(rng.integers(2, 4))          # 2 or 3 normal
    n_pneu = N_SAMPLES - n_normal               # 3 or 2 pneumonia
    normal_idx = rng.choice(np.where(labels == 0)[0], n_normal, replace=False)
    pneu_idx = rng.choice(np.where(labels == 1)[0], n_pneu, replace=False)
    batch = np.concatenate([normal_idx, pneu_idx])
    rng.shuffle(batch)
    return [int(i) for i in batch]


# ---------- page ----------
st.title("🫁 Pneumonia Detector")
st.caption("A small CNN trained on PneumoniaMNIST (28×28 chest X-rays).")
st.warning(
    "Educational demo only. This is **not** a medical device and must not be "
    "used for diagnosis. Consult a qualified doctor."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.05,
        help="Lower = catches more pneumonia cases but more false alarms.",
    )
    st.markdown("**Model:** 2× Conv + Dense CNN, ~56K parameters")
    st.markdown("**Test accuracy:** ~83% (PneumoniaMNIST test split)")

images, labels = load_samples()

if "selected" not in st.session_state:
    st.session_state.selected = None
if "batch" not in st.session_state and labels is not None:
    st.session_state.batch = new_batch(labels)

# ---------- upload section ----------
st.subheader("1. Upload an X-ray")
uploaded = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

# ---------- sample gallery (only when nothing is uploaded) ----------
if uploaded is None:
    st.subheader("…or test on a random sample")
    if images is None:
        st.info(
            f"Sample images not found. Run `python make_samples.py` to create "
            f"`{SAMPLES_PATH}`."
        )
    else:
        top = st.columns([3, 1])
        top[0].caption("Randomly picked from the test data (a mix of Normal and Pneumonia). "
                       "Click **Use this** to send one to the analyzer.")
        if top[1].button("🔀 Shuffle", use_container_width=True):
            st.session_state.batch = new_batch(labels)
            st.session_state.selected = None
            st.rerun()

        cols = st.columns(N_SAMPLES)
        for col, idx in zip(cols, st.session_state.batch):
            with col:
                st.image(thumb(images[idx]), use_container_width=True)
                chosen = st.session_state.selected == idx
                st.caption("✅ Selected" if chosen else f"Sample #{idx}")
                if st.button("Use this", key=f"use_{idx}", use_container_width=True,
                             type="primary" if chosen else "secondary"):
                    st.session_state.selected = idx
                    st.rerun()

# ---------- decide which image to analyze ----------
image, source, true_label = None, None, None
if uploaded is not None:
    image, source = Image.open(uploaded), "upload"
elif st.session_state.selected is not None and images is not None:
    idx = st.session_state.selected
    image, source, true_label = Image.fromarray(images[idx]), "sample", int(labels[idx])

# ---------- analyze ----------
st.subheader("2. Analyze")
if image is None:
    st.info("Upload an X-ray or pick a sample above.")
else:
    col1, col2 = st.columns(2)
    col1.image(image if source == "upload" else thumb(np.asarray(image), 224),
               caption="Uploaded image" if source == "upload" else f"Test sample #{st.session_state.selected}",
               use_container_width=True)
    col2.image(thumb(np.asarray(image.convert("L").resize((28, 28), Image.LANCZOS))),
               caption="What the model sees (28×28)")

    if source == "sample" and st.button("✖ Clear selection"):
        st.session_state.selected = None
        st.rerun()

    if st.button("Analyze", type="primary"):
        model = load_model()
        prob = float(model.predict(preprocess(image), verbose=0)[0][0])
        label = LABELS[int(prob >= threshold)]

        if label == "Pneumonia":
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")
        st.metric("Pneumonia probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

        if true_label is not None:
            truth = LABELS[true_label]
            if truth == label:
                st.success(f"✔ Correct. The true label is **{truth}**.")
            else:
                st.warning(f"✘ Wrong. The true label is **{truth}**.")
