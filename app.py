import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="SoilVision", page_icon="🌱", layout="wide")

# ================= CSS =================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#d1fae5,#bbf7d0);
}

/* Sidebar */
section[data-testid="stSidebar"]{
background:#064e3b !important;
}
section[data-testid="stSidebar"] *{
color:white !important;
}

/* Title */
.title-bar{
background:linear-gradient(90deg,#10b981,#34d399);
padding:20px;
border-radius:12px;
text-align:center;
font-size:32px;
font-weight:700;
color:white;
margin-bottom:20px;
}

/* Dashboard Box */
.result-box{
background:rgba(255,255,255,0.75);
backdrop-filter:blur(12px);
padding:25px;
border-radius:16px;
box-shadow:0 10px 30px rgba(0,0,0,0.15);
margin-top:20px;
}

/* Metric Cards */
.metric-box{
background:linear-gradient(135deg,#10b981,#06b6d4);
color:white;
padding:20px;
border-radius:14px;
text-align:center;
font-size:20px;
font-weight:600;
transition:0.3s;
}

.metric-box:hover{
transform:translateY(-5px) scale(1.03);
box-shadow:0 10px 25px rgba(0,0,0,0.2);
}

/* Info Cards */
.info-box{
background:rgba(255,255,255,0.95);
padding:16px;
border-radius:10px;
margin-bottom:10px;
color:#064e3b;
font-weight:500;
}

/* Progress bar */
.progress-bar{
height:20px;
background:#e5e7eb;
border-radius:10px;
overflow:hidden;
}

.progress-fill{
height:100%;
background:linear-gradient(90deg,#10b981,#22c55e);
text-align:right;
padding-right:5px;
color:white;
font-size:12px;
}

</style>
""", unsafe_allow_html=True)

# ================= MODEL =================
model = tf.keras.models.load_model("soil_model.h5", compile=False)
MODEL_ACCURACY = 84

classes = [
"Alluvial Soil","Arid Soil","Black Soil",
"Laterite Soil","Mountain Soil","Red Soil","Yellow Soil"
]

# ================= DATA =================
soil_info = {
"Alluvial Soil":"Found near river basins and very fertile.",
"Arid Soil":"Dry soil found in desert areas.",
"Black Soil":"Rich in clay and minerals.",
"Laterite Soil":"Formed in tropical regions.",
"Mountain Soil":"Found in hilly areas.",
"Red Soil":"Contains iron oxide.",
"Yellow Soil":"Formed in humid climates."
}

soil_quality = {
"Alluvial Soil":"High fertility",
"Arid Soil":"Low fertility",
"Black Soil":"Highly fertile",
"Laterite Soil":"Moderate fertility",
"Mountain Soil":"Moderate fertility",
"Red Soil":"Medium fertility",
"Yellow Soil":"Moderate fertility"
}

soil_crops = {
"Alluvial Soil":"Rice, Wheat",
"Arid Soil":"Millets",
"Black Soil":"Cotton",
"Laterite Soil":"Tea",
"Mountain Soil":"Coffee",
"Red Soil":"Pulses",
"Yellow Soil":"Maize"
}

soil_characteristics = {
"Alluvial Soil":{"Color":"Light Brown","Texture":"Loamy","Water Retention":"Moderate","Fertility":"High","Drainage":"Good"},
"Arid Soil":{"Color":"Pale Brown","Texture":"Sandy","Water Retention":"Low","Fertility":"Low","Drainage":"Fast"},
"Black Soil":{"Color":"Black","Texture":"Clayey","Water Retention":"High","Fertility":"High","Drainage":"Moderate"},
"Laterite Soil":{"Color":"Reddish","Texture":"Gravelly","Water Retention":"Low","Fertility":"Moderate","Drainage":"Fast"},
"Mountain Soil":{"Color":"Dark Brown","Texture":"Loamy","Water Retention":"Moderate","Fertility":"Moderate","Drainage":"Good"},
"Red Soil":{"Color":"Red","Texture":"Sandy","Water Retention":"Low","Fertility":"Medium","Drainage":"Fast"},
"Yellow Soil":{"Color":"Yellow","Texture":"Sandy","Water Retention":"Low","Fertility":"Medium","Drainage":"Good"}
}

# ================= UI =================
st.sidebar.title("🌱 SoilVision")
st.markdown('<div class="title-bar">🌱 SoilVision - Soil Classification Dashboard</div>', unsafe_allow_html=True)
st.sidebar.write("Deep Learning Soil Classification System")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Information")

st.sidebar.write("""
This system uses **Deep Learning (CNN)** to classify soil types from images.

Features:
- Soil Type Prediction
- Soil Quality Assessment
- Crop Recommendation
- Soil Characteristics
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset Statistics")
st.sidebar.write("Total Soil Types: 7")
st.sidebar.write("Training Images: 5382")
st.sidebar.write("Validation Images: 1341")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")
st.sidebar.write("Model: Convolutional Neural Network (CNN)")
st.sidebar.write("Framework: TensorFlow / Keras")
st.sidebar.write("Accuracy: ~84%")

st.sidebar.markdown("---")

st.sidebar.subheader("Created By :-")
st.sidebar.write("Group No :- G8")
st.sidebar.write("1) Om S. Dhadse")
st.sidebar.write("2) Suhani G. Kalsait")
st.sidebar.write("3) Himani Raut")

st.sidebar.subheader("Guided By :-")
st.sidebar.write("   Prof. V. V. Bais")






uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg","png","jpeg"])

if "result" not in st.session_state:
    st.session_state.result = None

# ================= PREDICTION =================
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, use_container_width=True)

    if st.button("Predict"):
        img = img.resize((128,128))
        arr = np.array(img)/255.0
        arr = np.expand_dims(arr,0)

        pred = model.predict(arr)
        result = classes[np.argmax(pred)]
        st.session_state.result = result

# ================= OUTPUT =================
if st.session_state.result:

    result = st.session_state.result
    char = soil_characteristics[result]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.markdown("## 🌱 Soil Intelligence Dashboard")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-box">Predicted Soil<br>{result}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box">Soil Fertility<br>{char["Fertility"]}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box">Accuracy<br>{MODEL_ACCURACY}%</div>', unsafe_allow_html=True)

    st.divider()

    # Main info
    left, right = st.columns(2)

    with left:
        st.subheader("📊 Soil Insights")

        st.markdown(f'<div class="info-box">{soil_info[result]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{soil_quality[result]}</div>', unsafe_allow_html=True)

        st.success(f"🌾 Recommended Crops: {soil_crops[result]}")

    with right:
        st.subheader("🌿 Soil Health Score")

        score_map = {"High":90,"Moderate":65,"Medium":60,"Low":40}
        score = score_map.get(char["Fertility"],50)

        st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width:{score}%">
                {score}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
        <b>AI Insight:</b><br>
        This soil shows <b>{char["Fertility"]}</b> fertility and is suitable for 
        <b>{soil_crops[result]}</b>. It has good agricultural potential.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Characteristics
    st.subheader("🧾 Soil Characteristics")

    cols = st.columns(3)
    for i, (k,v) in enumerate(char.items()):
        cols[i % 3].markdown(f'<div class="info-box">{k}: {v}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)