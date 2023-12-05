import streamlit as st
from PIL import Image
import numpy as np

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Potato Leaf Disease Prediction')

model = None
class_names = ["Early Blight", "Late Blight", "Healthy"]
BUCKET_NAME = "mawaru"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "model/potatoes.h5",
            "/tmp/potatoes.h5",
        )
        model = tf.keras.models.load_model("/tmp/potatoes.h5")

    image = Image.open(request).convert("RGB").resize((256, 256))
    image = np.array(image) / 255

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return {"class": predicted_class, "confidence": confidence}

footer = """
<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}

a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>

<div class="footer">
<p align="center"> Developed Kelvin Sila </a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)


