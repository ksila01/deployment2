import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import tensorflow_hub as hub

# Set the background color to light blue
st.markdown(
    """
    <style>
        body {
            background-color: #add8e6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set page title and description
st.title('Potato Leaf Disease Prediction')
with st.sidebar:
    st.image('img.jpeg')
    st.title("Kenya's Best Farm App")
    st.subheader("A one step to detect leaf disease and get evidence-based solutions.")
st.sidebar.title('Options')
st.sidebar.markdown('Upload an image and get predictions.')

def get_remedy(disease_class):
    remedies = {
        'Potato__Early_blight': " Use copper-based fungicides, practice crop rotation, and ensure proper spacing between plants.",
        'Potato__Late_blight': " Apply fungicides containing chlorothalonil, practice good garden hygiene, and avoid overhead watering.",
        'Potato__healthy': "No specific remedy needed for healthy leaves. Maintain proper plant nutrition and good gardening practices.",
    }
    return remedies.get(disease_class, "No specific remedy available.")

def main():
    # File uploader
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')

    if file_uploaded:
        # Display uploaded image in the first column
        col1, col2 = st.columns(2)
        with col1:
            st.image(file_uploaded, caption='Uploaded Image', use_column_width=True)

        # Prediction in the second column
        with col2:
            # Prediction
            result, confidence = predict_class(file_uploaded)
            st.subheader('Prediction:')
            st.write(f'Class: {result}')
            st.write(f'Confidence: {confidence}%')

            # Display remedy information
            remedy = get_remedy(result)
            st.subheader('Remedy:')
            st.write(remedy)

def predict_class(image):
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'base_cnn_model.h5', compile=False)

    shape = (256, 256, 3)
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])

    # Preprocess the image
    test_image = Image.open(image).resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    class_names = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

    # Make predictions
    prediction = model.predict(test_image)
    confidence = round(100 * np.max(prediction[0]), 2)
    final_pred = class_names[np.argmax(prediction)]

    return final_pred, confidence

# Footer
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px;
}
</style>

<div class="footer">
    <p>Developed by Kelvin Sila</p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()