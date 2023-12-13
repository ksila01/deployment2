import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import tensorflow_hub as hub
import os
import time

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
    st.title("Kenya's Best Farm App")
    st.subheader(" A one-step to detect leaf disease and get evidence-based solutions.")
    # Specify the path to the folder containing your images
    folder_path = "images"
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    # Create a button to advance to the next image
    next_button = st.button("Next Image")
    # Display the initial image
    current_index = 0
    image_path = os.path.join(folder_path, image_files[current_index])
    image_element = st.image(image_path, caption=f'Caption for {image_files[current_index]}', use_column_width=True)
    # Update the image when the button is clicked
    if next_button:
        current_index = (current_index + 1) % len(image_files)
        image_path = os.path.join(folder_path, image_files[current_index])
        image_element.image(image_path, caption=f'Caption for {image_files[current_index]}', use_column_width=True)

    st.sidebar.title('Please')
    st.sidebar.markdown('Upload an image and get predictions.')

def get_remedy(disease_class):
    remedies = {
        'Potato__Early_blight': " Use copper-based fungicides, practice crop rotation, and ensure proper spacing between plants.",
        'Potato__Late_blight': " Apply fungicides containing chlorothalonil, practice good garden hygiene, and avoid overhead watering.",
        'Potato__healthy': "No specific remedy needed for healthy leaves. Maintain proper plant nutrition and good gardening practices.",
    }
    return remedies.get(disease_class, "No specific remedy available.")

def predict_class(image):
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'potatoes.h5', compile=False)

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

def main():
    # File uploader
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')

    if file_uploaded:
        # Prediction
        result, confidence = predict_class(file_uploaded)

        # Check if the predicted class is a potato leaf class
        potato_leaf_classes = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
        if result in potato_leaf_classes:
            # Display uploaded image
            col1, col2 = st.columns(2)
            with col1:
                st.image(file_uploaded, caption='Uploaded Image', use_column_width=True)

            # Display prediction information
            with col2:
                st.subheader('Prediction:')
                st.write(f'Class: {result}')
                st.write(f'Confidence: {confidence}%')

                # Display remedy information
                remedy = get_remedy(result)
                st.subheader('Remedy:')
                st.write(remedy)
        else:
            # Display rejection message
            st.subheader('Prediction:')
            st.write('Not a potato leaf')
            st.subheader('Remedy:')
            st.write('Please upload an image of a potato leaf.')

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