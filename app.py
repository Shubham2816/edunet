import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Dictionary mapping class indices to disease names and descriptions
class_info = {
    'Apple___Apple_scab': {
        'disease': 'Apple Scab',
        'description': 'A fungal disease caused by Venturia inaequalis that affects apple trees, causing dark, scabby lesions on leaves and fruit.',
        'treatment': 'Apply fungicides in early spring, practice good sanitation by removing fallen leaves, and choose resistant apple varieties when planting new trees.'
    },
    'Apple___Black_rot': {
        'disease': 'Black Rot',
        'description': 'A fungal disease caused by Botryosphaeria obtusa affecting apples, causing circular lesions on leaves and rotting of fruit.',
        'treatment': 'Prune out dead or diseased wood, remove mummified fruit, apply fungicides during the growing season, and maintain good air circulation.'
    },
    'Apple___Cedar_apple_rust': {
        'disease': 'Cedar Apple Rust',
        'description': 'A fungal disease caused by Gymnosporangium juniperi-virginianae that requires both apple trees and cedar trees to complete its life cycle.',
        'treatment': 'Remove nearby cedar trees if possible, apply fungicides in spring, and plant resistant apple varieties.'
    },
    'Apple___healthy': {
        'disease': 'Healthy',
        'description': 'This apple plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Blueberry___healthy': {
        'disease': 'Healthy',
        'description': 'This blueberry plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'disease': 'Powdery Mildew',
        'description': 'A fungal disease that affects cherry trees, causing a white powdery coating on leaves and stems.',
        'treatment': 'Apply fungicides, ensure good air circulation by proper pruning, and avoid overhead watering.'
    },
    'Cherry_(including_sour)___healthy': {
        'disease': 'Healthy',
        'description': 'This cherry plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'disease': 'Gray Leaf Spot',
        'description': 'A fungal disease caused by Cercospora zeae-maydis that affects corn, causing rectangular lesions on leaves.',
        'treatment': 'Rotate crops, till residue, plant resistant hybrids, and apply fungicides when necessary.'
    },
    'Corn_(maize)___Common_rust_': {
        'disease': 'Common Rust',
        'description': 'A fungal disease caused by Puccinia sorghi that produces small, circular to elongate brown pustules on corn leaves.',
        'treatment': 'Plant resistant hybrids, apply fungicides, and monitor fields regularly during the growing season.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'disease': 'Northern Leaf Blight',
        'description': 'A fungal disease caused by Exserohilum turcicum resulting in long, elliptical lesions on corn leaves.',
        'treatment': 'Plant resistant hybrids, rotate crops, till infected residue, and apply fungicides if necessary.'
    },
    'Corn_(maize)___healthy': {
        'disease': 'Healthy',
        'description': 'This corn/maize plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Grape___Black_rot': {
        'disease': 'Black Rot',
        'description': 'A fungal disease caused by Guignardia bidwellii that affects grapes, causing brown circular lesions on leaves and rotting of fruit.',
        'treatment': 'Remove mummified berries, prune to improve air circulation, apply fungicides starting at bud break, and maintain a clean vineyard.'
    },
    'Grape___Esca_(Black_Measles)': {
        'disease': 'Esca (Black Measles)',
        'description': 'A complex fungal disease affecting grapevines, causing tiger-stripe patterns on leaves and black spots on fruit.',
        'treatment': 'Remove and destroy infected vines, avoid pruning during wet weather, protect pruning wounds, and consider trunk renewal.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'disease': 'Leaf Blight',
        'description': 'A fungal disease caused by Pseudocercospora vitis (formerly Isariopsis leaf spot) causing dark, irregular spots on grape leaves.',
        'treatment': 'Apply fungicides, ensure good air circulation, remove infected leaves, and maintain vineyard hygiene.'
    },
    'Grape___healthy': {
        'disease': 'Healthy',
        'description': 'This grape plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'disease': 'Citrus Greening (HLB)',
        'description': 'A bacterial disease spread by psyllid insects, causing mottled leaves, stunted growth, and bitter, misshapen fruit.',
        'treatment': 'Control psyllid populations, remove infected trees, plant disease-free nursery stock, and follow quarantine regulations.'
    },
    'Peach___Bacterial_spot': {
        'disease': 'Bacterial Spot',
        'description': 'A bacterial disease causing small, dark lesions on leaves, fruit, and twigs of peach trees.',
        'treatment': 'Apply copper-based bactericides, prune during dry weather, and plant resistant varieties.'
    },
    'Peach___healthy': {
        'disease': 'Healthy',
        'description': 'This peach plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'disease': 'Bacterial Spot',
        'description': 'A bacterial disease causing small, dark lesions on leaves and fruit of pepper plants.',
        'treatment': 'Rotate crops, use disease-free seeds, apply copper-based bactericides, and avoid overhead irrigation.'
    },
    'Pepper,_bell___healthy': {
        'disease': 'Healthy',
        'description': 'This bell pepper plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Potato___Early_blight': {
        'disease': 'Early Blight',
        'description': 'A fungal disease caused by Alternaria solani that affects potato plants, causing dark spots with concentric rings on leaves.',
        'treatment': 'Apply fungicides, practice crop rotation, ensure adequate plant nutrition, and remove infected plant debris.'
    },
    'Potato___Late_blight': {
        'disease': 'Late Blight',
        'description': 'A devastating water mold disease caused by Phytophthora infestans, which can rapidly kill potato plants and rot tubers.',
        'treatment': 'Apply fungicides preventatively, plant resistant varieties, destroy volunteer plants, and harvest tubers during dry weather.'
    },
    'Potato___healthy': {
        'disease': 'Healthy',
        'description': 'This potato plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Raspberry___healthy': {
        'disease': 'Healthy',
        'description': 'This raspberry plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Soybean___healthy': {
        'disease': 'Healthy',
        'description': 'This soybean plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Squash___Powdery_mildew': {
        'disease': 'Powdery Mildew',
        'description': 'A fungal disease causing a white powdery coating on the leaves and stems of squash plants.',
        'treatment': 'Apply fungicides, ensure proper spacing for air circulation, and plant resistant varieties when available.'
    },
    'Strawberry___Leaf_scorch': {
        'disease': 'Leaf Scorch',
        'description': 'A fungal disease caused by Diplocarpon earlianum, producing small purple spots that develop into brown lesions on strawberry leaves.',
        'treatment': 'Remove infected leaves, ensure good air circulation, avoid overhead irrigation, and apply fungicides when necessary.'
    },
    'Strawberry___healthy': {
        'disease': 'Healthy',
        'description': 'This strawberry plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    },
    'Tomato___Bacterial_spot': {
        'disease': 'Bacterial Spot',
        'description': 'A bacterial disease causing small, dark lesions on leaves, stems, and fruit of tomato plants.',
        'treatment': 'Rotate crops, use disease-free seeds, apply copper-based bactericides, and avoid overhead irrigation.'
    },
    'Tomato___Early_blight': {
        'disease': 'Early Blight',
        'description': 'A fungal disease caused by Alternaria solani, producing dark spots with concentric rings on older tomato leaves.',
        'treatment': 'Apply fungicides, practice crop rotation, remove lower infected leaves, and maintain good air circulation.'
    },
    'Tomato___Late_blight': {
        'disease': 'Late Blight',
        'description': 'A devastating water mold disease caused by Phytophthora infestans that can rapidly kill tomato plants.',
        'treatment': 'Apply fungicides preventatively, ensure good air circulation, avoid overhead watering, and remove infected plants immediately.'
    },
    'Tomato___Leaf_Mold': {
        'disease': 'Leaf Mold',
        'description': 'A fungal disease caused by Passalora fulva (formerly Fulvia fulva) that affects tomato leaves, especially in high humidity conditions.',
        'treatment': 'Improve greenhouse ventilation, reduce humidity, remove infected leaves, and apply fungicides if necessary.'
    },
    'Tomato___Septoria_leaf_spot': {
        'disease': 'Septoria Leaf Spot',
        'description': 'A fungal disease caused by Septoria lycopersici, producing numerous small, circular spots with dark borders on tomato leaves.',
        'treatment': 'Apply fungicides, practice crop rotation, remove infected leaves, and avoid overhead watering.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'disease': 'Spider Mites',
        'description': 'Tiny arachnids that cause yellow stippling on tomato leaves and can create fine webbing in severe infestations.',
        'treatment': 'Spray plants with water, apply insecticidal soap or miticide, introduce predatory mites, and maintain proper plant hydration.'
    },
    'Tomato___Target_Spot': {
        'disease': 'Target Spot',
        'description': 'A fungal disease caused by Corynespora cassiicola, producing brown circular lesions with concentric rings on tomato leaves and fruit.',
        'treatment': 'Apply fungicides, improve air circulation, avoid overhead watering, and practice crop rotation.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'disease': 'Yellow Leaf Curl Virus',
        'description': 'A viral disease spread by whiteflies, causing yellowing, curling, and stunting of tomato leaves and plants.',
        'treatment': 'Control whitefly populations, remove and destroy infected plants, use reflective mulches, and plant resistant varieties.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'disease': 'Mosaic Virus',
        'description': 'A viral disease causing mottled green and yellow patterns on tomato leaves and sometimes stunted growth.',
        'treatment': 'Remove and destroy infected plants, control insect vectors, disinfect tools, and plant resistant varieties.'
    },
    'Tomato___healthy': {
        'disease': 'Healthy',
        'description': 'This tomato plant shows no signs of disease and appears to be in good health.',
        'treatment': 'Continue regular maintenance, including appropriate watering, fertilization, and preventive care.'
    }
}

@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        model = tf.keras.models.load_model('plant_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model prediction."""
    if image is None:
        return None
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(model, image):
    """Predict the plant disease."""
    # List of class names
    class_names = list(class_info.keys())
    
    # Make prediction
    prediction = model.predict(image)
    
    # Get the class index with highest probability
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    
    # Get the class name and probability
    predicted_class_name = class_names[predicted_class_idx]
    confidence = float(prediction[0][predicted_class_idx]) * 100
    
    return predicted_class_name, confidence

def main():
    # Add custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
    }
    .medium-font {
        font-size:30px !important;
        font-weight: bold;
        color: #1e88e5;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .healthy {
        background-color: #c8e6c9;
    }
    .diseased {
        background-color: #ffccbc;
    }
    </style>
    """, unsafe_allow_html=True)

    # Application title
    st.markdown('<p class="big-font">🌿 Plant Disease Detection System 🌿</p>', unsafe_allow_html=True)
    
    st.write("""
    This application uses a deep learning model to detect diseases in plant leaves.
    Upload an image of a plant leaf and get instant disease diagnosis!
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if the model file exists and is valid.")
        return
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Add a button to trigger prediction
            if st.button('Analyze Image'):
                with st.spinner('Analyzing...'):
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Make prediction
                        predicted_class, confidence = predict_disease(model, processed_image)
                        
                        # Display results in the second column
                        with col2:
                            # Determine if the plant is healthy or diseased
                            is_healthy = "healthy" in predicted_class.lower()
                            box_class = "healthy" if is_healthy else "diseased"
                            
                            st.markdown(f'<div class="result-box {box_class}">', unsafe_allow_html=True)
                            
                            # Get the disease information
                            info = class_info[predicted_class]
                            
                            st.markdown(f'<p class="medium-font">Detection Result:</p>', unsafe_allow_html=True)
                            st.write(f"**Detected Condition:** {info['disease']}")
                            st.write(f"**Confidence:** {confidence:.2f}%")
                            
                            st.markdown("---")
                            st.markdown("### Description:")
                            st.write(info['description'])
                            
                            st.markdown("---")
                            st.markdown("### Treatment Recommendation:")
                            st.write(info['treatment'])
                            
                            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This application uses a Convolutional Neural Network (CNN) trained on a dataset of plant leaves.
        
        **Model Architecture:**
        - Input: RGB images of size 224x224 pixels
        - Multiple convolutional and pooling layers
        - Fully connected layers
        - Output: 38 classes representing different plant diseases
        
        **Supported Plants:**
        Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
        
        **Performance:**
        - Accuracy: ~90%
        - Precision: ~92%
        - Recall: ~89%
        """)
    
    # Add a FAQ section
    with st.expander("Frequently Asked Questions"):
        st.write("""
        **Q: How accurate is this model?**
        
        A: The model has approximately 90% accuracy on test data, but results may vary based on image quality and conditions.
        
        **Q: What types of plants can this model analyze?**
        
        A: Currently, the model can identify diseases in: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
        
        **Q: Why isn't my plant's disease being detected correctly?**
        
        A: The accuracy of detection depends on several factors:
        - Image quality and lighting
        - The angle at which the leaf is photographed
        - Whether the disease symptoms are clearly visible
        - If the disease is in an early stage, symptoms might not be distinct enough
        
        **Q: Can I use this for commercial purposes?**
        
        A: This is a demonstration application and should not be used as the sole basis for commercial agricultural decisions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🌱 Plant health is crucial for sustainable agriculture and food security!")
    st.write("This tool is made by Shubham Kumar and being completely meant to be educational and should not replace professional agricultural advice.")

if __name__ == "__main__":
    main()