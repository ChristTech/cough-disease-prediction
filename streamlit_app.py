import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
model = joblib.load('cough_random.pkl')

# Define function to make predictions
def predict_disease(features):
    prediction = model.predict([features])
    return prediction[0]

# Define function to load and prepare encoder
def load_encoder():
    encoders = {}
    # Example encoders for the selected features
    encoders['Age'] = LabelEncoder()
    encoders['Age'].fit([str(i) for i in range(1, 101)])  # Dummy fit, adjust range or categories as needed

    encoders['Shortness of breath'] = LabelEncoder()
    encoders['Shortness of breath'].fit(['Yes', 'No'])

    encoders['Periodic asthmatic suffering'] = LabelEncoder()
    encoders['Periodic asthmatic suffering'].fit(['Yes', 'No'])

    encoders['Nocturnal episode of dyspnea'] = LabelEncoder()
    encoders['Nocturnal episode of dyspnea'].fit(['Yes', 'No'])

    encoders['Chest pain'] = LabelEncoder()
    encoders['Chest pain'].fit(['Yes', 'No'])

    encoders['Weight loss'] = LabelEncoder()
    encoders['Weight loss'].fit(['Yes', 'No'])

    encoders['Malaise'] = LabelEncoder()
    encoders['Malaise'].fit(['Yes', 'No'])

    encoders['Chills'] = LabelEncoder()
    encoders['Chills'].fit(['Yes', 'No'])

    encoders['Tiredness'] = LabelEncoder()
    encoders['Tiredness'].fit(['Yes', 'No'])

    encoders['Anorexia'] = LabelEncoder()
    encoders['Anorexia'].fit(['Yes', 'No'])

    encoders['Condition of Cough'] = LabelEncoder()
    encoders['Condition of Cough'].fit(['Productive', 'Unproductive'])

    encoders['sore throat'] = LabelEncoder()
    encoders['sore throat'].fit(['Yes', 'No'])

    return encoders

# Load encoders
encoders = load_encoder()

# Streamlit UI
st.set_page_config(page_title="Cough disease Prediction App", page_icon="🔬")
st.title('Disease Prediction App')

# CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("C:\\Users\\adebi\\Desktop\\cough-type-chronic-disease-prediction\\background.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields
age = st.number_input('Age', min_value=1, max_value=100, value=30)
shortness_of_breath = st.selectbox('Shortness of Breath', ['Yes', 'No'])
periodic_asthmatic_suffering = st.selectbox('Periodic Asthmatic Suffering', ['Yes', 'No'])
nocturnal_episode_of_dyspnea = st.selectbox('Nocturnal Episode of Dyspnea', ['Yes', 'No'])
chest_pain = st.selectbox('Chest Pain', ['Yes', 'No'])
weight_loss = st.selectbox('Weight Loss', ['Yes', 'No'])
malaise = st.selectbox('Malaise', ['Yes', 'No'])
chills = st.selectbox('Chills', ['Yes', 'No'])
tiredness = st.selectbox('Tiredness', ['Yes', 'No'])
anorexia = st.selectbox('Anorexia', ['Yes', 'No'])
condition_of_cough = st.selectbox('Condition of Cough', ['Productive', 'Unproductive'])
sore_throat = st.selectbox('Sore Throat', ['Yes', 'No'])

# Collecting inputs
input_data = {
    'Age': age,
    'Shortness of breath': shortness_of_breath,
    'Periodic asthmatic suffering': periodic_asthmatic_suffering,
    'Nocturnal episode of dyspnea': nocturnal_episode_of_dyspnea,
    'Chest pain': chest_pain,
    'Weight loss': weight_loss,
    'Malaise': malaise,
    'Chills': chills,
    'Tiredness': tiredness,
    'Anorexia': anorexia,
    'Condition of Cough': condition_of_cough,
    'sore throat': sore_throat
}

# Convert inputs to the same format as the model expects
input_features = [
    encoders['Age'].transform([str(input_data['Age'])])[0],  # Assuming 'Age' is categorical
    encoders['Shortness of breath'].transform([input_data['Shortness of breath']])[0],
    encoders['Periodic asthmatic suffering'].transform([input_data['Periodic asthmatic suffering']])[0],
    encoders['Nocturnal episode of dyspnea'].transform([input_data['Nocturnal episode of dyspnea']])[0],
    encoders['Chest pain'].transform([input_data['Chest pain']])[0],
    encoders['Weight loss'].transform([input_data['Weight loss']])[0],
    encoders['Malaise'].transform([input_data['Malaise']])[0],
    encoders['Chills'].transform([input_data['Chills']])[0],
    encoders['Tiredness'].transform([input_data['Tiredness']])[0],
    encoders['Anorexia'].transform([input_data['Anorexia']])[0],
    encoders['Condition of Cough'].transform([input_data['Condition of Cough']])[0],
    encoders['sore throat'].transform([input_data['sore throat']])[0]
]

# Ensure the number of features matches the model's expectations
if len(input_features) == model.n_features_in_:  # Assuming model.n_features_in_ is available
    if st.button('Predict'):
        prediction = predict_disease(input_features)

        if prediction == 0:
            st.write(f'Predicted Disease: No disease')

        if prediction == 1:
            st.write(f'Predicted Disease: COPD')

        if prediction == 2:
            st.write(f'Predicted Disease: BRONCHIAL ASTHMA')

        if prediction == 3:
            st.write(f'Predicted Disease: PNEUMONIA')
else:
    st.error("Feature shape mismatch: Please ensure that all required features are provided.")
