import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import time 
import sounddevice as sd
import soundfile as sf
import tempfile
import os

# 1. Preprocessing Function
def preprocess_voice(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# 2. Simple Neural Network Model
class SimpleVoiceRecognitionModel(nn.Module):
    def __init__(self):
        super(SimpleVoiceRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer for 3 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Training the Model
def train_model():
    # Example: Process your three voice samples
    voice1_features = preprocess_voice('nirupam_audio.mp3')
    voice2_features = preprocess_voice('aryan_audio.mp3')
    voice3_features = preprocess_voice('ashutosh_audio.mp3')

    # Prepare the training data
    train_data = np.array([voice1_features, voice2_features, voice3_features])
    labels = np.array([0, 1, 2])  # Assign unique labels to each voice

    model = SimpleVoiceRecognitionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert data to tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Training loop
    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()
        outputs = model(train_data_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')

    return model

# 4. Prediction Function

# def predict(model, file_path):
#     features = preprocess_voice(file_path)
#     features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
#     with torch.no_grad():
#         output = model(features_tensor)
#     predicted_label = torch.argmax(output, dim=1).item()
#     return predicted_label

def predict(model, audio_data, sample_rate=16000):
    # Convert audio data to MFCC features
    mfcc = librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0)
    
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(features_tensor)
    predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label

# def main():
#     st.title("Voice Recognition System")

#     model = train_model()  # Train the model when the app starts

#     st.header("Option 1: Upload a voice sample")
#     uploaded_file = st.file_uploader("Upload a voice sample", type=["wav", "mp3"])

#     if uploaded_file is not None:
#         st.audio(uploaded_file, format="audio/wav")
#         predicted_label = predict(model, uploaded_file)
#         if predicted_label == 0:  # Adjust according to your labels
#             st.success("Unlocked!")
#         else:
#             st.error("Access Denied.")

#     st.header("Option 2: Record your voice")
    
#     # Add a button to start recording
#     if st.button("Start Recording"):
#         with st.spinner("Recording for 5 seconds..."):
#             audio_data = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
#             sd.wait()
        
#         st.success("Recording completed!")
        
#         # Save the recorded audio to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#             sf.write(temp_audio.name, audio_data, 16000)
        
#         # Play back the recorded audio
#         st.audio(temp_audio.name)
        
#         # Make prediction
#         predicted_label = predict(model, audio_data.flatten(), 16000)
        
#         if predicted_label == 0:  # Adjust according to your labels
#             st.success("Voice recognized! Unlocked!")
#         else:
#             st.error("Voice not recognized. Access Denied.")
        
#         # Clean up the temporary file
#         os.unlink(temp_audio.name)

# if __name__ == "__main__":
#     main()


def main():
    st.title("Voice Recognition System")

    model = train_model()  # Train the model when the app starts

    st.header("Option 1: Upload a voice sample")
    uploaded_file = st.file_uploader("Upload a voice sample", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        audio, sr = librosa.load(uploaded_file, sr=None)
        predicted_label = predict(model, audio, sr)
        if predicted_label == 0:  # Adjust according to your labels
            st.success("Hurray its a match! It's Nirupam !!")
        if predicted_label == 1:
            st.success("Hurray its a match! It's Aryan!")
        if predicted_label == 2:
            st.success("Hurray its a match ! It's Ashutosh !")
        else:
            st.error("Access Denied.")

    st.header("Option 2: Record your voice")
    
    # Add a button to start recording
    if st.button("Start Recording"):
        with st.spinner("Recording for 5 seconds..."):
            audio_data = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
            sd.wait()
        
        st.success("Recording completed!")
        
        # Save the recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            sf.write(temp_audio.name, audio_data, 16000)
        
        # Play back the recorded audio
        st.audio(temp_audio.name)
        
        # Make prediction
        predicted_label = predict(model, audio_data)
        
        if predicted_label == 0:  # Adjust according to your labels
            st.success("Hurray its a match! It's Nirupam !!")
        if predicted_label == 1:
            st.success("Hurray its a match! It's Aryan!")
        if predicted_label == 2:
            st.success("Hurray its a match ! It's Ashutosh !")
        else:
            st.error("Voice not Authorised!!!")
        
        # Clean up the temporary file
        os.unlink(temp_audio.name)

if __name__ == "__main__":
    main()