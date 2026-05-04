# Import all of the dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf
import subprocess
import cv2
import numpy as np
from utils import load_data, num_to_char
from modelutil import load_model
import base64

# Set layout
st.set_page_config(layout='wide')

# Title
st.title('LipSense: Decoding Speech')

# Load the model once globally
model = load_model()

# List video options
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Create two columns
col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.abspath(os.path.join('..', 'data', 's1', selected_video))
        output_dir = os.path.abspath('app')
        os.makedirs(output_dir, exist_ok=True)
        converted_filename = os.path.join(output_dir, f"{os.path.splitext(selected_video)[0]}_converted.mp4")

        # FFmpeg executable path
        ffmpeg_path = r"C:\ffmpeg\ffmpeg-2025-06-02-git-688f3944ce-full_build\bin\ffmpeg.exe"

        # Convert if not already converted
        if not os.path.exists(converted_filename):
            result = subprocess.run(
                [ffmpeg_path, '-i', file_path, '-vcodec', 'libx264', '-y', converted_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                st.error("ffmpeg failed to convert the video.")
                with st.expander("Show ffmpeg error details"):
                    st.code(result.stderr)
            else:
                st.success(" Video successfully converted.")

        # Show video if available
        if os.path.exists(converted_filename):
            with open(converted_filename, 'rb') as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode()
                video_html = f"""
                <video width="400" height="300" controls>
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """
                st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Converted video file not found.")

    with col2:
        st.info('This is what the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400)

        # Predict with the model
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.success(f"Predicted Text: {converted_prediction}")

# -------------------------------------------
# Webcam-based Lip Reading
# -------------------------------------------

# Function to capture 75 frames from webcam with preview window
def capture_webcam_frames(frame_count=75, width=140, height=46):
    st.info("Starting webcam... A preview window will open. Please stay centered and speak clearly.")
    cap = cv2.VideoCapture(0)
    frames = []
    frame_index = 0

    while frame_index < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # flip horizontally for selfie view

        # Display webcam preview with progress
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Recording frame {frame_index+1}/{frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Recording", display_frame)

        # Preprocess frame for model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height))
        normalized = resized / 255.0
        expanded = np.expand_dims(normalized, axis=-1)
        frames.append(expanded)

        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Pad with black frames if needed
    while len(frames) < frame_count:
        frames.append(np.zeros((height, width, 1)))

    st.success("✅ Webcam capture complete.")
    return np.expand_dims(np.array(frames[:75]), axis=0)

# Decode prediction
def decode_prediction(pred):
    input_len = tf.constant([pred.shape[1]])
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = tf.strings.reduce_join(num_to_char(results)).numpy().decode('utf-8')
    return output_text

# Real-time webcam lip reading section
st.markdown("---")
st.header("🎥 Real-Time Lip Reading from Webcam")

if st.button("Start Webcam Capture"):
    with st.spinner("Capturing and processing..."):
        webcam_input = capture_webcam_frames()
        webcam_prediction = model.predict(webcam_input)
        webcam_result = decode_prediction(webcam_prediction)
        st.success(f" Predicted Text from Webcam: {webcam_result}")
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera could not be opened.")
else:
    print("✅ Camera opened successfully. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
