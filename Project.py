import os
import numpy as np
import librosa
import sounddevice as sd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import smtplib
from email.mime.text import MIMEText

# Define the dataset directory
dataset_dir = "C:/Users/AMIT/Desktop/newnewcryfinal2023/data_set"

# Define the labels for the 5 /categories
labels = ['hunger', 'lowergas', 'burp', 'discomfort', 'sleepy']

# Define the number of MFCC coefficients to extract
num_mfcc =3000
# Load the data and extract MFCC features
X = []
y = []
for label in labels:
    label_dir = os.path.join(dataset_dir, label)
    for file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file)
        audio, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        X.append(mfccs_mean)
        y.append(label)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the SVM model
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = svm.predict(X_test)

# Calculate the accuracy of the model 
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)


# Record a new audio file and predict its label
duration = 5 # seconds
fs = 22050  # sample rate
print("Starting the cry recording...")
while True:
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    # Check if there is any sound above the threshold level
    if np.max(np.abs(recording)) >.05:
        new_mfccs = librosa.feature.mfcc(y=recording[:, 0], sr=fs, n_mfcc=num_mfcc)
        new_mfccs_mean = np.mean(new_mfccs.T, axis=0)
        new_prediction = svm.predict([new_mfccs_mean])
        print("The predicted label is:", new_prediction[0])
        # Send email with the predicted label
        email_address = 'majoraksav@gmail.com'
        email_password = 'ywknyhxydzkfqjoi'
        to_email_address = 'amitambaji2@gmail.com'
        subject = 'Crying Baby Alert'
        message = f"The baby is crying because of {new_prediction[0]}."
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = email_address
        msg['To'] = to_email_address
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_address, email_password)
        server.sendmail(email_address, to_email_address, msg.as_string())
        server.quit()
    else:
        print("No crying detected, waiting for sound...")
