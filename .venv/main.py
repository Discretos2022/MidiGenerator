import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import librosa

# Improved feature extraction function with logging
def extract_features(file_path, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        if y.size == 0:
            print(f"Warning: Empty audio file {file_path}")
            return None

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        zcr_mean = np.mean(zcr)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rolloff_mean = np.mean(rolloff)

        # Combine all features into one vector
        features = np.hstack([mfcc_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean, zcr_mean, rolloff_mean])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Function to determine labels from the filename
def label_from_folder(filename):
    filename_lower = filename.lower()
    if "piano_guitare" in filename_lower:
        return [1, 1]
    elif "piano" in filename_lower:
        return [0, 1]
    elif "guitare" in filename_lower:
        return [1, 0]
    else:
        print(f"Skipped file due to unclear label: {filename}")
        return None

# Function to load data directly from a folder with debugging
def load_data_from_folder(folder_path):
    X = []
    y = []

    print(f"Loading data from folder: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return np.array(X), np.array(y)

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        print(f"Processing subfolder: {subfolder_path}")

        if os.path.isdir(subfolder_path):
            label = label_from_folder(subfolder)
            if label is not None:
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)

                    if file.lower().endswith(('.wav', '.mp3')):
                        print(f"Extracting features from file: {file_path}")
                        features = extract_features(file_path)
                        if features is not None:
                            X.append(features)
                            y.append(label)
                        else:
                            print(f"Failed to extract features from: {file_path}")
            else:
                print(f"Skipping subfolder due to unclear label: {subfolder}")
        else:
            print(f"Skipping non-directory entry: {subfolder_path}")

    if len(X) == 0:
        print("No valid audio files were loaded.")
    else:
        print(f"Loaded {len(X)} samples from {folder_path}")

    return np.array(X), np.array(y)

# Load the data
folder_path = "res/musicForModel"
X, y = load_data_from_folder(folder_path)

# Check the data size
print(f"Number of samples in X: {len(X)}")
print(f"Number of samples in y: {len(y)}")

# Check if data is empty before splitting
if len(X) == 0 or len(y) == 0:
    print("Error: No data available for training.")
    exit()

# Transform labels to multi-label binary format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

# Split the dataset
print("Splitting the dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 'log2', None],  # Corrected values for max_features
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Starting Random Forest training with Grid Search...")
clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)
print("Training completed.")

# Save the trained model, scaler, and multi-label binarizer
model_dir = 'res/model'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(clf, os.path.join(model_dir, 'random_forest_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(mlb, os.path.join(model_dir, 'mlb.pkl'))
print("Model and supporting files saved successfully.")

# Prediction and Evaluation
print("Making predictions on the test set...")
y_pred = clf.predict(X_test)

# Print the classification report with string labels
target_names = [str(label) for label in mlb.classes_]
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
