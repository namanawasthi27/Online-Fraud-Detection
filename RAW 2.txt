import numpy as np
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import drive

# ============ MOUNT GOOGLE DRIVE ============
drive.mount('/content/drive')

# Specify the path to the ZIP file in Google Drive
zip_path = '/content/drive/MyDrive/Colab Notebooks/onlinefraud.zip'

# Validate if the uploaded file is a ZIP file and exists
if not zip_path.endswith('.zip'):
    print(f"❌ The specified file '{zip_path}' is not a ZIP file. Please check the file path.")
    data = None
elif not os.path.exists(zip_path):
    print("❌ ZIP file not found. Check the file path!")
    data = None
else:
    try:
        # Open and read the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as z:
            print("📂 Files in ZIP:", z.namelist())

            # Read the CSV file inside the ZIP
            csv_file = 'onlinefraud.csv'  # Update this if needed
            if csv_file in z.namelist():
                with z.open(csv_file) as f:
                    data = pd.read_csv(f)
                    print("\n📊 Data preview:")
                    print(data.head())
                    print("🔢 Data shape:", data.shape)
            else:
                print("❌ CSV file not found in the ZIP!")
                data = None
    except zipfile.BadZipFile:
        print("❌ Error: The file is not a valid ZIP file or is corrupted.")
        data = None

# ======================================================================================================
# Proceed only if data is loaded successfully
if data is not None:
    # ============== DATA CLEANING ==============
    print("\n🛠 Checking for missing values...")
    print(data.isnull().sum())

    # Drop irrelevant columns
    data = data.drop(columns=['nameOrig', 'nameDest'], errors='ignore')

    # Remove duplicates
    before = data.shape[0]
    data = data.drop_duplicates().copy()  # Ensure a deep copy to avoid SettingWithCopyWarning
    after = data.shape[0]
    print(f"🗑 Removed {before - after} duplicate rows.")

    # Ensure correct data types using .loc[]
    print("\n🔍 Data types before conversion:")
    print(data.dtypes)

    data.loc[:, 'step'] = data['step'].astype(int)
    data.loc[:, 'isFraud'] = data['isFraud'].astype(int)

    print("\n✅ Data types after conversion:")
    print(data.dtypes)

    print("\n📏 Data shape after cleaning:", data.shape)
    print("\n📌 Cleaned data preview:")
    print(data.head())

    # ================= DATA VISUALIZATION =================
    plt.style.use('ggplot')

    # Fraud distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='isFraud', data=data)
    plt.title('Fraudulent vs Non-Fraudulent Transactions')
    plt.xlabel('Is Fraud')
    plt.ylabel('Count')
    plt.show()

    # Transaction type distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='type', hue='isFraud', data=data)
    plt.title('Transaction Types and Fraud Occurrences')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.show()

    # Amount distribution by fraud status
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='isFraud', y='amount', data=data)
    plt.ylim(0, 100000)
    plt.title('Transaction Amount Distribution by Fraud Status')
    plt.show()

    # ================= FEATURE ENGINEERING =================
    print("\n🔧 Encoding categorical variables...")

    # Encode 'type' column using one-hot encoding
    data = pd.get_dummies(data, columns=['type'], drop_first=True)

    print("✅ Feature encoding completed.")
    print(data.head())

    # ================= SPLITTING DATA =================
    print("\n📊 Splitting data into training and testing sets...")

    X = data.drop(columns=['isFraud'])  # Features
    y = data['isFraud']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("✅ Data split completed.")
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")

    # ================= MODIFY HYPERPARAMETERS AND SHUFFLE DATA =================
    print("\n🚀 Training models with suboptimal hyperparameters and shuffled labels...")

    # Introduce label noise by shuffling target labels (for training set)
    y_train_shuffled = y_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Decision Tree Model with suboptimal parameters
    model_DT = DecisionTreeClassifier(max_depth=1, random_state=42)  # Shallow tree
    model_DT.fit(X_train, y_train_shuffled)

    # Random Forest Model with suboptimal parameters
    model_RF = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=0)  # Few trees, shallow
    model_RF.fit(X_train, y_train_shuffled)

    # Support Vector Machine (SVM) Model
    model_SVM = SVC(kernel='rbf', random_state=42)
    model_SVM.fit(X_train, y_train)

    print("✅ Model training completed.")

    # ================= MODEL EVALUATION =================
    print("\n📈 Evaluating model performance...")

    # Predictions
    y_pred_DT = model_DT.predict(X_test)
    y_pred_RF = model_RF.predict(X_test)
    y_pred_SVM = model_SVM.predict(X_test)

    # Compute accuracy
    accuracy_DT = accuracy_score(y_test, y_pred_DT)
    accuracy_RF = accuracy_score(y_test, y_pred_RF)
    accuracy_SVM = accuracy_score(y_test, y_pred_SVM)

    # Classification reports & confusion matrices
    report_DT = classification_report(y_test, y_pred_DT)
    conf_matrix_DT = confusion_matrix(y_test, y_pred_DT)

    report_RF = classification_report(y_test, y_pred_RF)
    conf_matrix_RF = confusion_matrix(y_test, y_pred_RF)

    report_SVM = classification_report(y_test, y_pred_SVM)
    conf_matrix_SVM = confusion_matrix(y_test, y_pred_SVM)

    # Print results
    print(f"🎯 Decision Tree Accuracy: {accuracy_DT:.4f}")
    print("\n📊 Decision Tree Classification Report:\n", report_DT)
    print("\n🔢 Decision Tree Confusion Matrix:\n", conf_matrix_DT)

    print(f"🎯 Random Forest Accuracy: {accuracy_RF:.4f}")
    print("\n📊 Random Forest Classification Report:\n", report_RF)
    print("\n🔢 Random Forest Confusion Matrix:\n", conf_matrix_RF)

    print(f"🎯 SVM Accuracy: {accuracy_SVM:.4f}")
    print("\n📊 SVM Classification Report:\n", report_SVM)
    print("\n🔢 SVM Confusion Matrix:\n", conf_matrix_SVM)

    # ================= CONFUSION MATRIX PLOTS =================
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_DT, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Decision Tree Confusion Matrix')
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_RF, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest Confusion Matrix')
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_SVM, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('SVM Confusion Matrix')
    plt.show()
