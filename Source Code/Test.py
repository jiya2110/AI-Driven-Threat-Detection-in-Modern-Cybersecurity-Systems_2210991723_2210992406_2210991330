"""
AI-Driven Threat Detection in Cybersecurity Systems
Complete Experimental Framework for IEEE Research Paper

This script performs:
1. NSL-KDD dataset download and preprocessing
2. Training of 7 ML algorithms (RF, SVM, DT, XGBoost, KNN, Naive Bayes, ANN)
3. Both Binary and Multi-class classification
4. Comprehensive evaluation metrics
5. Publication-ready visualizations
6. Results exported to CSV/JSON for paper

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
import json
from datetime import datetime
import urllib.request
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# XGBoost
from xgboost import XGBClassifier

# TensorFlow/Keras for ANN
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Handle imbalanced data
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

# NSL-KDD Column Names
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Attack type mapping
ATTACK_CATEGORIES = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
    'udpstorm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}


def download_nslkdd_data(data_dir='./data'):
    """Download NSL-KDD dataset if not already present"""
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    files = ['KDDTrain+.txt', 'KDDTest+.txt']
    
    print("=" * 80)
    print("DOWNLOADING NSL-KDD DATASET")
    print("=" * 80)
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(base_url + filename, filepath)
                print(f"✓ {filename} downloaded successfully")
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")
                print(f"Please manually download from: {base_url}{filename}")
                return False
        else:
            print(f"✓ {filename} already exists")
    
    print("\n")
    return True


def load_and_preprocess_data(data_dir='./data', use_smote=False):
    """Load and preprocess NSL-KDD dataset"""
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    # Load datasets
    train_file = os.path.join(data_dir, 'KDDTrain+.txt')
    test_file = os.path.join(data_dir, 'KDDTest+.txt')
    
    print("Loading training data...")
    train_df = pd.read_csv(train_file, names=COLUMN_NAMES, header=None)
    
    print("Loading test data...")
    test_df = pd.read_csv(test_file, names=COLUMN_NAMES, header=None)
    
    print(f"Training set size: {train_df.shape}")
    print(f"Test set size: {test_df.shape}")
    
    # Remove difficulty column
    train_df = train_df.drop('difficulty', axis=1)
    test_df = test_df.drop('difficulty', axis=1)
    
    # Map attack types to categories (for multi-class)
    train_df['attack_category'] = train_df['label'].apply(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))
    test_df['attack_category'] = test_df['label'].apply(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))
    
    # Create binary labels (Normal vs Attack)
    train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Display class distribution
    print("\n--- Binary Classification Distribution ---")
    print("Training set:")
    print(train_df['binary_label'].value_counts())
    print("\nTest set:")
    print(test_df['binary_label'].value_counts())
    
    print("\n--- Multi-class Classification Distribution ---")
    print("Training set:")
    print(train_df['attack_category'].value_counts())
    print("\nTest set:")
    print(test_df['attack_category'].value_counts())
    
    # Encode categorical features
    print("\nEncoding categorical features...")
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    for col in categorical_columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        # Handle unseen categories in test set
        test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col] = le.transform(test_df[col])
    
    # Separate features and labels
    feature_columns = [col for col in train_df.columns if col not in ['label', 'attack_category', 'binary_label']]
    
    X_train = train_df[feature_columns].values
    X_test = test_df[feature_columns].values
    
    # Binary labels
    y_train_binary = train_df['binary_label'].values
    y_test_binary = test_df['binary_label'].values
    
    # Multi-class labels
    le_multi = LabelEncoder()
    y_train_multi = le_multi.fit_transform(train_df['attack_category'])
    y_test_multi = le_multi.transform(test_df['attack_category'])
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE if requested (only on training data)
    if use_smote:
        print("Applying SMOTE for handling class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train_binary = smote.fit_resample(X_train_scaled, y_train_binary)
        print(f"After SMOTE - Training set size: {X_train_scaled.shape}")
    
    print("\n✓ Data preprocessing complete!\n")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train_binary': y_train_binary,
        'y_test_binary': y_test_binary,
        'y_train_multi': y_train_multi,
        'y_test_multi': y_test_multi,
        'feature_names': feature_columns,
        'class_names_multi': le_multi.classes_
    }


#==============================================================================
# SECTION 2: MODEL DEFINITIONS
#==============================================================================

def create_ann_model(input_dim, num_classes, task='binary'):
    """Create Artificial Neural Network model"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes if task == 'multi' else 1, 
                    activation='softmax' if task == 'multi' else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy' if task == 'multi' else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_dict(task='binary', num_classes=5):
    """Get dictionary of all models to train"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
    }
    
    return models


#==============================================================================
# SECTION 3: TRAINING AND EVALUATION
#==============================================================================

def train_and_evaluate_models(data, task='binary'):
    """Train and evaluate all models for given task"""
    print("=" * 80)
    print(f"TRAINING MODELS - {task.upper()} CLASSIFICATION")
    print("=" * 80)
    
    X_train = data['X_train']
    X_test = data['X_test']
    
    if task == 'binary':
        y_train = data['y_train_binary']
        y_test = data['y_test_binary']
        num_classes = 2
        class_names = ['Normal', 'Attack']
    else:
        y_train = data['y_train_multi']
        y_test = data['y_test_multi']
        num_classes = len(data['class_names_multi'])
        class_names = data['class_names_multi']
    
    models = get_model_dict(task, num_classes)
    results = {}
    
    # Train traditional ML models
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        start_pred = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_pred
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        if task == 'binary':
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        else:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate False Positive Rate and Detection Rate for binary
        if task == 'binary':
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            fpr = None
            detection_rate = None
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'training_time': train_time,
            'prediction_time': pred_time,
            'y_pred': y_pred,
            'fpr': fpr,
            'detection_rate': detection_rate
        }
        
        print(f"✓ Accuracy: {accuracy:.4f}")
        print(f"✓ Precision: {precision:.4f}")
        print(f"✓ Recall: {recall:.4f}")
        print(f"✓ F1-Score: {f1:.4f}")
        print(f"✓ Training Time: {train_time:.2f}s")
        print(f"✓ Prediction Time: {pred_time:.4f}s")
        if task == 'binary':
            print(f"✓ False Positive Rate: {fpr:.4f}")
            print(f"✓ Detection Rate: {detection_rate:.4f}")
    
    # Train ANN model
    print(f"\n{'='*60}")
    print("Training Artificial Neural Network...")
    print(f"{'='*60}")
    
    ann_model = create_ann_model(X_train.shape[1], num_classes, task)
    
    start_time = time.time()
    
    if task == 'binary':
        history = ann_model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=128,
            validation_split=0.2,
            verbose=0
        )
    else:
        # One-hot encode for multi-class
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        history = ann_model.fit(
            X_train, y_train_cat,
            epochs=20,
            batch_size=128,
            validation_split=0.2,
            verbose=0
        )
    
    train_time = time.time() - start_time
    
    # Predictions
    start_pred = time.time()
    if task == 'binary':
        y_pred_prob = ann_model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        y_pred_prob = ann_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
    
    pred_time = time.time() - start_pred
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    if task == 'binary':
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    else:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    cm = confusion_matrix(y_test, y_pred)
    
    if task == 'binary':
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        fpr = None
        detection_rate = None
    
    results['ANN'] = {
        'model': ann_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'training_time': train_time,
        'prediction_time': pred_time,
        'y_pred': y_pred,
        'history': history,
        'fpr': fpr,
        'detection_rate': detection_rate
    }
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")
    print(f"✓ Training Time: {train_time:.2f}s")
    print(f"✓ Prediction Time: {pred_time:.4f}s")
    if task == 'binary':
        print(f"✓ False Positive Rate: {fpr:.4f}")
        print(f"✓ Detection Rate: {detection_rate:.4f}")
    
    print("\n✓ All models trained successfully!\n")
    
    return results, y_test, class_names


#==============================================================================
# SECTION 4: VISUALIZATION
#==============================================================================

def create_visualizations(results_binary, results_multi, y_test_binary, y_test_multi, 
                         class_names_multi, output_dir='./results'):
    """Create all publication-ready visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Performance Comparison - Binary Classification
    print("Creating performance comparison charts...")
    
    metrics_binary = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for name, result in results_binary.items():
        metrics_binary['Model'].append(name)
        metrics_binary['Accuracy'].append(result['accuracy'])
        metrics_binary['Precision'].append(result['precision'])
        metrics_binary['Recall'].append(result['recall'])
        metrics_binary['F1-Score'].append(result['f1_score'])
    
    df_metrics_binary = pd.DataFrame(metrics_binary)
    
    # Plot binary classification metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_metrics_binary))
    width = 0.2
    
    ax.bar(x - 1.5*width, df_metrics_binary['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, df_metrics_binary['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, df_metrics_binary['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, df_metrics_binary['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Machine Learning Algorithms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison - Binary Classification (Normal vs Attack)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics_binary['Model'], rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binary_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Comparison - Multi-class Classification
    metrics_multi = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for name, result in results_multi.items():
        metrics_multi['Model'].append(name)
        metrics_multi['Accuracy'].append(result['accuracy'])
        metrics_multi['Precision'].append(result['precision'])
        metrics_multi['Recall'].append(result['recall'])
        metrics_multi['F1-Score'].append(result['f1_score'])
    
    df_metrics_multi = pd.DataFrame(metrics_multi)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_metrics_multi))
    
    ax.bar(x - 1.5*width, df_metrics_multi['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, df_metrics_multi['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, df_metrics_multi['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, df_metrics_multi['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Machine Learning Algorithms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison - Multi-class Classification (Attack Type Detection)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics_multi['Model'], rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiclass_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Training Time Comparison
    print("Creating training time comparison...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = list(results_binary.keys())
    train_times_binary = [results_binary[m]['training_time'] for m in models]
    train_times_multi = [results_multi[m]['training_time'] for m in models]
    
    ax1.barh(models, train_times_binary, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Binary Classification', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    ax2.barh(models, train_times_multi, color='coral', alpha=0.7)
    ax2.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Multi-class Classification', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    fig.suptitle('Training Time Comparison Across Algorithms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confusion Matrices (3x3 grid for best models)
    print("Creating confusion matrices...")
    
    # Select top 3 models by accuracy for binary
    top_models_binary = sorted(results_binary.items(), 
                               key=lambda x: x[1]['accuracy'], 
                               reverse=True)[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confusion Matrices - Binary Classification (Top 3 Models)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (name, result) in enumerate(top_models_binary):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        axes[idx].set_title(f'{name}\nAcc: {result["accuracy"]:.3f}', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_binary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix for best multi-class model
    best_model_multi = max(results_multi.items(), key=lambda x: x[1]['accuracy'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = best_model_multi[1]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
               xticklabels=class_names_multi,
               yticklabels=class_names_multi)
    ax.set_title(f'Confusion Matrix - Multi-class Classification\n{best_model_multi[0]} (Acc: {best_model_multi[1]["accuracy"]:.3f})', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_multiclass_best.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detection Rate vs False Positive Rate (Binary only)
    print("Creating Detection Rate vs FPR plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, result in results_binary.items():
        if result['fpr'] is not None and result['detection_rate'] is not None:
            ax.scatter(result['fpr'], result['detection_rate'], 
                      s=200, alpha=0.7, label=name, edgecolors='black', linewidth=1.5)
            ax.annotate(name, 
                       (result['fpr'], result['detection_rate']),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=9)
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Rate (True Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Detection Rate vs False Positive Rate - Binary Classification', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, max([r['fpr'] for r in results_binary.values() if r['fpr'] is not None]) + 0.05)
    ax.set_ylim(0.85, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_vs_fpr.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Accuracy Ranking
    print("Creating accuracy ranking chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by accuracy
    df_sorted_binary = df_metrics_binary.sort_values('Accuracy', ascending=True)
    df_sorted_multi = df_metrics_multi.sort_values('Accuracy', ascending=True)
    
    colors_binary = plt.cm.RdYlGn(df_sorted_binary['Accuracy'])
    colors_multi = plt.cm.RdYlGn(df_sorted_multi['Accuracy'])
    
    ax1.barh(df_sorted_binary['Model'], df_sorted_binary['Accuracy'], color=colors_binary, alpha=0.8)
    ax1.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Binary Classification Accuracy Ranking', fontsize=12, fontweight='bold')
    ax1.set_xlim(0.85, 1.0)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(df_sorted_binary['Accuracy']):
        ax1.text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=9)
    
    ax2.barh(df_sorted_multi['Model'], df_sorted_multi['Accuracy'], color=colors_multi, alpha=0.8)
    ax2.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Multi-class Classification Accuracy Ranking', fontsize=12, fontweight='bold')
    ax2.set_xlim(0.65, 1.0)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(df_sorted_multi['Accuracy']):
        ax2.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=9)
    
    fig.suptitle('Model Performance Ranking', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ All visualizations saved to {output_dir}/\n")


#==============================================================================
# SECTION 5: RESULTS EXPORT
#==============================================================================

def export_results(results_binary, results_multi, output_dir='./results'):
    """Export results to CSV and JSON for paper"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    # Binary Classification Results
    binary_data = []
    for name, result in results_binary.items():
        binary_data.append({
            'Algorithm': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'Training Time (s)': result['training_time'],
            'Prediction Time (s)': result['prediction_time'],
            'False Positive Rate': result['fpr'] if result['fpr'] is not None else 'N/A',
            'Detection Rate': result['detection_rate'] if result['detection_rate'] is not None else 'N/A'
        })
    
    df_binary = pd.DataFrame(binary_data)
    df_binary = df_binary.sort_values('Accuracy', ascending=False)
    df_binary.to_csv(os.path.join(output_dir, 'binary_classification_results.csv'), index=False)
    print(f"✓ Binary classification results saved to binary_classification_results.csv")
    
    # Multi-class Classification Results
    multi_data = []
    for name, result in results_multi.items():
        multi_data.append({
            'Algorithm': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'Training Time (s)': result['training_time'],
            'Prediction Time (s)': result['prediction_time']
        })
    
    df_multi = pd.DataFrame(multi_data)
    df_multi = df_multi.sort_values('Accuracy', ascending=False)
    df_multi.to_csv(os.path.join(output_dir, 'multiclass_classification_results.csv'), index=False)
    print(f"✓ Multi-class classification results saved to multiclass_classification_results.csv")
    
    # Export detailed JSON
    detailed_results = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'NSL-KDD',
            'algorithms': list(results_binary.keys()),
            'classification_types': ['Binary', 'Multi-class']
        },
        'binary_classification': {
            name: {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'training_time': float(result['training_time']),
                'prediction_time': float(result['prediction_time']),
                'false_positive_rate': float(result['fpr']) if result['fpr'] is not None else None,
                'detection_rate': float(result['detection_rate']) if result['detection_rate'] is not None else None,
                'confusion_matrix': result['confusion_matrix'].tolist()
            }
            for name, result in results_binary.items()
        },
        'multiclass_classification': {
            name: {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'training_time': float(result['training_time']),
                'prediction_time': float(result['prediction_time']),
                'confusion_matrix': result['confusion_matrix'].tolist()
            }
            for name, result in results_multi.items()
        }
    }
    
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=4)
    print(f"✓ Detailed results saved to detailed_results.json")
    
    # Print summary tables
    print("\n" + "=" * 80)
    print("BINARY CLASSIFICATION RESULTS SUMMARY")
    print("=" * 80)
    print(df_binary.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("MULTI-CLASS CLASSIFICATION RESULTS SUMMARY")
    print("=" * 80)
    print(df_multi.to_string(index=False))
    
    print("\n✓ All results exported successfully!\n")


#==============================================================================
# MAIN EXECUTION
#==============================================================================

def main():
    """Main execution function"""
    print("\n")
    print("=" * 80)
    print(" AI-DRIVEN THREAT DETECTION IN CYBERSECURITY SYSTEMS")
    print(" Experimental Framework for IEEE Research Paper")
    print("=" * 80)
    print(f" Author: Jashit")
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n")
    
    # Step 1: Download data
    if not download_nslkdd_data():
        print("ERROR: Could not download dataset. Please download manually.")
        return
    
    # Step 2: Load and preprocess data
    data = load_and_preprocess_data(use_smote=False)
    
    # Step 3: Train models for binary classification
    results_binary, y_test_binary, _ = train_and_evaluate_models(data, task='binary')
    
    # Step 4: Train models for multi-class classification
    results_multi, y_test_multi, class_names_multi = train_and_evaluate_models(data, task='multi')
    
    # Step 5: Create visualizations
    create_visualizations(results_binary, results_multi, y_test_binary, y_test_multi, 
                         class_names_multi)
    
    # Step 6: Export results
    export_results(results_binary, results_multi)
    
    print("=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\n📊 Results saved in ./results/ directory:")
    print("  - CSV files: binary_classification_results.csv, multiclass_classification_results.csv")
    print("  - JSON file: detailed_results.json")
    print("  - Visualizations: *.png files")
    print("\n✓ You can now use these results in your IEEE research paper!")
    print("\n")


if __name__ == "__main__":
    main()
