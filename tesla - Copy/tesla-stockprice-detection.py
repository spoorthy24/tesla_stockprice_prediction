import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')


# Load the dataset
df = pd.read_csv("C:/Users/ukshr/OneDrive/Desktop/tesla/TSLA.csv")

df = df.drop(['Adj Close'], axis=1)

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['is_quarter_end'] = np.where(df['Date'].dt.is_quarter_end, 1, 0)

# Create the 'target' column
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Visualizations
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Tesla Close Price', fontsize=15)
plt.ylabel('Price in Dollars')
plt.show()

features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.distplot(df[col], bins=20, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(data=df, y=col)
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.title('Correlation Matrix')
plt.show()

# Prepare features and target
features = df[['Open', 'Close', 'Volume', 'day', 'month', 'year', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)

# Model training and evaluation
models = [LogisticRegression(), SVC(kernel='poly', probability=True), RandomForestClassifier()]

for model in models:
    model.fit(X_train, Y_train)
    print(f'{model} : ')
    print('Training AUC : ', roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]))
    print('Validation AUC : ', roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]))
    print()

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(Y_valid, model.predict(X_valid))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model}')
    plt.show()

    # AUC graph
    plt.figure(figsize=(10, 6))
    Y_pred_proba = model.predict_proba(X_valid)[:, 1]
    fpr, tpr, _ = roc_curve(Y_valid, Y_pred_proba)
    auc = roc_auc_score(Y_valid, Y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model}')
    plt.legend(loc="lower right")
    plt.show()
