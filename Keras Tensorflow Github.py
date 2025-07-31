#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error


# In[2]:


# Load dataset
df = pd.read_csv('creditcard.csv')


# In[3]:


# EDA
print(df.shape)
print(df.head())
print(df.isnull().sum())

df.hist(bins=20, figsize=(20,15))
plt.show()

sns.countplot(x='Class', data=df)
plt.show()


# In[4]:


# Preprocess data
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape, X_test.shape)


# In[5]:


# Label encode target
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


# In[6]:


# Keras model
model = keras.Sequential()

model.add(keras.layers.Dense(16, input_shape=(30,), activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[7]:


# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)


# In[8]:


# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

y_pred = model.predict(X_test)


# In[9]:


mse = mean_squared_error(y_test, y_pred)
print(mse)
print(accuracy)


# In[10]:


# ROC-AUC Score
y_pred_prob = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print('ROC-AUC Score:', roc_auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[11]:


# Confusion Matrix
# Predict probabilities for each class
y_pred_prob = model.predict(X_test)

# Threshold probabilities to obtain class labels (0 or 1)
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred))


# In[12]:


# Save model
model.save('creditcard_model.h5')

