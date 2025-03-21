#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


df=pd.read_csv('Churn_Modelling.csv')


# In[3]:


print(df.info())
print(df.head())


# In[4]:


df.fillna(df.median(), inplace=True)


# In[5]:


df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True, errors='ignore')


# In[6]:


label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# In[7]:


X = df.drop(columns=['Exited'])  # 'Exited' is the churn label
y = df['Exited']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



# In[11]:


y_pred = model.predict(X_test)


# In[12]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[13]:


feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()


# In[ ]:




