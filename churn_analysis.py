import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir(r'C:\Users\Pranav Shajan\Desktop\Excel Project\Telco_Churn_Project\churn_project')

# ── 1. LOAD AND CLEAN ──────────────────────────────────────────
df = pd.read_excel('Telco_customer_churn.xlsx')
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
df.dropna(subset=['Total_Charges'], inplace=True)
df['Churn'] = df['Churn_Value']

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.1%}")

# ── 2. EDA CHARTS ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
axes[0].bar(contract_churn['Contract'], contract_churn['Churn'] * 100,
            color=['#e74c3c', '#3498db', '#2ecc71'])
axes[0].set_title('Churn Rate by Contract Type')
axes[0].set_ylabel('Churn Rate (%)')

axes[1].hist(df[df['Churn']==0]['Monthly_Charges'], bins=30,
             alpha=0.6, label='Retained', color='#3498db')
axes[1].hist(df[df['Churn']==1]['Monthly_Charges'], bins=30,
             alpha=0.6, label='Churned', color='#e74c3c')
axes[1].set_title('Monthly Charges Distribution')
axes[1].set_xlabel('Monthly Charges ($)')
axes[1].legend()

df['tenure_group'] = pd.cut(df['Tenure_Months'],
                             bins=[0,12,24,48,72],
                             labels=['0-12m','13-24m','25-48m','48m+'])
tenure_churn = df.groupby('tenure_group')['Churn'].mean().reset_index()
axes[2].bar(tenure_churn['tenure_group'].astype(str),
            tenure_churn['Churn'] * 100, color='#e67e22')
axes[2].set_title('Churn Rate by Tenure')
axes[2].set_ylabel('Churn Rate (%)')

plt.tight_layout()
plt.savefig('churn_eda.png', dpi=150)
plt.show()
print("EDA chart saved.")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────
drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City',
             'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
             'Churn_Label', 'Churn_Value', 'Churn_Score',
             'Churn_Reason', 'CLTV', 'tenure_group']
df_model = df.drop(columns=drop_cols, errors='ignore')

le = LabelEncoder()
cat_cols = df_model.select_dtypes(include='object').columns
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# ── 4. TRAIN MODEL ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\n── MODEL PERFORMANCE ──")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

# ── 5. FEATURE IMPORTANCE ────────────────────────────────────
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='importance', y='feature', palette='Blues_r')
plt.title('Top 10 Churn Predictors (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("Feature importance chart saved.")

# ── 6. EXPORT PREDICTIONS ────────────────────────────────────
X_test_orig = X_test.copy()
X_test_orig['actual_churn'] = y_test.values
X_test_orig['predicted_churn'] = y_pred
X_test_orig['churn_probability'] = y_prob
# Add readable contract type back
contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
X_test_orig['Contract_Type'] = X_test_orig['Contract'].map(contract_map)
X_test_orig.to_csv('churn_predictions.csv', index=False)

print(f"\nPredictions exported: {len(X_test_orig)} customers")
print(f"High risk (probability > 0.7): {(y_prob > 0.7).sum()}")
