import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Giả lập dữ liệu khách hàng (Mock Data)
# Feature: [Income, Debt, Credit_History_Score, Age]
data = {
    'Income': [5000, 3000, 10000, 2000, 8000, 1200, 9500, 4000],
    'Debt': [1000, 2000, 500, 1500, 100, 1000, 200, 3000],
    'Credit_History': [700, 500, 800, 450, 750, 400, 780, 600],
    'Default': [0, 1, 0, 1, 0, 1, 0, 0]  # 0: Good, 1: Bad (Default)
}

df = pd.DataFrame(data)

# 2. Chuẩn bị dữ liệu
X = df[['Income', 'Debt', 'Credit_History']]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Huấn luyện mô hình (Banking Standard: Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Dự báo và đánh giá
predictions = model.predict(X_test)
print("=== CREDIT RISK REPORT ===")
print(classification_report(y_test, predictions, target_names=['Good Customer', 'High Risk']))
