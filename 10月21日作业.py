import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载数据并预处理
df = pd.read_csv('bike.csv')
df = df.drop('id', axis=1)
shanghai_data = df[df['city'] == 1].drop('city', axis=1)
shanghai_data['hour'] = shanghai_data['hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)
y_vector = shanghai_data.pop('y').to_numpy().reshape(-1, 1)
X = shanghai_data.to_numpy()

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_vector, test_size=0.2, random_state=42)

# 数据归一化
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)
y_train_normalized = scaler.fit_transform(y_train)
y_test_normalized = scaler.transform(y_test)

# 模型构建与训练
model = LinearRegression()
model.fit(X_train_normalized, y_train_normalized)

# 模型测试
y_pred = model.predict(X_test_normalized)
mse = mean_squared_error(y_test_normalized, y_pred)
print("Mean Squared Error:", mse)

# 计算并输出RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)