# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

# 加载数据
data = pd.read_csv("./fraudulent.csv")

# 检查数据的基本信息和缺失值情况
print("数据集信息：")
print(data.info())
print("\n数据集描述：")
print(data.describe())

# 处理缺失值 - 使用众数填充缺失值
imputer = SimpleImputer(strategy="most_frequent")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 分离特征和标签
X = data_imputed.drop("y", axis=1)  # 特征
y = data_imputed["y"]               # 标签

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 初始化和训练决策树模型
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_dt = dt.predict(X_test)

# 计算并输出F1分数
f1_dt = f1_score(y_test, y_pred_dt)
print("决策树模型的F1分数:", f1_dt)