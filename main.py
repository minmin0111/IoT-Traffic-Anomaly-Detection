import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. 加载数据并自动筛选数值特征（修复空格分隔问题）
# ==========================================
data_path = 'bot_iot_sample.csv'

# 【核心修复】sep=r'\s+' 表示使用一个或多个空格/制表符作为分隔符进行解析
df = pd.read_csv(data_path, sep=r'\s+')
print(f"成功加载数据集，正确切分后的形状为: {df.shape}")

# 自动过滤掉非数值列（排除IP、协议等字符串），只保留数字特征
numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] < 2:
    raise ValueError("错误：未能成功提取到数值特征，请检查CSV文件的分隔符或内容！")

# 约定：最后一列是分类标签(Label)，前面的数字列是特征(Features)
X = numeric_df.iloc[:, :-1]
y = numeric_df.iloc[:, -1]

print(f"提取出的轻量化数值特征包括: {list(X.columns)}")
print(f"用于训练的特征维度: {X.shape[1]}，样本数: {X.shape[0]}")

# 如果标签变成了浮点数，将其转换为整数分类
y = y.astype(int)

# ==========================================
# 2. 数据预处理与特征降维 (体现物联网轻量化设计)
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 PCA 将特征降维到 2 维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"经过PCA轻量化降维后维度: {X_pca.shape[1]}")

# ==========================================
# 3. 划分数据集并训练轻量级分类器
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 限制树深度的轻量级随机森林
clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# ==========================================
# 4. 模型评估与报告输出
# ==========================================
y_pred = clf.predict(X_test)

print("\n--- 实验评估结果 ---")
unique_labels = [str(l) for l in np.unique(y_test)]
print(classification_report(y_test, y_pred, target_names=unique_labels))

# ==========================================
# 5. 自动绘制并保存混淆矩阵图
# ==========================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap=plt.cm.Blues)

plt.title('IoT Traffic Detection Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300)
print("\n[成功] 混淆矩阵图表已保存为 'confusion_matrix.png'！")