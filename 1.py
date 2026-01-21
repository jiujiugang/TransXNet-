import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 手动输入混淆矩阵数值
confusion_matrix = np.array([[77, 4, 15, 3],  # 这里输入你的数据
                             [5, 6, 4, 1],
                             [14, 1, 15, 0],
                             [2, 3, 2, 10]])

# 归一化处理（按行归一化）
normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

# 手动调整最后一行归一化误差，如果需要
normalized_confusion_matrix[-1, 1] = 0.17  # 手动修改最后一行的数值

# 绘制调整后的混淆矩阵图，不显示cbar (参考幅度)
plt.figure(figsize=(8, 6))
sns.heatmap(normalized_confusion_matrix, annot=True, cmap='Blues', fmt='.2f', cbar=False,
            xticklabels=['Negative', 'Positive', 'Surprise', 'Others'],
            yticklabels=['Negative', 'Positive', 'Surprise', 'Others'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
