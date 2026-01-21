import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 示例的混淆矩阵数据 (未归一化)
confusion_matrix = np.array([[77, 4, 15, 3],
                             [5, 6, 4, 1],
                             [14, 1, 15, 0],
                             [2, 3, 2, 10]])

# 归一化处理（按行归一化）
normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

# 手动调整最后一行归一化误差
normalized_confusion_matrix_adjusted = normalized_confusion_matrix.copy()

# 计算最后一行的总和
row_sum = normalized_confusion_matrix_adjusted[-1, :].sum()

# 将最后一行的所有元素按比例调整，使总和为 1
normalized_confusion_matrix_adjusted[-1, :] /= row_sum

# 绘制调整后的归一化混淆矩阵图，不显示cbar (参考幅度)
plt.figure(figsize=(8, 6))
sns.heatmap(normalized_confusion_matrix_adjusted, annot=True, cmap='Blues', fmt='.2f', cbar=False, xticklabels=['Negative', 'Positive', 'Surprise', 'Others'], yticklabels=['Negative', 'Positive', 'Surprise', 'Others'])
plt.title(' Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
