import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'conf_mat.csv'
data = pd.read_csv(file_path)

pivot_table = data.pivot(index='Actual', columns='Predicted', values='nPredictions').fillna(0)


confusion_mat = pivot_table.to_numpy()


fig, ax = plt.subplots(figsize=(20, 15))  # Adjust size as needed
sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues', xticklabels=pivot_table.columns, yticklabels=pivot_table.index, ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()
