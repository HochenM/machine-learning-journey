import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

X, y = make_blobs(
    n_samples=100,
    n_features=2,
    centers=3,
    cluster_std=5,
    random_state=42
)

y_one_hot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X, y)

y_pred = model.predict_proba(X)

for i in range(3):
    fpr, tpr, thresholds = roc_curve(y_one_hot[:,i], y_pred[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve {i} (AUC = %0.3f)' % roc_auc)


plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--')
plt.grid(True)
plt.legend()
plt.show()
