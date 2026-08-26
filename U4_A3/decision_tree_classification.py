from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# (a) Display first 5 rows and data types
print("FIRST 5 ROWS")
print(df.head())
print("\nDATA TYPES")
print(df.dtypes)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

# (b) Build Decision Tree using Gini Index
model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X_train, y_train)

print("\nROOT NODE:", iris.feature_names[model.tree_.feature[0]])
print("\nTREE STRUCTURE")
print(model.tree_)

plt.figure(figsize=(16, 9))
plot_tree(model, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree - Iris Dataset")
plt.tight_layout()
plt.savefig("decision_tree_output.png", dpi=180)
plt.show()

# (c) Evaluation
y_pred = model.predict(X_test)

print("\nACCURACY:", accuracy_score(y_test, y_pred))
print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nMISCLASSIFIED INSTANCES")
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print("Actual:", iris.target_names[y_test[i]],
              "| Predicted:", iris.target_names[y_pred[i]],
              "| Features:", X_test[i])
