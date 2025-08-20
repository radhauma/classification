from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# Example data
X = np.array([[600],[700],[800],[900],[1000],[1200]])   # feature: size_sqft
y = np.array([180,200,220,240,260,300])                # target: price in $k

# Train
tree = DecisionTreeRegressor(
    criterion="squared_error",     # or "absolute_error"
    max_depth=2,                   # simple, avoids overfitting in the toy set
    min_samples_leaf=1,
    random_state=42
)
tree.fit(X, y)

# Predict
print("Pred(850 sqft) =", tree.predict([[850]])[0], "k")

# Visualize
plt.figure(figsize=(9,5))
plot_tree(tree, filled=True, feature_names=['size_sqft'], rounded=True)
plt.show()

# Feature importance
print("Feature importances:", tree.feature_importances_)
