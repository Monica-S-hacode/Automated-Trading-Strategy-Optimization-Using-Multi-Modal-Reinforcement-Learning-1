import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load cleaned data (with Target already added in Step 3)
data = pd.read_csv("reliance_1y_data_with_next_close.csv", index_col="Date", parse_dates=True)

# Make sure Target column is present (safety check)
if "Target" not in data.columns:
    data["Target"] = (data["Next_Close"] > data["Close"]).astype(int)

# Features and labels
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data["Target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Evaluate KNN for multiple K values
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K = {k}, Accuracy = {acc}")
