import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load training dataset
df = pd.read_csv("train(1).csv")

# Assuming target column is 'SalePrice'
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Convert categorical columns to numeric
X = pd.get_dummies(X)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
print("Linear Regression Accuracy:", r2_score(y_test, predictions))
