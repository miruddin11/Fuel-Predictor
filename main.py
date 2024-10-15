import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('maruti_fuel_efficiency.csv')

# Convert categorical data to numerical data
data['Fuel_Type'] = data['Fuel_Type'].apply(lambda x: 1 if x == 'Petrol' else 0)

# Define features and target variable
X = data[['Engine_Size', 'Weight', 'Horsepower', 'Fuel_Type']]
y = data['Fuel_Efficiency']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'fuel_efficiency_model.pkl')

print("Model training complete and saved as 'fuel_efficiency_model.pkl'")