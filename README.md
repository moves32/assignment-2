import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv('accident_data.csv')

# Define features and target
X = df[['speed', 'weather_code', 'road_type_code', 'time_of_day', 'driver_age']]
y = df['accident_severity']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'accident_severity_model.pkl')













# Load saved model
model = joblib.load('accident_severity_model.pkl')

# Hypothetical input
sample_input = pd.DataFrame([{
    'speed': 80,
    'weather_code': 2,   # e.g., 2 = rainy
    'road_type_code': 1, # e.g., 1 = highway
    'time_of_day': 22,   # 10 PM
    'driver_age': 30
}])

# Predict severity
predicted_severity = model.predict(sample_input)
print(f"Predicted Accident Severity: {predicted_severity[0]}")
