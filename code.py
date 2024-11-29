!pip install pandas scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
dataset_path = '/content/drive/MyDrive/TaxiFare.csv'  # Replace with the actual path to your dataset file
data = pd.read_csv(dataset_path)

X = data[['longitude_of_pickup', 'latitude_of_pickup', 'longitude_of_dropoff', 'latitude_of_dropoff', 'no_of_passenger']]
y = data['amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(data.head())
print(data.describe())
print(data.isnull().sum())
model = LinearRegression()
model.fit(X_train, y_train)
plt.hist(data['amount'], bins=20, color='blue')
plt.xlabel('Fare Amount')
plt.ylabel('Count')
plt.title('Distribution of Fare Amount')
plt.show()
# Visualize the relationship between 'amount' and 'no_of_passenger' using a scatter plot

plt.figure(figsize=(8, 6))
plt.scatter(data['no_of_passenger'], data['amount'], alpha=0.5)
plt.xlabel('Number of Passengers')
plt.ylabel('Fare Amount')
plt.title('Scatter Plot of Fare Amount vs. Number of Passengers')
plt.show()


# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Example prediction
#sample_input = [[-35,54,-35,68,4]]        # Replaced with actual values
a=int(input("Enter the Longitude of pickup: "))
b=int(input("Enter the Latiude of pickup: "))
c=int(input("Enter the Longitude of dropoff: "))
d=int(input("Enter the Latitude of dropoff: "))
e=int(input("Enter the No. of Passengers: "))
sample_input = [[a,b,c,d,e]]


predicted_amount = model.predict(sample_input)
print(f"Predicted Amount: ${predicted_amount[0]:.2f}")
