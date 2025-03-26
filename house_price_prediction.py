# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# # Load the dataset
# data = pd.read_csv(r"C:\python\train.csv")
# # Select features and target variable
# X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath','Neighborhood']]
# y = data['SalePrice']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the Linear Regression model
# model = LinearRegression()
# # Train the model
# model.fit(X_train, y_train)

# # Function to predict house price based on user input
# def predict_house_price():
#     print("Enter the following details to predict the house price:")
#     square_footage = float(input("Square Footage: "))
#     bedrooms = int(input("Number of Bedrooms: "))
#     bathrooms = int(input("Number of Bathrooms: "))
#     location=str(input("Enter the Location :"))

#     # Create a DataFrame with the user inputs
#     new_data = pd.DataFrame({
#         'GrLivArea': [square_footage],
#         'BedroomAbvGr': [bedrooms],
#         'FullBath': [bathrooms],
#         'Neighborhood': [location]
#     })

#     # Predict the price
#     predicted_price = model.predict(new_data)
#     print(f'\nPredicted House Price: ${predicted_price[0]:,.2f}')

# # Run the prediction function
# predict_house_price()

#new
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv(r"C:\python\train.csv")

# Encode the 'Neighborhood' column
le = LabelEncoder()
data['Neighborhood_Encoded'] = le.fit_transform(data['Neighborhood'])

# Select features and target variable
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'Neighborhood_Encoded']]
y = data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Function to predict house price based on user input
def predict_house_price():
    print("Enter the following details to predict the house price:")
    square_footage = float(input("Square Footage: "))
    bedrooms = int(input("Number of Bedrooms: "))
    bathrooms = int(input("Number of Bathrooms: "))
    location = input("Enter the Location (Neighborhood): ")
    #neighborhoods: ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker']
    
    try:
        # Encode the user's location input
        location_encoded = le.transform([location])[0]
    except ValueError:
        print(f"Error: '{location}' is not a valid neighborhood in our dataset.")
        print("Please enter one of these neighborhoods:", list(le.classes_))
        return

    # Create a DataFrame with the user inputs
    new_data = pd.DataFrame({
        'GrLivArea': [square_footage],
        'BedroomAbvGr': [bedrooms],
        'FullBath': [bathrooms],
        'Neighborhood_Encoded': [location_encoded]
    })

    # Predict the price
    predicted_price = model.predict(new_data)
    print(f'\nPredicted House Price: ${predicted_price[0]:,.2f}')

# Run the prediction function
predict_house_price()