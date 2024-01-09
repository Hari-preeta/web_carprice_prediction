
import streamlit as sts
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("D:\Music\projects'\data science\car prediction model\car_data.csv")  # read complete CSV file
m = dataset.head()  # head for the first 5 rows
# print(m)

null = dataset["selling_price"].isnull().sum()  # sum of null elements in a particular column

# value_count..count of elements in each category
fuel_cnt = dataset["fuel"].value_counts()
own_cnt = dataset["owner"].value_counts()
# print(own_cnt)



# iloc..separating specific row and column
# input and output categories for prediction

X = dataset.iloc[:, [1, 4, 6, 3]].values  # input: YOM, fuel, transmission, km driven
Y = dataset.iloc[:, 2].values  # output: selling price
# print(X)
# print(Y)

# giving a label value ...1,2,3..... for fuel and transmission
lb = LabelEncoder()
X[:, 1] = lb.fit_transform(X[:, 1])  # label for fuel
lb1 = LabelEncoder()
X[:, 2] = lb1.fit_transform(X[:, 2])  # label for transmission
# print(X)

# train_test split used to separate training set and test set at test size as 0.05
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
# print(X_train[:,:]

# regressor is for estimating continuous numerical values based on given input features
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
reg = regressor.fit(X_train, y_train)
# print(reg)

# accuracy of prediction...from the test set
accuracy = regressor.score(X_test, y_test)
# print(accuracy*100,'%')

# sample prediction
# details required for prediction: year of manufacturing, fuel type, transmission type, km driven


# Streamlit UI elements for user input
sts.markdown("# Car Price Prediction App")
year = sts.text_input("Enter the year of manufacturing:")
# Dropdown for fuel type
fuel_type_options = ['Petrol', 'Diesel', 'CNG']
fuel_type = sts.selectbox("Select the fuel type:", fuel_type_options)
transmission_type_options = ['Manual','Automatic']
transmission_type = sts.selectbox("Select the transmission type:", transmission_type_options)
km_driven = sts.text_input("Enter the km driven:")

# Button to trigger prediction
if sts.button("Predict"):
    # Convert input to the format expected by the model
    fuel_type = lb.transform([fuel_type])[0]
    transmission_type = lb1.transform([transmission_type])[0]
    input_data = [[year, fuel_type, transmission_type, km_driven]]

    # Make prediction
    predicted_price = regressor.predict(input_data)

    # Display the result
    sts.success(f"Predicted selling price of the car is {int(predicted_price[0])}")
