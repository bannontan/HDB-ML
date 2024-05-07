import streamlit as st
from zipfile import ZipFile
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="HDB Predictive modelling", page_icon=":tada:", layout="wide")

# Define the path to your data.zip file
zip_file_path = "./data/data.zip"

# Extract the contents of the zip file to a temporary directory
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall("./data/tmp")

# Read each CSV file from the extracted directory
price1999 = pd.read_csv("./data/tmp/data/resale-flat-prices-based-on-approval-date-1990-1999.csv")
price2012 = pd.read_csv("./data/tmp/data/resale-flat-prices-based-on-approval-date-2000-feb-2012.csv")
price2014 = pd.read_csv("./data/tmp/data/resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv")
price2016 = pd.read_csv("./data/tmp/data/resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv")
price2020 = pd.read_csv("./data/tmp/data/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv")
cpi = pd.read_csv("./data/tmp/data/CPI.csv")

# Remove the temporary directory
# os.rmdir("./data/tmp")
for filename in os.listdir("./data/tmp"):
    file_path = os.path.join("./data/tmp", filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Remove the temporary directory
shutil.rmtree("./data/tmp")



# price1999 = pd.read_csv("./data/resale-flat-prices-based-on-approval-date-1990-1999.csv")
# price2012 = pd.read_csv("./data/resale-flat-prices-based-on-approval-date-2000-feb-2012.csv")
# price2014 = pd.read_csv("./data/resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv")
# price2016 = pd.read_csv("./data/resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv")
# price2020 = pd.read_csv("./data/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv")
# cpi = pd.read_csv("./data/CPI.csv")

def getYears(text):
    if isinstance(text, str):
        yearmonth= [int(s) for s in text.split() if s.isdigit()]

        if len(yearmonth) > 1:
            years = yearmonth[0] + yearmonth[1] / 12
        else:
            years = yearmonth[0]

        return years
    else: # if text is an int/float
        return text

def process_range(x):
    start, end = map(int, x.split(" TO "))
    return (start + end) // 2

prices = pd.concat([price1999, price2012, price2014, price2016, price2020], ignore_index = True, sort = False)
prices['month'] = pd.to_datetime(prices['month'])
unique_years = prices.dropna()
unique_years = unique_years['month'].dt.year.unique()
cpi["month"] = pd.to_datetime(cpi["month"], format = "%Y %b")
prices = prices.merge(cpi, on = "month", how = "left")
prices["real_price"] = (prices["resale_price"] / prices["cpi"]) * 100
prices["remaining_lease"] = prices["remaining_lease"].apply(lambda x: getYears(x))
median_remaining_lease = prices["remaining_lease"].median()
prices["remaining_lease"].fillna(median_remaining_lease, inplace = True)
prices["flat_type"] = prices["flat_type"].str.replace("MULTI-GENERATION", "MULTI GENERATION")
replace_values = {'NEW GENERATION':'New Generation', 'SIMPLIFIED':'Simplified', 'STANDARD':'Standard', 'MODEL A-MAISONETTE':'Maisonette', 'MULTI GENERATION':'Multi Generation', 
                  'IMPROVED-MAISONETTE':'Executive Maisonette', 'Improved-Maisonette':'Executive Maisonette', 'Premium Maisonette':'Executive Maisonette', '2-ROOM':'2-room', 
                  'MODEL A':'Model A', 'MAISONETTE':'Maisonette', 'Model A-Maisonette':'Maisonette', 'IMPROVED':'Improved', 'TERRACE':'Terrace', 'PREMIUM APARTMENT':'Premium Apartment', 
                  'Premium Apartment Loft':'Premium Apartment', 'APARTMENT':'Apartment', 'Type S1':'Type S1S2', 'Type S2':'Type S1S2'}

prices["flat_model"] = prices["flat_model"].replace(replace_values)
prices['flat_type'] = prices['flat_type'].map({'1 ROOM':1, '2 ROOM':2, '3 ROOM':3, '4 ROOM':4, '5 ROOM':5, 'EXECUTIVE':6, 'MULTI GENERATION':7})
prices['year'] = prices['month'].dt.year
prices.drop(columns='month', inplace=True)
prices['storey'] = prices['storey_range'].apply(lambda x: process_range(x))
prices.drop(columns='storey_range', inplace=True)
prices.drop(columns=['block', 'street_name'], inplace=True)
prices = pd.get_dummies(prices, columns=['town', 'flat_model'])

st.subheader("ML")

town_list = ['town_ANG MO KIO', 'town_BEDOK',
       'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH',
       'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA',
       'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG',
       'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA',
       'town_LIM CHU KANG', 'town_MARINE PARADE', 'town_PASIR RIS',
       'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG',
       'town_SERANGOON', 'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS',
       'town_YISHUN']
town = st.selectbox("Select a town", town_list)
confirm_button = st.button("Confirm")

if confirm_button:

    df_town = prices
    df_town = df_town[df_town[town] != False]
    x_town = df_town.drop(["resale_price"], axis=1)
    y_town = df_town["resale_price"]
    x_town_year = x_town[["year"]]
    prices["max_resale_price"] = prices.groupby('year')['resale_price'].transform("max")

    df_town_year = prices
    df_town_year = df_town_year[df_town_year[town] != False]
    y_town_max = df_town_year["max_resale_price"]
    degree = 3 
    poly_features = PolynomialFeatures(degree=degree)
    x_town_year_poly = poly_features.fit_transform(x_town_year)
    x_town_year_train, x_town_year_test, y_town_max_train, y_town_max_test = train_test_split(x_town_year_poly, y_town_max, test_size=0.2, random_state=42)
    town_year_max_model = LinearRegression()
    town_year_max_model.fit(x_town_year_train, y_town_max_train)
    y_pred_town_year = town_year_max_model.predict(x_town_year_test)

    prices["min_resale_price"] = prices.groupby('year')['resale_price'].transform("min")
    df_town_year = prices
    df_town_year = df_town_year[df_town_year[town] != False]
    y_town_min = df_town_year["min_resale_price"]
    x_town_year_train2, x_town_year_test2, y_town_min_train, y_town_min_test = train_test_split(x_town_year_poly, y_town_min, test_size=0.2, random_state=42)
    town_year_min_model = LinearRegression()
    town_year_min_model.fit(x_town_year_train2, y_town_min_train)
    y_pred_town_year_min = town_year_min_model.predict(x_town_year_test2)

    future_years = []
    for _ in range(1990, 2026):
        future_years.append([_])
        x_future = np.array(future_years).reshape(-1, 1)
    x_future_poly = poly_features.fit_transform(x_future)

    y_future_pred_max = town_year_max_model.predict(x_future_poly)
    y_future_pred_min = town_year_min_model.predict(x_future_poly)

    st.title('Predicted Max and Min Resale Prices for Future Years')

    # Display the plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the predicted maximum resale prices
    ax.scatter(x_future, y_future_pred_max, color='red', label='Predicted Maximum')

    # Plot the predicted minimum resale prices
    ax.scatter(x_future, y_future_pred_min, color='blue', label='Predicted Minimum')

    # Add labels and title to the plot
    ax.set_xlabel('Year')
    ax.set_ylabel('Resale Price')
    ax.set_title('Predicted Max and Min Resale Prices for Future Years')

    # Add legend and grid to the plot
    ax.legend()
    ax.grid(True)

    # Display the plot in the Streamlit app
    st.pyplot(fig)
