import streamlit as st
from serpapi import GoogleSearch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

st.set_page_config(
    page_title="GetRight",
    page_icon="📈", 
    layout="centered",
)

# Set your SerpAPI key (hardcoded)
SERPAPI_KEY = "your_api_key_here"

# Function to fetch historical stock data using SerpAPI
def fetch_historical_stock_data(stock_symbol):
    # Example API call to fetch historical data
    # Replace with actual data fetching logic
    params = {
        "engine": "google_finance",
        "q": f"{stock_symbol}:NASDAQ",
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Placeholder: Replace with actual data extraction
    dates = np.arange(1, 61)  # Dummy dates for 5 years (60 months)
    prices = np.linspace(100, 200, 60)  # Dummy prices
    
    return dates, prices

# List of stock symbols for user reference
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "FB", "BRK.A", "NVDA", "JPM", "UNH",
                  "V", "MA", "WMT", "DIS", "HD", "NFLX", "INTC", "CSCO", "PFE", "MRK",
                  "T", "XOM", "BA", "NKE", "WFC", "ADBE", "CMCSA", "CVX", "ORCL", "IBM", "GE"]

# Streamlit App with Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Prediction", "Final Prediction"])

# Display stock symbols in the sidebar for selection
selected_stocks = st.sidebar.multiselect("Choose stock symbols for comparison:", stock_symbols)

if page == "Home":
    st.title("Stock Comparison (E-Commerce)")
    st.write("""
    Welcome to the Stock Comparison Dashboard. 

    This app allows you to compare different stocks using various machine learning models to predict future stock prices. 
    You can select two stocks and see predictions using multiple algorithms, and in the end, get the best-performing model.""")

    st.write("Priya Dharshini G - 2347247")

    # Displaying the 5 algorithms with their definitions and formula images
    st.header("Overview of Algorithms Used")
    
    st.subheader("1. Linear Regression")
    st.write("""
    **Definition:** Linear Regression is a supervised learning algorithm used to predict the value of a dependent variable (y) based on one or more independent variables (x). The model assumes a linear relationship between input variables and the output.
    """)
    lr_image = Image.open("Linear-Regression-Equation.jpg")  #
    st.image(lr_image, caption="Linear Regression Formula", use_column_width=True)

    st.subheader("2. Decision Tree Regressor")
    st.write("""
    **Definition:** A Decision Tree Regressor is a non-parametric model that predicts the target variable by learning simple decision rules from the data features. It splits the dataset into branches based on feature thresholds until it reaches a decision.
    """)
    dt_image = Image.open("DesicionTree.png")
    st.image(dt_image, caption="Decision Tree Formula", use_column_width=True)

    st.subheader("3. Random Forest Regressor")
    st.write("""
    **Definition:** Random Forest is an ensemble method that builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting. It uses bootstrapping and feature randomness when constructing each tree.
    """)
    rf_image = Image.open("RandomForest.png")
    st.image(rf_image, caption="Random Forest Formula", use_column_width=True)

    st.subheader("4. K-Nearest Neighbors (KNN) Regressor")
    st.write("""
    **Definition:** KNN is a non-parametric algorithm that predicts the target by averaging the values of the k-nearest neighbors in the feature space. It’s based on the idea that similar data points are close to each other.
    """)
    knn_image = Image.open("KNN.png")
    st.image(knn_image, caption="K-Nearest Neighbors Formula", use_column_width=True)

    st.subheader("5. Support Vector Regressor (SVR)")
    st.write("""
    **Definition:** Support Vector Regression is an algorithm that finds a hyperplane that best fits the data while maintaining a margin of tolerance. It uses kernel tricks to handle non-linear data.
    """)
    svr_image = Image.open("SVR.png")
    st.image(svr_image, caption="Support Vector Regressor Formula", use_column_width=True)

elif page == "Prediction":
    st.title("Prediction Page")

    if len(selected_stocks) == 2:
        stock1, stock2 = selected_stocks

        # Fetch stock data
        dates1, prices1 = fetch_historical_stock_data(stock1)
        dates2, prices2 = fetch_historical_stock_data(stock2)
        
        # Machine Learning Models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(n_estimators=10),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=3),
            "Support Vector Machine": SVR(kernel='linear')
        }
        
        for model_name, model in models.items():
            st.subheader(f"{model_name} Analysis")
            
            # Train and predict for stock 1
            model.fit(dates1.reshape(-1, 1), prices1)
            predictions1 = model.predict(dates1.reshape(-1, 1))
            mse1 = mean_squared_error(prices1, predictions1)
            r2_1 = r2_score(prices1, predictions1)
            
            # Train and predict for stock 2
            model.fit(dates2.reshape(-1, 1), prices2)
            predictions2 = model.predict(dates2.reshape(-1, 1))
            mse2 = mean_squared_error(prices2, predictions2)
            r2_2 = r2_score(prices2, predictions2)
            
            # Display results for stock 1
            st.write(f"**{stock1} Predictions with {model_name}:**")
            st.write(f"MSE: {mse1:.2f}, R²: {r2_1:.2f}")
            
            # Display results for stock 2
            st.write(f"**{stock2} Predictions with {model_name}:**")
            st.write(f"MSE: {mse2:.2f}, R²: {r2_2:.2f}")
            
            # Plotting the predictions for 5 years
            plt.figure(figsize=(10, 6))
            plt.plot(dates1, prices1, label=f'Actual {stock1}', color='blue')
            plt.plot(dates1, predictions1, label=f'Predicted {stock1}', color='red', linestyle='--')
            plt.plot(dates2, prices2, label=f'Actual {stock2}', color='green')
            plt.plot(dates2, predictions2, label=f'Predicted {stock2}', color='orange', linestyle='--')
            plt.xlabel('Month')
            plt.ylabel('Price')
            plt.legend()
            plt.title(f'{model_name} Predictions for 5 Years')
            st.pyplot(plt)
    else:
        st.write("Please select exactly two stock symbols to start.")

elif page == "Final Prediction":
    st.title("Final Prediction Page")
    
    if len(selected_stocks) == 2:
        stock1, stock2 = selected_stocks

        # Fetch stock data
        dates1, prices1 = fetch_historical_stock_data(stock1)
        dates2, prices2 = fetch_historical_stock_data(stock2)
        
        # Machine Learning Models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(n_estimators=10),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=3),
            "Support Vector Machine": SVR(kernel='linear')
        }
        
        best_model = None
        best_mse = float('inf')

        # Train models and calculate the MSE for both stocks
        for model_name, model in models.items():
            # Train for stock 1
            model.fit(dates1.reshape(-1, 1), prices1)
            mse1 = mean_squared_error(prices1, model.predict(dates1.reshape(-1, 1)))
            
            # Train for stock 2
            model.fit(dates2.reshape(-1, 1), prices2)
            mse2 = mean_squared_error(prices2, model.predict(dates2.reshape(-1, 1)))
            
            # Take the average MSE for both stocks
            avg_mse = (mse1 + mse2) / 2
            
            # Check for the best model
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_model = model_name

        st.subheader("Final Analysis")
        st.write(f"The best performing model is **{best_model}**.")
        st.write(f"The stock predicted to give you more profit is **{stock1 if np.mean(prices1) > np.mean(prices2) else stock2}**.")
    else:
        st.write("Please go to the Prediction page and select exactly two stock symbols.")
