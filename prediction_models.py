import pandas as pd
import numpy as np
import streamlit as st
from decimal import Decimal
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import psycopg2
from dotenv import load_dotenv
from sklearn.inspection import PartialDependenceDisplay
from xgboost import plot_importance
import shap
from sklearn.tree import plot_tree

from data_analysis import fetch_values_from_metadata, convert_to_dataframe
# Load environment variables from .env file
load_dotenv()

# Retrieve connection parameters from environment variables
conn_params = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}
feature_mapping = {
'Price-to-Earnings Ratio': 'pe_ratio',
'Revenue Growth': 'revenue_growth',
'Earnings Growth': 'earnings_growth'
}

duration = {
'1 Day': 1,
'7 Days': 7,
'30 Days': 30
}

# Function to fetch models from PostgreSQL
def get_models():
    global conn_params
    # Establish the connection to the database
    connection = psycopg2.connect(**conn_params)
    # Create a cursor object
    cursor = connection.cursor()
    # Query to fetch model names and the default model
    cursor.execute("""
        SELECT model_name, is_default
        FROM model.models;
    """)
    
    # Fetch all rows
    models = cursor.fetchall()
    
    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Separate the model names and identify the default model
    model_names = [row[0] for row in models]
    default_model = next((row[0] for row in models if row[1]), None)

    return model_names, default_model

# Function to fetch features from PostgreSQL
def get_features():
    global conn_params
    # Establish the connection to the database
    connection = psycopg2.connect(**conn_params)
    # Create a cursor object
    cursor = connection.cursor()
    # Query to fetch metadata_long_name and the default features
    cursor.execute("""
        SELECT metadata_long_name, is_default
        FROM metadata.metadata_view
        WHERE is_visible = true
        AND frame_title = 'Analysis Data Frame';
    """)
    
    # Fetch all rows
    features = cursor.fetchall()
    
    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Separate the feature names and identify the default features
    feature_names = [row[0] for row in features]
    default_features = [row[0] for row in features if row[1]]

    return feature_names, default_features


def reference_table(long_name_list):
    try:
        global conn_params
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Define the SQL query template to fetch schema, table, and column names
        query = """
            SELECT schema_name, table_name, field_label 
            FROM metadata.metadata_view
            WHERE metadata_long_name = %s  and is_visible = true and frame_title= 'Analysis Data Frame';
        """
        
        # Dictionary to store the results for each metadata_long_name
        metadata_results = {}
        
        # Iterate over each item in the metadata_long_name list and fetch corresponding metadata
        for metadata_long_name in long_name_list:
            # Execute the query with the current metadata_long_name
            cursor.execute(query, (metadata_long_name,))
            
            # Fetch the result for the current metadata_long_name
            result = cursor.fetchone()
            
            # If a result is found, add it to the dictionary
            if result:
                schema_name, table_name, field_label = result
                metadata_results[metadata_long_name] = {
                    "schema_name": schema_name,
                    "table_name": table_name,
                    "field_label": field_label
                }
            else:
                # If no metadata is found, store None or an empty dict for that metadata_long_name
                metadata_results[metadata_long_name] = None
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        # Return the dictionary containing all the results
        return metadata_results

    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return None

def visalize_data(long_name_list):
    print(long_name_list)
    # Call the function and store the returned data
    metadata = reference_table(long_name_list)
   
    # Call the function and store the returned data
    values = fetch_values_from_metadata(metadata)

    # Convert the dictionary to a pandas dataframe
    df = convert_to_dataframe(values)
    # Drop duplicate rows based on specific columns
    df_unique = df.drop_duplicates(subset=long_name_list)
    return df_unique

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(by=['ticker', 'date'], ascending=[True, False])
    return df


def select_and_normalize_columns(df, selected_features, selected_predictor, selected_duration_key):
    mapped_duration = duration[selected_duration_key]

    # Add the selected predictor column to the features before normalization
    all_features = selected_features + [selected_predictor]
    
    # Select the relevant columns directly from the dataframe
    df_selected = df[all_features]

    # Handle percentage strings and convert to float where necessary
    df_selected = df_selected.apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) and x.endswith('%') else x)

    # Forward fill and backward fill to handle missing data
    df_selected.ffill(inplace=True)
    df_selected.bfill(inplace=True)
    
    # Convert all data to numeric
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_selected), columns=all_features)

    # Combine with 'date' and 'ticker' columns
    df_final_normalized = pd.concat([df[['ticker', 'date']].reset_index(drop=True), df_normalized.reset_index(drop=True)], axis=1)

    # Create lagged columns based on the selected duration key
    df_final_normalized[f'{selected_predictor}_lag'] = df_final_normalized[selected_predictor].shift(mapped_duration)
    # Final DataFrame that retains the original number of rows
    df_final = df_final_normalized.copy()
    # Drop the lagged column
    df_final_normalized = df_final_normalized.drop(columns=[f'{selected_predictor}_lag'])

    return df_final_normalized, df_final, scaler



def split_data(df_final):
    train_list = []
    test_list = []
    for ticker, group in df_final.groupby('ticker'):
        # Only split if the group has more than 1 sample
        if len(group) > 1:
            group_train, group_test = train_test_split(group, test_size=0.2, random_state=42)
        else:
            # If the group has only 1 sample, add it entirely to the training set (or test set, depending on your preference)
            group_train = group
            group_test = pd.DataFrame()  # Empty test set

        train_list.append(group_train)
        test_list.append(group_test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    
    return train_df, test_df


def prepare_ml_data(train_df, test_df, target_col):
    X_train = train_df.drop(columns=['date', 'ticker', target_col, f'{target_col}_lag'])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=['date', 'ticker', target_col, f'{target_col}_lag'])
    y_test = test_df[target_col]
   
    return X_train, y_train,X_test, y_test


def build_and_train_lr_model(X_train, y_train, X_test, y_test):
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get intercept and coefficients
    intercept = model.intercept_
    coefficients = model.coef_

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_pred, mse, r2, intercept, coefficients



def build_and_train_rf_model(X_train, y_train, X_test, y_test, n_estimators=100, random_state=None):
    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_pred, mse, r2

def build_and_train_xgb_model(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
    # Initialize and train the XGBoost model
    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_pred, mse, r2


def build_and_train_dt_model(X_train, y_train, X_test, y_test, max_depth=None, random_state=None):
    # Initialize and train the Decision Tree model
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_pred, mse, r2


def remove_outliers(X_test, y_test, y_pred):
    # Convert to NumPy arrays if not already
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for y_test
    Q1 = np.percentile(y_test, 25)
    Q3 = np.percentile(y_test, 75)
    IQR = Q3 - Q1  # Interquartile range
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find the indices of non-outliers (i.e., points within the bounds)
    non_outlier_indices = (y_test >= lower_bound) & (y_test <= upper_bound)
    
    # Filter X_test, y_test, and y_pred to remove outliers
    X_test_clean = X_test[non_outlier_indices]
    y_test_clean = y_test[non_outlier_indices]
    y_pred_clean = y_pred[non_outlier_indices]
    
    return X_test_clean, y_test_clean, y_pred_clean

def model_plot(X_test, y_test, y_pred, scaler=None,remove_outliers_flag=False):
    # Ensure the data is in NumPy array form (for pandas DataFrame/Series)
    if hasattr(X_test, 'to_numpy'):
        X_test = X_test.to_numpy()

    # Ensure y_test and y_pred are NumPy arrays for manipulation
    y_test = y_test.to_numpy() if isinstance(y_test, (pd.Series, pd.DataFrame)) else np.array(y_test)
    y_pred = y_pred.to_numpy() if isinstance(y_pred, (pd.Series, pd.DataFrame)) else np.array(y_pred)

    # Combine X_test and y_test into one array to match the scaler's expected input shape
    combined_test = np.hstack([X_test, y_test.reshape(-1, 1)])
    
    # Apply inverse transformation if scaler is provided
    if scaler is not None:
        combined_test = scaler.inverse_transform(combined_test)

    # Split the combined array back into X_test and y_test
    X_test = combined_test[:, :-1]  # All columns except the last one (features)
    y_test = combined_test[:, -1]   # Last column (target)

    # Combine X_test and y_pred into one array for inverse transforming predicted values
    combined_pred = np.hstack([X_test, y_pred.reshape(-1, 1)])
    
    # Apply inverse transformation to predicted values as well
    if scaler is not None:
        combined_pred = scaler.inverse_transform(combined_pred)
    
    # Extract the predicted target values from the last column
    y_pred = combined_pred[:, -1]
    
    if remove_outliers_flag==True:
        # Remove outliers from X_test, y_test, and y_pred
        X_test, y_test, y_pred = remove_outliers(X_test, y_test, y_pred)

    # Create a color map for features
    num_features = X_test.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_features))  # Generate distinct colors

    # Create a scatter plot for each feature
    plt.figure(figsize=(12, 8))
    
    for i in range(num_features):
        feature_to_plot = X_test[:, i]
        
        # Plot actual values for each feature
        plt.scatter(feature_to_plot, y_test, color=colors[i], label=f'Actual Values (Feature {i + 1})', alpha=0.5)

    # Plot the predicted values as a red line
    plt.plot(X_test[:, 0], y_pred, color='red', linestyle='--', linewidth=2, label='Predicted Line')

    # Set labels and title
    plt.xlabel('Feature Values')
    plt.ylabel('Target Value')
    plt.title('Regression Lines: Actual vs Predicted for Each Feature')

    # Show legend
    plt.legend()
    
    # Show the plot
    plt.show()





def visualize_random_forest_results(model, X_test, y_test, y_pred, feature_names):

    # 1. Actual vs. Predicted Plot
    st.subheader('Actual vs Predicted Plot')
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Diagonal line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid()
    st.pyplot(plt.gcf())  # Display the plot in Streamlit
    plt.clf()  # Clear the figure for the next plot

    # 2. Feature Importance Plot
    st.subheader("Feature Importance Plot")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort in descending order

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # 3. Partial Dependence Plot (for first two features for simplicity)
    st.subheader("Partial Dependence Plot")
    features_to_plot = feature_names[:2]  # Adjust as necessary
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_test,
        features=features_to_plot,
        grid_resolution=50
    )
    st.pyplot(plt.gcf())
    plt.clf()

    # 4. Residual Plot
    st.subheader('Residual Plot')
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid()
    st.pyplot(plt.gcf())
    plt.clf()

    # 5. 3D Surface Plot (for first two features)
    if X_test.shape[1] >= 2:  # Ensure there are at least two features
        from mpl_toolkits.mplot3d import Axes3D

        st.subheader('3D Surface Plot: Actual vs Predicted')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label='Actual', color='b', alpha=0.5)
        ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, label='Predicted', color='r', alpha=0.5)

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel('Target Value')
        plt.legend()
        plt.title('3D Surface Plot: Actual vs Predicted')
        st.pyplot(fig)
        plt.clf()

    # #. SHAP Summary Plot
    # st.subheader('SHAP Summary Plot')
    # explainer = shap.Explainer(model)  # Use the general Explainer for Random Forest
    # shap_values = explainer(X_test)
    
    # # Create the SHAP summary plot
    # shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    # plt.tight_layout()
    # st.pyplot(plt.gcf())
    # plt.clf()




def visualize_xgboost_regression(model, X_test, y_test, y_pred, feature_names):
    """
    Generate various plots to visualize the results of an XGBoost regression model using Streamlit.
    
    Parameters:
        model: Trained XGBoost model.
        X_test: Test features (2D array).
        y_test: Actual target values (1D array).
        y_pred: Predicted target values (1D array).
        feature_names: Names of the features (list).
    """
    
    # 1. Actual vs. Predicted Plot
    st.subheader('Actual vs Predicted Plot')
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Diagonal line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid()
    st.pyplot(plt.gcf())
    plt.clf()

    # 2. Feature Importance Plot
    st.subheader("Feature Importance Plot")
    plot_importance(model, importance_type='weight', title='Feature Importance', xlabel='F score')
    st.pyplot(plt.gcf())
    plt.clf()

    # 3. SHAP Summary Plot
    st.subheader('SHAP Summary Plot')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    # 4. Residual Plot
    st.subheader('Residual Plot')
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid()
    st.pyplot(plt.gcf())
    plt.clf()



def visualize_decision_tree_regression(model, X_test, y_test, y_pred, feature_names):
    """
    Generate various plots to visualize the results of a Decision Tree regression model using Streamlit.
    
    Parameters:
        model: Trained Decision Tree model.
        X_test: Test features (2D array).
        y_test: Actual target values (1D array).
        y_pred: Predicted target values (1D array).
        feature_names: Names of the features (list).
    """
    
    # 1. Actual vs. Predicted Plot
    st.subheader('Actual vs Predicted Plot')
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Diagonal line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid()
    st.pyplot(plt.gcf())
    plt.clf()

    # 2. Feature Importance Plot
    st.subheader("Feature Importance Plot")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort in descending order
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # 3. Residual Plot
    st.subheader('Residual Plot')
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid()
    st.pyplot(plt.gcf())
    plt.clf()

    # 4. Decision Tree Visualization
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True)
    st.subheader('Decision Tree Visualization')
    st.pyplot(plt.gcf())
    plt.clf()

def inverse_transform_and_plot_predictions(model, X_test, y_test, scaler, target_col, n_rows_per_ticker=1, n_display=10):
    # Get the most recent dates for each ticker
    newest_dates = X_test.groupby('ticker')['date'].max().reset_index()
    X_test_newest = pd.merge(X_test, newest_dates, on=['ticker', 'date'])
    y_test_newest = pd.merge(y_test, newest_dates, on=['ticker', 'date'])

    # All unique tickers
    unique_tickers = X_test_newest['ticker'].unique()
    
    all_actual_prices = []
    all_predicted_prices = []
    all_tickers = []
    
    # Loop over all tickers
    for ticker in unique_tickers:
        # Limit to n rows per ticker
        X_test_ticker = X_test_newest[X_test_newest['ticker'] == ticker].drop(columns=['ticker', 'date', target_col, f'{target_col}_lag']).head(n_rows_per_ticker)
        y_test_ticker = y_test_newest[y_test_newest['ticker'] == ticker].drop(columns=['ticker', 'date']).head(n_rows_per_ticker)
        
        # Predict
        y_pred_test = model.predict(X_test_ticker)

        # Prepare arrays for inverse transform
        dummy_shape = np.zeros((y_pred_test.shape[0], scaler.n_features_in_))
        dummy_shape[:, -1] = y_pred_test.flatten()
        y_pred_test_rescaled = scaler.inverse_transform(dummy_shape)[:, -1]

        dummy_shape[:, -1] = y_test_ticker.values.flatten()
        y_test_rescaled = scaler.inverse_transform(dummy_shape)[:, -1]

        all_actual_prices.append(y_test_rescaled.flatten())
        all_predicted_prices.append(y_pred_test_rescaled.flatten())
        all_tickers.extend([ticker] * len(y_test_ticker))
    
    # Concatenate all actual and predicted prices
    all_actual_prices = np.concatenate(all_actual_prices)
    all_predicted_prices = np.concatenate(all_predicted_prices)

    # Streamlit plot
    st.title(f'Actual vs Predicted {target_col} for Tickers')

    # Pagination setup
    total_items = len(all_tickers)
    total_pages = math.ceil(total_items / n_display)

    # Calculate the start and end indices based on the selected page number
    current_page = st.session_state.get('page', 1)  # Use session state for page persistence
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    # Page selection input
    page_input = st.number_input(f"Select Page (1 to {total_pages}):", min_value=1, max_value=total_pages, value=current_page)
    
    # Update the current page based on the input
    if page_input != current_page:
        st.session_state['page'] = page_input
        current_page = page_input
    
    start_index = (current_page - 1) * n_display
    end_index = min(start_index + n_display, total_items)

    # Get the current tickers and prices to display
    current_tickers = all_tickers[start_index:end_index]
    current_actual_prices = all_actual_prices[start_index:end_index]
    current_predicted_prices = all_predicted_prices[start_index:end_index]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(18, 6))
    width = 0.35  # Bar width

    # Set positions for bars
    indices = np.arange(len(current_tickers))
    
    bars_actual = ax.bar(indices - width/2, current_actual_prices, width, label=f'Actual {target_col}', color='green')
    bars_predicted = ax.bar(indices + width/2, current_predicted_prices, width, label=f'Predicted {target_col}', color='red')

    # Set titles and labels
    ax.set_title(f'Actual vs Predicted {target_col} for Tickers (Page {current_page})')
    ax.set_ylabel(f'{target_col}')
    ax.set_xlabel('Ticker')
    ax.set_xticks(indices)
    ax.set_xticklabels(current_tickers, rotation=45, ha='right')
    ax.legend()

    # Show the plot
    st.pyplot(fig)

    # Show the current page and range information
    st.write(f'Page {current_page} of {total_pages}')


