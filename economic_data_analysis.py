from dotenv import load_dotenv
import os
import psycopg2
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import os
import psycopg2
from dotenv import load_dotenv

import numpy as np
import streamlit as st
from datetime import datetime
# Load environment variables from .env file
load_dotenv()


# Get the active environment from the .env file
env = os.getenv('ENV', 'dev')  # Default to 'dev' if not set

# Dynamically set connection parameters based on the active environment
if env == 'dev':
    conn_params = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    }
    active_user_id = int(os.getenv('user', 1))  # Default to user 1 for dev
elif env == 'test':
    conn_params = {
        "dbname": os.getenv("DB_NAME_TEST"),
        "user": os.getenv("DB_USER_TEST"),
        "password": os.getenv("DB_PASSWORD_TEST"),
        "host": os.getenv("DB_HOST_TEST"),
        "port": os.getenv("DB_PORT_TEST")
    }
    active_user_id = int(os.getenv('user_test', 2))  # Default to user 2 for test
elif env == 'prod':
    conn_params = {
        "dbname": os.getenv("DB_NAME_PROD"),
        "user": os.getenv("DB_USER_PROD"),
        "password": os.getenv("DB_PASSWORD_PROD"),
        "host": os.getenv("DB_HOST_PROD"),
        "port": os.getenv("DB_PORT_PROD")
    }
    active_user_id = int(os.getenv('user_prod', 0))  # Default to user 0 for prod
else:
    raise ValueError("Invalid environment specified in .env file")


duration = {
'1 Day': 1,
'7 Days': 7,
'30 Days': 30
}

feature_name_mapping = {
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "CSUSHPISA": "S&P/Case-Shiller U.S. National Home Price Index",
    "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate",
    "DRALACBS": "Delinquency Rate on Loans",
    "FEDFUNDS": "Effective Federal Funds Rate",
    "GDI": "Gross Domestic Income",
    "GDP": "Gross Domestic Product",
    "GS10": "10-Year Treasury Constant Maturity Rate",
    "HOUST": "Housing Starts",
    "MHHNGSP": "Natural Gas Prices",
    "MORTGAGE30US": "30-Year Mortgage Rate",
    "PI": "Personal Income",
    "RRVRUSQ156N": "Real Retail and Food Services Sales",
    "RSAFS": "Retail Sales",
    "T10Y2Y": "10-Year Minus 2-Year Treasury Yield Spread",
    "TOTALSA": "Total Sales",
    "TTLCONS": "Total Construction Spending",
    "UMCSENT": "Consumer Sentiment Index",
    "UNRATE": "Unemployment Rate"
}

# Helper function to map feature names to full names
def map_feature_names(feature_names):
    return [feature_name_mapping.get(name, name) for name in feature_names]


# Function to fetch features based on data_frame_id = 9
def economic_feature():
    global conn_params
    # Establish the connection to the database
    connection = psycopg2.connect(**conn_params)
    # Create a cursor object
    cursor = connection.cursor()
    
    # Query to fetch metadata_long_name and the default features for data_frame_id = 9
    cursor.execute("""
        SELECT indicator_type
        FROM analytix.economic_indicators
        group by indicator_type
    """)
    
    # Fetch all rows
    feature_names = [row[0] for row in cursor.fetchall()]
    mapped_feature_names = map_feature_names(feature_names)
    
    # Close the cursor and connection
    cursor.close()
    connection.close()

    return mapped_feature_names


# Function to fetch features from PostgreSQL
def get_economic_features():
    global conn_params
    # Establish the connection to the database
    connection = psycopg2.connect(**conn_params)
    # Create a cursor object
    cursor = connection.cursor()
    # Query to fetch indicator_type and group by it
    cursor.execute("""
    SELECT indicator_type
    FROM analytix.economic_indicators
    GROUP BY indicator_type
    """)
    
    # Fetch all rows and extract the first element from each tuple
    feature_names = [row[0] for row in cursor.fetchall()]
    mapped_feature_names = map_feature_names(feature_names)
    
    # Close the cursor and connection
    cursor.close()
    connection.close()

    return mapped_feature_names


# Function to fetch tickers from PostgreSQL
def get_tickers():
    global conn_params
    # Establish the connection to the database
    connection = psycopg2.connect(**conn_params)
    # Create a cursor object
    cursor = connection.cursor()
    # Query to fetch unique tickers
    cursor.execute("""
    SELECT ticker
    FROM reference.securities
    where security_name like '%SPDR%';
    """)
    
    # Fetch all rows
    tickers = cursor.fetchall()
    
    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Extract the tickers from the result
    ticker_list = [row[0] for row in tickers]

    return ticker_list

# New function to fetch metadata directly based on query results
def fetch_metadata():
    try:
        global conn_params
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Define the SQL query to fetch schema, table, and field label
        query = """
            SELECT schema_name, table_name, field_label
            FROM metadata.metadata_view
            where frame_title like '%Econ%';
        """
        
        # Execute the query to fetch all relevant metadata
        cursor.execute(query)
        
        # Fetch all the results
        results = cursor.fetchall()
        
        # Dictionary to store the metadata with field_label as key
        metadata_results = {}
        
        # Iterate over each result row and store it in the dictionary
        for schema_name, table_name, field_label in results:
            metadata_results[field_label] = {
                "schema_name": schema_name,
                "table_name": table_name
            }
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        # Return the dictionary containing all the results
        return metadata_results

    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return None

# Reverse the mapping for full name to short name
reversed_feature_mapping = {v: k for k, v in feature_name_mapping.items()}

# Helper function to map full names to short names
def map_to_short_names(full_names):
    return [reversed_feature_mapping.get(name, name) for name in full_names]

def fetch_data_by_indicator_type(metadata, indicator_type_list, selected_ticker):
    try:
        global conn_params
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
       
        # Prepare the dictionary to store results for each indicator_type
        data_results = {}

        # Prepare the list of field labels
        field_labels = ', '.join(metadata.keys())
         # Map full names in indicator_type_list to their short names
        short_names = map_to_short_names(indicator_type_list)
        # Assuming indicator_type_list is a list, we'll format it as a string for the SQL query
        indicator_types = ', '.join([f"'{indicator}'" for indicator in short_names])

        # Fetch the schema and table name (assuming the same for all fields based on the metadata)
        # Since the schema_name and table_name seem to be the same for all fields,
        # you can fetch it from any of the fields in metadata, e.g., the first one.
        first_meta = next(iter(metadata.values()))
        schema_name = first_meta['schema_name']
        table_name = first_meta['table_name']
        
        # Build the SQL query dynamically based on schema, table, and fields (indicator_type)
        query = f"""

            SELECT ed.{field_labels}, p.ticker, p.price
            FROM {schema_name}.{table_name} ed
            INNER JOIN analytix.prices p ON ed.date = p.date
            WHERE ed.indicator_type IN ({indicator_types}) AND p.ticker = %s
            ORDER BY ed.id ASC; 

        """

        # Execute the query
        cursor.execute(query,(selected_ticker,))
      
        # Fetch all the rows from the result
        rows = cursor.fetchall()

        # Fetch column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Process and store the results
        for row in rows:
            row_dict = dict(zip(column_names, row))
            indicator_type = row_dict['indicator_type']
            if indicator_type not in data_results:
                data_results[indicator_type] = []
            data_results[indicator_type].append(row_dict)
    
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        return data_results

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def convert_to_dataframe(data_results):
    # Initialize a list to store DataFrames for each feature
    dataframes = []

    # Iterate through the dictionary for each feature (indicator_name)
    for indicator_name, rows in data_results.items():
        # Create a DataFrame from the list of tuples/lists
        temp_df = pd.DataFrame(rows, columns=['date', 'id', 'indicator_name', 'indicator_type', 'source', 'updated_at', 'value', 'ticker', 'price'])
        # Drop 'id' and 'updated_at' columns
        temp_df.drop(['id', 'updated_at'], axis=1, inplace=True)
        # Append the temp_df to the list of DataFrames
        dataframes.append(temp_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    # Drop rows with all NaNs except for 'date' column
    df.dropna(how='all', subset=df.columns.difference(['date']), inplace=True)
    
    # Pivot the DataFrame to use 'indicator_type' and 'ticker' as new columns for 'value' and 'price'
    df_pivot = df.pivot(index=['date', 'indicator_name', 'source'], 
                        columns=['indicator_type', 'ticker'], 
                        values=['value', 'price'])
    
    # Flatten the MultiIndex columns and create descriptive names for each
    new_columns = []
    for col in df_pivot.columns:
        if col[0] == 'value':
            # For value columns, use the indicator type name
            new_columns.append(f"{col[1]}")  # e.g., 'TTLCONS'
        elif col[0] == 'price':
            # For price columns, use the ticker name with '_price'
            new_columns.append(f"{col[1]}_price")  # e.g., 'TTLCONS_price'
        ticker_column=col[2]
    
    # Assign these new column names to the DataFrame
    df_pivot.columns = new_columns
    
    # Identify columns with '_price' suffix as price columns
    price_columns = [col for col in df_pivot.columns if col.endswith('_price')]
    if price_columns:
        # Sum the price columns to create a new ticker column with combined values
        df_pivot[ticker_column] = df_pivot[price_columns].sum(axis=1) #e.g SPY

    # Drop individual price columns after summing
    df_pivot = df_pivot.drop(columns=price_columns)
    
    # Reset index to turn multi-level index into columns
    df_pivot.reset_index(inplace=True)
    
    # Now we need to combine rows with the same date
    # Define the columns you know need special handling
    indicator_col = 'indicator_name'
    source_col = 'source'
    last_col = ticker_column  # Use the dynamic ticker name identified

    # Define dynamic columns 
    dynamic_cols = [col for col in df_pivot.columns if col not in [indicator_col, source_col, last_col, 'date']]
  
    # Define aggregation rules
    agg_dict = {
        indicator_col: ', '.join,        # Join indicator names with commas
        source_col: 'first',             # Take the first source value
        last_col: 'first',               # Take the first value of the ticker column
    }

    # Add dynamic columns to the aggregation dictionary with 'first' non-None/non-NaN value logic
    for col in dynamic_cols:
        agg_dict[col] = 'first'

    # Apply the grouping and aggregation
    df_combined = df_pivot.groupby('date').agg(agg_dict).reset_index()
    # Reorder columns to have last_col at the end
    reordered_columns = [col for col in df_combined.columns if col != last_col] + [last_col]
    df_combined = df_combined[reordered_columns]

    return df_combined


def display_economic_data(economic_indicator_list,selected_ticker):
   
    # Call the function and store the returned data
    metadata = fetch_metadata()
    # Call the function and store the returned data
    values = fetch_data_by_indicator_type(metadata,economic_indicator_list,selected_ticker)
    # Check if values is empty or None, and return an empty DataFrame if so
    if not values:  # values is empty or None
        return pd.DataFrame()  # Return an empty DataFrame
    # Convert the dictionary to a pandas dataframe
    df = convert_to_dataframe(values)
    df = df.sort_values(by=[ 'date'], ascending=[False])
    return df


def visualize_economic_data(vis_economic_indicator_list,selected_ticker):
    # Call the function and store the returned data
    metadata = fetch_metadata()
    # Call the function and store the returned data
    values = fetch_data_by_indicator_type(metadata,vis_economic_indicator_list,selected_ticker)
    if not values:  # values is empty or None
        return pd.DataFrame()  # Return an empty DataFrame
    # Convert the dictionary to a pandas dataframe
    df = convert_to_dataframe(values)

    return df

def economic_preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(by=[ 'date'], ascending=[False])
    return df
def select_and_normalize_eco_columns(df, selected_features_full_name, selected_ticker, selected_duration_key):
    mapped_duration = duration[selected_duration_key]
    selected_features = map_to_short_names(selected_features_full_name)
    
    # Add the selected predictor column to the features before normalization
    all_features = selected_features + [selected_ticker]
    
    # Filter out features that are not present in the DataFrame
    available_features = [feature for feature in all_features if feature in df.columns]
    
    if not available_features:
        # If no features are available, return empty DataFrames and None for the scaler
        return pd.DataFrame(), pd.DataFrame(), None

    # Select the relevant columns directly from the dataframe
    df_selected = df[available_features]

    # Handle percentage strings and convert to float where necessary
    df_selected = df_selected.apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) and x.endswith('%') else x)

    # Forward fill and backward fill to handle missing data
    df_selected.ffill(inplace=True)
    df_selected.bfill(inplace=True)
    
    # Convert all data to numeric
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=available_features)

    # Combine with 'date' and any other identifier columns
    df_final_normalized = pd.concat([df[['date', 'indicator_name', 'source']].reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    # Create lagged columns based on the selected duration key
    if selected_ticker in df_final_normalized.columns:
        df_final_normalized[f'{selected_ticker}_lag'] = df_final_normalized[selected_ticker].shift(mapped_duration)
    
    # Final DataFrame that retains the original number of rows
    df_final = df_final_normalized.copy()
    
    # Drop the lagged column for the normalized DataFrame
    df_final_normalized = df_final_normalized.drop(columns=[f'{selected_ticker}_lag'], errors='ignore')

    return df_final_normalized, df_final, scaler


def split_eco_data(df_final):
    # Directly split the data into training and testing sets
    train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)
    
    return train_df, test_df


def prepare_ml_data_eco_data(train_df, test_df, target_col):
    X_train = train_df.drop(columns=['date', 'indicator_name','source', target_col, f'{target_col}_lag'])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=['date', 'indicator_name','source', target_col, f'{target_col}_lag'])
    y_test = test_df[target_col]
   
    return X_train, y_train,X_test, y_test

def inverse_transform_and_plot_predictions_eco_data(model, X_test, y_test, scaler, target_col, n_display=10):
    # Filter columns in X_test to match model's input feature names
    expected_features = model.feature_names_in_  # Model's expected features from training
    X_test_filtered = X_test[expected_features]
    
    # Extract dates for x-axis labels
    dates = y_test['date'].values

    # Prepare target column in y_test for plotting
    y_test_single = y_test[target_col].values.flatten()

    # Predict
    y_pred_test = model.predict(X_test_filtered)

    # Prepare arrays for inverse transform
    dummy_shape = np.zeros((y_pred_test.shape[0], scaler.n_features_in_))
    dummy_shape[:, -1] = y_pred_test.flatten()
    y_pred_test_rescaled = scaler.inverse_transform(dummy_shape)[:, -1]

    dummy_shape[:, -1] = y_test_single
    y_test_rescaled = scaler.inverse_transform(dummy_shape)[:, -1]

    # Store all actual and predicted values
    all_actual_prices = y_test_rescaled.flatten()
    all_predicted_prices = y_pred_test_rescaled.flatten()

    # Pagination setup
    total_items = len(all_actual_prices)
    total_pages = math.ceil(total_items / n_display)

    # Get the current page from session state for pagination persistence
    current_page = st.session_state.get('page', 1)
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    # Page selection input
    page_input = st.number_input(f"Select Page (1 to {total_pages}):", min_value=1, max_value=total_pages, value=current_page)

    # Update the current page based on the input
    if page_input != current_page:
        st.session_state['page'] = page_input
        current_page = page_input

    # Calculate start and end indices based on the selected page number
    start_index = (current_page - 1) * n_display
    end_index = min(start_index + n_display, total_items)

    # Get the current subset of actual and predicted prices to display
    current_actual_prices = all_actual_prices[start_index:end_index]
    current_predicted_prices = all_predicted_prices[start_index:end_index]

    # Convert numpy.datetime64 to date part only for the current page
    current_dates = dates[start_index:end_index].astype('datetime64[D]').astype(str)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(18, 6))
    width = 0.35  # Bar width

    # Set positions for bars
    indices = np.arange(len(current_actual_prices))
    
    bars_actual = ax.bar(indices - width/2, current_actual_prices, width, label=f'Actual {target_col}', color='green')
    bars_predicted = ax.bar(indices + width/2, current_predicted_prices, width, label=f'Predicted {target_col}', color='red')

    # Set titles and labels
    ax.set_title(f'Actual vs Predicted {target_col} (Page {current_page})')
    ax.set_ylabel(f'{target_col}')
    ax.set_xlabel('Date')
    ax.set_xticks(indices)
    ax.set_xticklabels(current_dates, rotation=45, ha='right')  # Use formatted dates as x-axis labels
    ax.legend()

    # Show the plot
    st.pyplot(fig)

    # Show current page and range information
    st.write(f'Page {current_page} of {total_pages}')

