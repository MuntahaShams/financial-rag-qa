from dotenv import load_dotenv
import os
import psycopg2
from decimal import Decimal
import pandas as pd
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

# Function to fetch features based on data_frame_id = 9
def display_feature():
    global conn_params
    # Establish the connection to the database
    connection = psycopg2.connect(**conn_params)
    # Create a cursor object
    cursor = connection.cursor()
    
    # Query to fetch metadata_long_name and the default features for data_frame_id = 9
    cursor.execute("""
        SELECT metadata_long_name, is_default
        FROM metadata.metadata_view
        WHERE data_frame_id = 9
        AND is_visible = True;
    """)
    
    # Fetch all rows
    features = cursor.fetchall()
    
    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Separate the feature names and identify the default features
    feature_names = [row[0] for row in features]
    default_features = [row[0] for row in features if row[1]]  # is_default = True

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
            WHERE metadata_long_name = %s;
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

def fetch_values_from_metadata(metadata):
    """
    Fetch values from the database based on the provided metadata. The query will fetch both 'ticker' and 'date' 
    where available in the table.
    
    :param metadata: Dictionary containing the schema, table, and column information.
    :return: Dictionary containing data fetched from the database, including both 'ticker' and 'date' columns.
    """
    try:
        global conn_params
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Dictionary to store the results for each long_name
        data_results = {}

        # Iterate over each long_name and its corresponding metadata
        for long_name, meta in metadata.items():
            schema_name = meta['schema_name']
            table_name = meta['table_name']
            field_label = meta['field_label']
            
            # Dynamically build the SQL query to fetch both 'ticker' and 'date' columns
            query = f"""
                SELECT distinct fd.security_id, fd.ticker, fd.date,{field_label}, p.price, p.price_change
                FROM analytix.fundamental_data fd
                JOIN analytix.prices p
                ON fd.security_id = p.security_id
                and fd.date=p.date
                where fd.eps < 1000
                order by fd.ticker, date desc;
            """
            # Execute the query
            cursor.execute(query)
            
            # Fetch all the rows from the result
            rows = cursor.fetchall()

            # Process rows
            processed_rows = []
            for row in rows:
                _,ticker_value, date_value, price_value, previous_close_value, column_value = row
                # Convert Decimal objects to float if necessary
                if isinstance(column_value, Decimal):
                    column_value = float(column_value)
                processed_rows.append((ticker_value, date_value, column_value, price_value, previous_close_value))
            
            # Store the processed result for the current long_name
            data_results[long_name] = processed_rows
    
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        return data_results

    except Exception as e:
        print(f"Error fetching data from metadata: {e}")
        return None

def convert_to_dataframe(data_results):
    # Initialize the main DataFrame as None
    df = None

    # Iterate through the dictionary for each feature (long_name)
    for long_name, rows in data_results.items():
        # Convert the list of tuples (ticker, date, column_value, price, previous_close) into a DataFrame
        temp_df = pd.DataFrame(rows, columns=['ticker', 'date', 'price_change', long_name, 'price'])

        # If df is None, initialize it with 'ticker', 'date', 'price', and 'price_change' plus the first feature column
        if df is None:
            df = temp_df
        else:
            # Merge on 'ticker' and 'date' columns to ensure proper alignment, while adding only the new feature
            df = pd.merge(df, temp_df[['ticker', 'date', long_name]], on=['ticker', 'date'], how='outer')

    # Handle null values (optional): Drop rows where all columns except 'ticker' and 'date' are NaN
    df.dropna(how='all', subset=df.columns.difference(['ticker', 'date']), inplace=True)

    # Optionally fill NaN values in the feature columns, leaving 'price' and 'price_change' untouched
    df.fillna(0, inplace=True)

    # Rearrange columns to move 'price' and 'price_change' to the end
    # Get a list of all columns except 'price' and 'price_change'
    cols = [col for col in df.columns if col not in ['price', 'price_change']]

    # Append 'price' and 'price_change' to the end of the column list
    cols += ['price', 'price_change']

    # Reorder the DataFrame columns
    df = df[cols]

    return df

def display_data(long_name_list):
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