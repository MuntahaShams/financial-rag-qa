from dotenv import load_dotenv
import os
import psycopg2
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


    

def summary_sentiment_score():
    """
    Fetches data from the sentiment_score table and joins it with the security_view table
    based on the ticker column. The result includes selected columns from the security_view
    table, ordered by sentiment_score descending and ticker ascending.

    Returns:
        pd.DataFrame: A DataFrame containing the joined data.
    """
    global conn_params

    # SQL query to fetch and join data
    query = f"""
        SELECT 
            *
        FROM 
            documents.summary_sentiment_score
        ORDER BY 
            sum_sent_score DESC,
            ticker ASC;
    """

    try:
        # Connect to the database
        connection = psycopg2.connect(**conn_params)
        
        # Fetch the joined data into a DataFrame
        df = pd.read_sql(query, connection)
        
        # Close the connection
        connection.close()
        
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# Function to fetch unique values for dropdown filters
def get_unique_values():
    """
    Fetch unique years and quarters for sidebar filters.
    """
    try:
        query = """
            SELECT DISTINCT year, quarter 
            FROM documents.summary_sentiment_score
            ORDER BY year ASC, quarter ASC;
        """
        with psycopg2.connect(**conn_params) as conn:
            df = pd.read_sql(query, conn)
        return df["year"].unique(), df["quarter"].unique()
    except Exception as e:
        print(f"Error fetching filter data: {e}")
        return [], []

# Function to fetch filtered data from the database
def fetch_filtered_data(years, quarters, ticker_search, security_name_search):
    """
    Fetch filtered data based on selected filters.
    """
    try:
        # Base query
        query = """
            SELECT * 
            FROM documents.summary_sentiment_score
            WHERE 1=1
        """

        params = []
        if years:
            query += " AND year = ANY(%s)"
            params.append(years)
        if quarters:
            query += " AND quarter = ANY(%s)"
            params.append(quarters)
        if ticker_search:
            query += " AND ticker ILIKE %s"
            params.append(f"%{ticker_search}%")
        if security_name_search:
            query += " AND security_name ILIKE %s"
            params.append(f"%{security_name_search}%")

        query += " ORDER BY sum_sent_score DESC, ticker ASC"

        # Execute the query
        with psycopg2.connect(**conn_params) as conn:
            df = pd.read_sql(query, conn, params=tuple(params))
        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def create_filtered_dataframe():
    """
    Call summary_sentiment_score() to get a DataFrame and filter it to retain 
    specific columns while removing columns with only null values.
    
    Returns:
        pd.DataFrame: Filtered DataFrame with specific columns and non-null data.
    """
    # Step 1: Call the function to get the original DataFrame
    df = summary_sentiment_score()
    
    # Step 2: List of required columns
    required_columns = [ 'document_id',
        'title', 'author', 'publication_date', 'year', 'quarter', 'ticker', 
        'isin', 'cusip', 'security_name', 'announced_date', 'event_date_time',
        'scoring_model', 'sum_sent_score', 'xfer_date'
    ]
    
    # Step 3: Filter the DataFrame to include only the required columns
    filtered_df = df[required_columns]
    
    # Step 4: Drop columns that contain only null values
    filtered_df = filtered_df.dropna(axis=1, how='all')
    
    # Step 5: Return the filtered DataFrame
    return filtered_df


def filter_selected_col(years, quarters, ticker_search, security_name_search):
    
    # Step 1: Call the function to get the original DataFrame
    df = fetch_filtered_data(years, quarters, ticker_search, security_name_search)
    
    # Step 2: List of required columns
    required_columns = [ 'document_id',
        'title', 'author', 'publication_date', 'year', 'quarter', 'ticker', 
        'isin', 'cusip', 'security_name', 'announced_date', 'event_date_time',
        'scoring_model', 'sum_sent_score', 'xfer_date'
    ]
    
    # Step 3: Filter the DataFrame to include only the required columns
    filtered_df = df[required_columns]
    
    # Step 4: Drop columns that contain only null values
    filtered_df = filtered_df.dropna(axis=1, how='all')
    
    # Step 5: Return the filtered DataFrame
    return filtered_df


# Function to fetch summary_url for a specific document_id
def fetch_summary_url(document_id):
    """
    Fetches the summary_url for a given document_id from the documents.summary_sentiment_score table.

    Args:
        document_id (int): The ID of the document.

    Returns:
        str: The summary_url if found; otherwise, None.
    """
    query = """
        SELECT summary_url
        FROM documents.summary_sentiment_score
        WHERE document_id = %s;
    """
    try:
        # Connect to the database
        connection = psycopg2.connect(**conn_params)
        cursor = connection.cursor()
        
        # Execute the query
        cursor.execute(query, (document_id,))
        
        # Fetch the result
        result = cursor.fetchone()
        
        # Close connection
        cursor.close()
        connection.close()
        
        # Return the summary_url if found
        if result:
            return result[0]  # summary_url is the first (and only) column
        else:
            return None

    except Exception as e:
        print(f"An error occurred while fetching summary_url: {e}")
        return None



# Function to fetch file_loc for a specific document_id
def fetch_document_url(document_id):
    query = """
        SELECT file_loc
        FROM documents.summary_sentiment_score
        WHERE document_id = %s;
    """
    try:
        # Connect to the database
        connection = psycopg2.connect(**conn_params)
        cursor = connection.cursor()
        
        # Execute the query
        cursor.execute(query, (document_id,))
        
        # Fetch the result
        result = cursor.fetchone()
        
        # Close connection
        cursor.close()
        connection.close()
        
        # Return the document_url if found
        if result:
            return result[0]  # document_url is the first (and only) column
        else:
            return None

    except Exception as e:
        print(f"An error occurred while fetching document_url: {e}")
        return None
    

# Function to check if a file path is valid and return the content
def check_valid_path(path):
    if os.path.exists(path):  # Check if the file exists
        with open(path, 'r') as file:
            return file.read()  # Return the file content
    else:
        return "Invalid path"

# Function to get the summary for a document_id
def selected_row_summary(document_id):
    # Fetch the summary URL for the given document_id
    summary_url = fetch_summary_url(document_id)
    
    if summary_url:
        # Check if the URL path is valid and retrieve content
        summary_content = check_valid_path(summary_url)
        return {
            "document_id": document_id,
            "summary_url": summary_url,
            "summary_content": summary_content
        }
    else:
        return {
            "document_id": document_id,
            "summary_url": None,
            "summary_content": "No summary URL found for this document."
        }



def selected_row_document(document_id):
    # Fetch the document URL for the given document_id
    document_url = fetch_document_url(document_id)
    
    if document_url:
        # Check if the URL path is valid and retrieve content
        document_content = check_valid_path(document_url)
        return {
            "document_id": document_id,
            "document_url": document_url,
            "document_content": document_content
        }
    else:
        return {
            "document_id": document_id,
            "document_url": None,
            "document_content": "No document URL found for this document."
        }
