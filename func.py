import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2

def pair_plot(preprocessed_df):
    sns.pairplot(preprocessed_df, diag_kind='kde') #normalized
    return plt

def get_openai_key(conn_params):
    try:
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Define the SQL query to fetch the OpenAI API key
        query = """
            SELECT api_key
            FROM system_control.api_keys
            WHERE service_name = 'gpt-4o';
        """
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch the result (assuming there's only one matching row)
        result = cursor.fetchone()
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        # If a result is found, return the API key; otherwise, return None
        if result:
            return result[0]
        else:
            return None

    except Exception as e:
        print(f"Error fetching OpenAI API key: {e}")
        return None

def get_default_embedding_model(conn_params):
    try:
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Define the SQL query
        query = """
            SELECT 
                config_value AS embedd_model 
            FROM 
                system_control.system_configuration 
            WHERE 
                config_key = 'embedding_model' 
                AND "default" = true;
        """
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch the result
        result = cursor.fetchone()
        
        # If a result is found, return the embedding model
        if result:
            embedd_model = result[0]
        else:
            embedd_model = None
        
        return embedd_model
    
    except psycopg2.Error as e:
        print(f"Error fetching embedding model from the database: {e}")
        return None
    
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

