import os
import time
import pandas as pd
import psycopg2
from openai._client import OpenAI
from func import get_openai_key, get_default_embedding_model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast  # To safely evaluate string representations of lists

from dotenv import load_dotenv
import os

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

openai_api_key = get_openai_key(conn_params)
client = OpenAI(api_key = openai_api_key) 
embedding_model= get_default_embedding_model(conn_params)


def generate_embedding(text):
    try:
        global embedding_model
        response = client.embeddings.create(
            input=text,
            model=embedding_model  # Specify the model for generating embeddings
        )
        # Extract and return the embedding vector
        embedding_vector = response.data[0].embedding
        return embedding_vector
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def update_questions_table(question, vector_embedding):
    try:
        global embedding_model,conn_params
        # Establish the connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set the schema to use
        cursor.execute('SET search_path TO documents;')

        # Prepare the SQL query to insert a new record into the questions table
        insert_query = """
            INSERT INTO questions (question, embedding_model, vector_embedding)
            VALUES (%s, %s, %s);
        """

        # Execute the insert query with the provided parameters
        cursor.execute(insert_query, (question, embedding_model, vector_embedding))

        # Commit the changes to the database
        conn.commit()

        print(f"Successfully inserted question: {question} into the questions table.")

    except Exception as e:
        print(f"Error inserting question into the database: {e}")
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def find_best_matching_summary(question_vector):
    try:
        global conn_params
        # Convert question_vector to a NumPy array
        question_vector = np.array(question_vector)

        # Establish the connection to the database
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Retrieve all vector_embeddings and document_id from the embeddings table
        query_summaries = """
            SELECT vector_embedding, document_id
            FROM documents.embeddings;
        """
        cursor.execute(query_summaries)
        summaries = cursor.fetchall()

        # Prepare data for cosine similarity calculation
        summary_embeddings = []
        document_ids = []

        for vector_embedding_str, document_id in summaries:
            # Convert the string representation of the list to an actual list
            try:
                vector_embedding = ast.literal_eval(vector_embedding_str)
                vector_array = np.array(vector_embedding)

                # Ensure all vectors are of the same length as the question_vector
                if vector_array.shape[0] == question_vector.shape[0]:
                    summary_embeddings.append(vector_array)
                    document_ids.append(document_id)
                # else:
                #     print(f"Skipping vector with inconsistent length: {vector_array.shape}")
                    
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing vector embedding: {e}")
                continue

        # Convert list of summary embeddings to a NumPy array
        if summary_embeddings:
            summary_embeddings = np.array(summary_embeddings)
            
            # Compute cosine similarity between question_vector and each summary_vector
            similarities = cosine_similarity([question_vector], summary_embeddings)
            max_index = np.argmax(similarities)
            
            # Retrieve the document_id with the maximum similarity
            best_document_id = document_ids[max_index]
            
            # Retrieve summary_text for the best_document_id
            query_summary_text = """
                SELECT summary_text
                FROM documents.summaries
                WHERE document_id = %s;
            """
            cursor.execute(query_summary_text, (best_document_id,))
            summary_text = cursor.fetchone()

            print("best_document_id:",best_document_id)
            print("summary_text:",summary_text)

            
            if summary_text:
                return summary_text[0],best_document_id  # Return the summary_text
            else:
                return "No summary text found for the given document_id.", None
        else:
            return "No summary embeddings available.", None

    except Exception as e:
        print(f"Error finding best matching summary: {e}")
        return None, None
    finally:
        # Close connections
        cursor.close()
        conn.close()


def get_system_prompt():
    try:
        global conn_params
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Define the SQL query to fetch the system prompt
        query = """
            SELECT config_value
            FROM system_control.system_configuration
            WHERE 
              config_key = 'system prompt'
              AND "default" = true;
        """
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch the result (assuming there's only one matching row)
        result = cursor.fetchone()
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        # If a result is found, return the system prompt; otherwise, return None
        if result:
            return result[0]
        else:
            return None

    except Exception as e:
        print(f"Error fetching system prompt: {e}")
        return None
    
def get_default_chatbot_model():
    try:
        global conn_params
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Define the SQL query to fetch both the LLM Model 
        query = """
            SELECT config_key, config_value
            FROM system_control.system_configuration
            WHERE 
              config_key IN ('LLM Model')
              AND "default" = true;
        """
        
         # Execute the query
        cursor.execute(query)
        
        # Fetch the result (assuming there's only one matching row)
        result = cursor.fetchone()
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        # If a result is found, return the system prompt; otherwise, return None
        if result:
            return result[0]
        else:
            return None

    except Exception as e:
        print(f"Error fetching LLM model: {e}")
        return None


def chatbot(user_message, selected_model, reference):
    if selected_model == "None":
        selected_model = get_default_chatbot_model()

    if selected_model != "None":
        system_prompt = get_system_prompt()

        assistant = client.beta.assistants.create(
            name="Question answer chatbot",
            instructions=f'''
            {system_prompt}
            This is provided reference information to answer from: {reference}.
            ''',
            model=selected_model,
            tools=[{"type": "code_interpreter"}],
        )

        # Create a Thread
        thread = client.beta.threads.create()

        # Add a Message to the Thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id, 
            role="user",
            content=user_message
        )

        return assistant, thread


def answer_query(assistant, thread):
      # Run the Assistant
      run = client.beta.threads.runs.create(thread_id=thread.id,assistant_id=assistant.id)

      # If run is 'completed', get messages and print
      while True:
        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id,run_id=run.id)
        time.sleep(15)
        if run_status.status == 'completed':
          messages = client.beta.threads.messages.list(thread_id=thread.id)
          break
        else:
          ### sleep again
          time.sleep(2)

      for messages in reversed(messages.data):
        if messages.role=="assistant":
          return messages.content[0].text.value
        

def document_reference(document_id):
    try:
        global conn_params
        # Establish the connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set the schema to use
        cursor.execute('SET search_path TO documents;')

        # Prepare the SQL query to fetch title and file_loc using document_id
        fetch_query = """
            SELECT title, file_loc
            FROM documents
            WHERE document_id = %s;
        """

        # Execute the fetch query
        cursor.execute(fetch_query, (document_id,))

        # Fetch the title and file_loc
        document_info = cursor.fetchone()

        if document_info:
            title, file_loc = document_info
            return title, file_loc
        else:
            print(f"No document found with document_id {document_id}")
            return None, None

    except Exception as e:
        print(f"Error fetching document details: {e}")
        return None, None
    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()