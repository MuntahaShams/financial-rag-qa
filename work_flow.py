import re
from openai import OpenAI
import os
import pandas as pd
import streamlit as st
import psycopg2
import textwrap
import re
import json
from datetime import date
from func import get_openai_key,get_default_embedding_model

import tiktoken
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


openai_api_key=""
openai_api_key = get_openai_key(conn_params)
client = OpenAI(api_key = openai_api_key)

def update_default_user_status(active_user_id):
    """
    Updates the `default` status in `system_control.system_configuration` for the active user.
    It sets the 'default' to true for the active user_id and false for all other users.
    Also fetches the summary file location for the active user.
    """
    try:
        global conn_params
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set all rows with user_id to "default" = false
        cursor.execute("""
            UPDATE system_control.system_configuration
            SET "default" = false
            WHERE "user_id" IS NOT NULL;
        """)

        # Set the active user_id's "default" = true
        cursor.execute("""
            UPDATE system_control.system_configuration
            SET "default" = true
            WHERE user_id = %s
        """, (active_user_id,))

        # Fetch the summary file location for the active user
        cursor.execute("""
            SELECT config_value
            FROM system_control.system_configuration
            WHERE config_key = 'summary file location'
              AND user_id = %s
        """, (active_user_id,))

        # Fetch the result (assuming one row is returned)
        result = cursor.fetchone()
        summary_directory = result[0] if result else None

        # Commit the changes
        conn.commit()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        # Print confirmation
        print(f"Default status updated for user_id {active_user_id}")

        # Return the summary file directory
        return summary_directory

    except Exception as e:
        print(f"Error updating default status: {e}")
        return None




def get_default_generative_models():
    try:
        global conn_params
        # Establish the connection to the database
        connection = psycopg2.connect(**conn_params)
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Define the SQL query to fetch both the LLM Model and Summary Model
        query = """
            SELECT config_key, config_value
            FROM system_control.system_configuration
            WHERE 
              config_key IN ('Financial Metrics Model', 'Summary Model','Scoring Model')
              AND "default" = true;
        """
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch all matching rows
        results = cursor.fetchall()
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        # Initialize variables to store the model values
        financial_model = None
        summary_model = None
        sentiment_model=None
        
        # Iterate over the results and assign the values to the appropriate variables
        for config_key, config_value in results:
            if config_key == 'Financial Metrics Model':
                financial_model = config_value
            elif config_key == 'Summary Model':
                summary_model = config_value
            elif config_key == 'Scoring Model':
                sentiment_model= config_value
        
        # Return both models as a tuple (Financial Metrics Model, Summary Model,sentiment model)
        return financial_model, summary_model,sentiment_model

    except Exception as e:
        print(f"Error fetching models: {e}")
        return None, None, None


    
def update_summary_table(document_id,final_summary,summary_filename,summary_model,total_summ_cost, summ_in_cost, summ_out_cost,
                         financial_metrices,financial_model,total_finc_cost, in_finan_cost, out_finan_cost,
                         sent_score,sentiment_model,total_sent_cost, in_sent_cost, out_sent_cost):
    try:
        global conn_params
        # Establish the connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set the schema to use
        cursor.execute('SET search_path TO documents;')

        # Prepare the SQL query to update the summaries table
        update_query = """
            INSERT INTO summaries (
                document_id,
                summary_text,
                summary_url,
                summary_model,
                total_cost_summary,input_cost_summary, output_cost_summary,
                financial_metrics,
                llm_model,
                cost_fm, input_cost_fm, output_cost_fm, 
                sentiment_score,
                scoring_model,
                cost_ss, input_cost_ss, output_cost_ss

            ) VALUES (%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING summary_id;
            """


        # Execute the update query with the provided parameters
        cursor.execute(update_query, (
            document_id,final_summary,summary_filename,summary_model,total_summ_cost, summ_in_cost, summ_out_cost,
            financial_metrices,financial_model, total_finc_cost, in_finan_cost, out_finan_cost,
            sent_score,sentiment_model, total_sent_cost, in_sent_cost, out_sent_cost
        ))
        # Fetch the generated summary_id
        summary_id = cursor.fetchone()[0]

        # Commit the changes to the database
        conn.commit()

        print(f"Successfully updated summary with document_id {document_id}")

        # Close the cursor and connection
        cursor.close()
        conn.close()
        return summary_id
    
    except Exception as e:
        print(f"Error updating summary in the database: {e}")

def update_embeddings(document_id,summary_id,embedding_model,current_date,vector_embedding,total_cost):
    try:
        global conn_params
        # Establish the connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set the schema to use
        cursor.execute('SET search_path TO documents;')

        # Prepare the SQL query to insert into the documents table and return the document_id
        update_query = """
            INSERT INTO embeddings (
            document_id, summary_id, embedding_model, process_date, vector_embedding,total_cost
            ) VALUES (%s,%s, %s, %s, %s,%s)
          
        """

        # Execute the insert query with the provided parameters
        cursor.execute(update_query, (
            document_id,summary_id,embedding_model,current_date,vector_embedding,total_cost
        ))

        # Commit the changes to the database
        conn.commit()

        if summary_id==None:
            print(f"Successfully updated document embeddings with document_id {document_id}")
        else:
            print(f"Successfully updated summary embeddings with document_id {document_id} and summary_id {summary_id}")

        # Close the cursor and connection
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error updating document vectors in the database: {e}")
        return None


def update_document_table(title, file_loc, year, quarter,current_date):
    try:
        global conn_params
        # Establish the connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set the schema to use
        cursor.execute('SET search_path TO documents;')

        # Prepare the SQL query to insert into the documents table and return the document_id
        update_query = """
            INSERT INTO documents (
                title,
                file_loc,
                year,
                quarter,
                process_date
            ) VALUES (%s, %s,%s, %s, %s)
            RETURNING document_id;
        """

        # Execute the insert query with the provided parameters
        cursor.execute(update_query, (
            title, file_loc, year, quarter, current_date,
        ))

        # Fetch the generated document_id
        document_id = cursor.fetchone()[0]

        # Commit the changes to the database
        conn.commit()

        print(f"Successfully updated process date for document_id {document_id}")

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return document_id  # Return the generated document_id

    except Exception as e:
        print(f"Error updating process date in the document table: {e}")
        return None

def count_sentences(text, min_words):
    # Split text into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    new_list = []

    # Filter sentences based on word count
    for sentence in sentences:
        if len(sentence.split()) >= min_words:
            new_list.append(sentence)

    num_sentences = len(new_list)
    return new_list, num_sentences


def summary(text_input, model, temperature=0):
    prompt_with_data = f"""
      Generate a comprehensive summary for the text {text_input}, highlighting crucial financial metrics and providing insights into key events and contextual considerations.\
      The focus should be on essential financial indicators, including total sales and sales growth rate, net profit and profit growth rate, as well as total debt, debt-to-equity\
      ratio, and debt growth rate. Additionally, incorporate information on significant dates such as the fiscal year start and end dates, earnings release dates, and major product\
      launches or events. It is essential to offer insights into the broader context, covering market trends influencing sales and profit, noteworthy financial achievements or \
      challenges, and strategic decisions impacting debt management.

      Note:
      Do not include any information about Motley Fool
      Do not hallucinate any information
      Do not exceed summary more than 250 words.
    """
    messages = [{"role": "user", "content": prompt_with_data}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output

    )
    input_cost, output_cost=calculate_token_cost(prompt_with_data,response.choices[0].message.content,model )
    return response.choices[0].message.content,input_cost, output_cost

def financial_metrix(summary, model, temperature=0):
    prompt_with_data = f"""
    Given the following summary{summary}, extract the  financial metrics and forecasts that found in summary. 
    Return the extracted values in a structured JSON format.
    Financial metrics that are not found, do not return them.s

    Metrics may found:
    pe_ratio, fpe_ratio, profit_margin, revenue_growth, earnings_growth, cash, debt, market_value, book_value, cash_flow,\
    levered_cash_flow, dividend_yield, ev, ev_to_ebitda, peg_ratio, roa, roe

    Do not return extra text and ```json just give json dictionary, do not give nested dictionaries, just one key and one value.
    """
    messages = [{"role": "user", "content": prompt_with_data}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output

    )
    input_cost, output_cost=calculate_token_cost(prompt_with_data,response.choices[0].message.content,model )
    return response.choices[0].message.content,input_cost, output_cost

def sentiments_score(summary, model, temperature=0):
    prompt_with_data = f"""
    As a specialized chatbot tasked with analyzing company conference transcriptions{summary}, your objective is to assess the company's performance in terms of sales, profit, and debt.\
    Provide a numeric score on a scale of 1 to 100 based on the transcript content. Assign a score between 1 to 30 for a poor report, 31-60 for a neutral report, 61-90 for a good report, \
    and 91-100 for an excellent report.
    Output only the integer score without additional information for example: 30
    """
    messages = [{"role": "user", "content": prompt_with_data}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output

    )
    input_cost, output_cost=calculate_token_cost(prompt_with_data,response.choices[0].message.content,model )
    return response.choices[0].message.content,input_cost, output_cost

def generate_embedding(text,model):
    try:
        response = client.embeddings.create(
            input=text,
            model=model  # Specify the model for generating embeddings
        )
        # Extract and return the embedding vector
        embedding_vector = response.data[0].embedding
        cost=calculate_embedding_cost(text,model )
        return embedding_vector,cost
         
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None, None

    
def process_list(input_list, max_characters, list_elements):
    new_text = []
    i = 0

    while i < len(input_list):
        # Take the next n elements from the list
        sublist = input_list[i:i + list_elements]

        # Convert each item to a string and concatenate them
        sublist_string = ''.join(str(item) for item in sublist)

        # Calculate the character count of the current sublist
        sublist_characters = len(sublist_string)
        # print(sublist_characters)
        # Check if adding the current sublist exceeds the max_characters limit
        j=0
        while sublist_characters > max_characters:
            # If yes, drop the last element and recalculate the character count
            sublist.pop()
            j =j+1
            sublist_string = ''.join(str(item) for item in sublist)
            sublist_characters = len(sublist_string)
            # print("after:",sublist_characters)

        # # Add the current sublist to the result list
        new_text.append(sublist_string)

        # Move to the next set of n elements
        i += list_elements-j

    return new_text



def extract_file_info(filename):
    # Remove the file extension
    base_name = filename.rsplit('.', 1)[0]

    # Use regular expression to extract parts
    match = re.match(r'([A-Z]+)_(\d{4})_Q(\d)', base_name)
    if match:
        title = match.group(1) + " quaterly report"        # Company ticker: INTC
        year = match.group(2)         # Year: 2023
        quarter = match.group(3)      # Quarter: Q4
        return title, year, quarter
    else:
        return None, None, None


# Function to calculate generative model costs
def calculate_token_cost(prompt, response, model):
    # Initialize the tokenizer for the specified GPT model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Convert prompt and response to strings (if not already)
    request = str(prompt)
    response = str(response)

    # Tokenize the request and response
    request_tokens = tokenizer.encode(request)
    response_tokens = tokenizer.encode(response)

    # Count the total tokens for request and response separately
    input_tokens = len(request_tokens)
    output_tokens = len(response_tokens)
    if model=="gpt-4o-mini":
        # Actual costs per 1 million tokens (based on current pricing)
        cost_per_1M_input_tokens = 0.15  # $0.150 per 1M input tokens
        cost_per_1M_output_tokens = 0.60  # $0.600 per 1M output tokens

    elif model == "gpt-4o":
        cost_per_1M_input_tokens = 5  # $5 per 1M input tokens
        cost_per_1M_output_tokens = 15  # $15 per 1M output tokens

    # Calculate the costs
    input_cost = (input_tokens / 10**6) * cost_per_1M_input_tokens
    output_cost = (output_tokens / 10**6) * cost_per_1M_output_tokens

    # Return the input tokens, output tokens, and total cost
    return input_cost,output_cost
    
# Function to calculate embedding model costs
def calculate_embedding_cost(prompt, model):
    # Initialize the tokenizer for the specified GPT model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Convert prompt and response to strings (if not already)
    request = str(prompt)

    # Tokenize the request and response
    request_tokens = tokenizer.encode(request)

    # Count the total tokens for request and response separately
    input_tokens = len(request_tokens)
    
    if model == "text-embedding-ada-002":
        cost_per_1M_tokens = 0.100  # $$0.100 per 1M input tokens

    # Calculate the costs
    cost = (input_tokens / 10**6) * cost_per_1M_tokens

    # Return the input tokens, output tokens, and total cost
    return cost

def process_file(file_path, selected_summary_model,selected_finan_model,selected_sent_model,summary_directory):
    current_date = date.today()
    rows = []
    
    txt_file = os.path.basename(file_path)
    title, year, quarter = extract_file_info(txt_file)
    file_url = file_path

    embedd_model = get_default_embedding_model(conn_params)
    print("embedd_model:",embedd_model)

    # Load the clean transcript doc
    with open(file_path, 'r') as f:
        text = f.read()

    # Split the text into chunks based on the token limit
    text_chunks = textwrap.wrap(text, 6000 * 4, break_long_words=False)

    # Initialize an empty list to store the embeddings
    all_embeddings, docum_embedd_cost = [],[]

    # Process each chunk to generate embeddings
    for chunk in text_chunks:
        # generating doc embeddings
        doc_embedd, chunck_cost = generate_embedding(chunk, embedd_model)
        all_embeddings.extend(doc_embedd)
        docum_embedd_cost.append(chunck_cost)
    
    #summ all document embedding cost
    total_cost_doc_embedd= sum(docum_embedd_cost)

    # updating process date in doc table
    document_id = update_document_table(title, file_url, year, quarter,current_date)

    # Process the text
    new_text, sentences = count_sentences(text, min_words=7)
    max_characters = 112000 # max tokens for gpt-4o-mini
    list_elements = 1700 
    new_sen_results = process_list(new_text, max_characters, list_elements)

    sub_summaries,subsummary_input_cost,subsummary_output_cost = [],[],[]
    for items in new_sen_results:
        subsummary, subsumm_in_cost, subsumm_out_cost = summary(text_input=items, model=selected_summary_model)
        sub_summaries.append(subsummary)
        subsummary_input_cost.append(subsumm_in_cost)
        subsummary_output_cost.append(subsumm_out_cost)

    # Final Summary
    sub_summaries_len = ''.join(str(item) for item in sub_summaries)
    sub_summaries_length = len(sub_summaries_len)
    final_summary, final_summ_inp_cost, final_summ_out_cost = summary(sub_summaries_len, model=selected_summary_model)
    
    #total summary cost
    if isinstance(final_summ_inp_cost, float):
        final_summ_inp_cost = [final_summ_inp_cost]

    if isinstance(final_summ_out_cost, float):
        final_summ_out_cost = [final_summ_out_cost]

    # Extend the lists with the new costs
    subsummary_input_cost.extend(final_summ_inp_cost)
    subsummary_output_cost.extend(final_summ_out_cost)

    # Calculate the sum of each list
    summary_input_cost = sum(subsummary_input_cost)    
    summary_output_cost = sum(subsummary_output_cost)  
  
    # generating summary embeddings
    summary_embedd, summary_embedd_cost = generate_embedding(final_summary, embedd_model)

    # Generating financial metrics
    financial_metrices,finan_input_cost, finan_output_cost = financial_metrix(final_summary, selected_finan_model, temperature=0)
    
    # Convert financial_metrices to a dictionary if it's a JSON string
    if isinstance(financial_metrices, str):
        financial_metrices = json.loads(financial_metrices)

    # Add rows for each financial metric, filter out None or empty values
    for key, value in financial_metrices.items():
        if isinstance(value, str):
            if value.strip():  # Check if the string value is not empty after stripping whitespace
                rows.append({
                    'ticker': txt_file,
                    'Metric Key': key,
                    'Metric Value': value
                })
    embedding_model = embedd_model

    # Save each summary as a separate .txt file
    summary_filename = os.path.join(summary_directory, f"{txt_file}_summary.txt")
    with open(summary_filename, 'w') as summary_file:
        summary_file.write(final_summary)

    # generating sentiments score
    sent_score, sent_input_cost, sent_output_cost= sentiments_score(final_summary, selected_sent_model)
    financial_metrics_json=json.dumps(financial_metrices)

    # updating summary db
    summary_id=update_summary_table(document_id,final_summary,summary_filename,selected_summary_model,summary_input_cost+summary_output_cost, summary_input_cost, summary_output_cost,
                         financial_metrics_json,selected_finan_model,finan_input_cost+finan_output_cost, finan_input_cost, finan_output_cost,
                         sent_score,selected_sent_model,sent_input_cost+sent_output_cost, sent_input_cost, sent_output_cost)
    
    #updating embeddings for document
    update_embeddings(document_id,None,embedding_model,current_date,all_embeddings,total_cost_doc_embedd)

    #updating embeddings for summary
    update_embeddings(document_id,summary_id,embedding_model,current_date,summary_embedd,summary_embedd_cost)

    
    # Append to the list of DataFrames of summary
    df = pd.DataFrame({"ticker": [txt_file], "summary": [final_summary]})

    # Convert the list of rows to a DataFrame
    financial_metrices_df = pd.DataFrame(rows)

    return df, financial_metrices_df


def generate_summaries(input_path, selected_summary_model,selected_finan_model,selected_sent_model,clean_up=True):
    dfs_list, financial_metrics_list = [], []
    # Update the default user status in the database
    summary_directory=update_default_user_status(active_user_id)
    print("summary_directory:",summary_directory)

    if os.path.exists(input_path):
        if os.path.isdir(input_path):
            # Traverse the directory tree
            for root, dirs, files in os.walk(input_path):
                # Filter only the text files (assuming they have a .txt extension)
                txt_files = [file for file in files if file.endswith('.txt')]

                for txt_file in txt_files:
                    file_path = os.path.join(root, txt_file)
                    df, financial_metrices_df = process_file(file_path, selected_summary_model,selected_finan_model,selected_sent_model,summary_directory)
                    dfs_list.append(df)
                    financial_metrics_list.append(financial_metrices_df)

        elif os.path.isfile(input_path) and input_path.endswith('.txt'):
            df, financial_metrices_df = process_file(input_path, selected_summary_model,selected_finan_model,selected_sent_model,summary_directory)
            dfs_list.append(df)
            financial_metrics_list.append(financial_metrices_df)
        else:
            raise ValueError(f"The input path {input_path} is neither a directory nor a valid text file.")
    else:
        raise FileNotFoundError(f"The path {input_path} does not exist.")

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(dfs_list, ignore_index=True) if dfs_list else pd.DataFrame(columns=["ticker", "summary"])
    final_financial_metrices_df = pd.concat(financial_metrics_list, ignore_index=True) if financial_metrics_list else pd.DataFrame(columns=["ticker", "Metric Key", "Metric Value"])

    return final_df, final_financial_metrices_df

######################################################################################################################################################
# automate job from UI

def extract_file_locs_with_unique_combinations():
    try:
        # Connect to the database
        global conn_params
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Query to select file_loc, title, document_id, year, and quarter where process_date is NULL
        query = """
        SELECT file_loc, title, document_id, year, quarter
        FROM documents.documents
    	where document_id>32000;
        """
        # WHERE process_date IS NULL
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Initialize a set to track unique combinations of year, quarter, and title
        unique_combinations = set()
        
        # Lists to store filtered results
        file_locs = []
        titles = []
        document_ids = []
        years = []
        quarters = []
        
        for row in results:
            file_loc, title, document_id, year, quarter = row
            
            # Create a combination tuple
            combination = (year, quarter, title)
            
            # Check if the combination has already been seen
            if combination not in unique_combinations:
                # Add to result lists
                file_locs.append(file_loc)
                titles.append(title)
                document_ids.append(document_id)
                years.append(year)
                quarters.append(quarter)
                
                # Update the set with this unique combination
                unique_combinations.add(combination)
        
        # Close the connection
        cursor.close()
        conn.close()
        
        return file_locs, titles, document_ids, years, quarters

    except Exception as e:
        print(f"Error extracting file locations: {e}")
        return [], [], [], [], []


def automate_update_process_date(document_id, current_date):
    try:
        global conn_params
        # Establish the connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set the schema to use
        cursor.execute('SET search_path TO documents;')

        # Prepare the SQL query to update the specified fields
        update_query = """
            UPDATE documents
            SET
                process_date = %s
            WHERE document_id = %s;
        """

        # Execute the update query with the provided parameters
        cursor.execute(update_query, (
            current_date,
            document_id
        ))

        # Commit the changes to the database
        conn.commit()

        print(f"Successfully updated process date in doc table for document_id {document_id}")

        # Close the cursor and connection
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error updating process date in doc table: {e}")

def automate_process_file(file_path, financial_model, summary_model,sentiment_model, summary_directory, document_id):
    current_date = date.today()
    rows = []
    
    txt_file = os.path.basename(file_path)
    file_url = file_path

    embedd_model = get_default_embedding_model(conn_params)

    # Load the clean transcript doc
    with open(file_path, 'r') as f:
        text = f.read()

    # Split the text into chunks based on the token limit
    text_chunks = textwrap.wrap(text, 6000 * 4, break_long_words=False)

    # Initialize an empty list to store the embeddings
    all_embeddings, docu_embedd_cost = [],[]

    # Process each chunk to generate embeddings
    for chunk in text_chunks:
        # generating doc embeddings
        doc_embedd, chunck_cost = generate_embedding(chunk, embedd_model)
        all_embeddings.extend(doc_embedd)
        docu_embedd_cost.append(chunck_cost)
    
    #summ all document embedding cost
    total_doc_embedd_cost= sum(docu_embedd_cost)
    # updating process_date in doc table
    automate_update_process_date(document_id,current_date)

    # Process the text
    new_text, sentences = count_sentences(text, min_words=7)
    max_characters = 112000 # max tokens for gpt-4o-mini
    list_elements = 1700 
    new_sen_results = process_list(new_text, max_characters, list_elements)

    sub_summaries,subsummary_input_cost,subsummary_output_cost = [],[],[]
    for items in new_sen_results:
        subsummary, subsumm_in_cost, subsumm_out_cost = summary(text_input=items, model=summary_model)
        sub_summaries.append(subsummary)
        subsummary_input_cost.append(subsumm_in_cost)
        subsummary_output_cost.append(subsumm_out_cost)

    # Final Summary
    sub_summaries_len = ''.join(str(item) for item in sub_summaries)
    sub_summaries_length = len(sub_summaries_len)
    final_summary, final_summ_inp_cost, final_summ_out_cost = summary(sub_summaries_len, model=summary_model)
    
    # Ensure final_summ_inp_cost and final_summ_out_cost are lists
    if isinstance(final_summ_inp_cost, float):
        final_summ_inp_cost = [final_summ_inp_cost]

    if isinstance(final_summ_out_cost, float):
        final_summ_out_cost = [final_summ_out_cost]

    # Extend the lists with the new costs
    subsummary_input_cost.extend(final_summ_inp_cost)
    subsummary_output_cost.extend(final_summ_out_cost)

    # Calculate the sum of each list
    summary_input_cost = sum(subsummary_input_cost) #push
    summary_output_cost = sum(subsummary_output_cost) #push
    total_summary_cost= summary_input_cost +summary_output_cost #push

    # generating summary embeddings
    summary_embedd, summary_embedd_cost= generate_embedding(final_summary, embedd_model)

    # Generating financial metrics
    financial_metrices,finan_input_cost, finan_output_cost = financial_metrix(final_summary, financial_model, temperature=0)
    #finan_input_cost, finan_output_cost push

    # Convert financial_metrices to a dictionary if it's a JSON string to display on UI
    if isinstance(financial_metrices, str):
        financial_metrices = json.loads(financial_metrices)

    # Add rows for each financial metric, filter out None or empty values
    for key, value in financial_metrices.items():
        if isinstance(value, str):
            if value.strip():  # Check if the string value is not empty after stripping whitespace
                rows.append({
                    'ticker': txt_file,
                    'Metric Key': key,
                    'Metric Value': value
                })

    embedding_model = embedd_model

    # Save each summary as a separate .txt file
    summary_filename = os.path.join(summary_directory, f"{txt_file}_summary.txt")
    with open(summary_filename, 'w') as summary_file:
        summary_file.write(final_summary)

    # generating sentiments score
    sent_score,sent_input_cost, sent_output_cost= sentiments_score(final_summary, sentiment_model)
    #sent_input_cost, sent_output_cost

    financial_metrics_json=json.dumps(financial_metrices)
    # updating summary db
    summary_id= update_summary_table(document_id,final_summary,summary_filename,summary_model,summary_input_cost+summary_output_cost, summary_input_cost, summary_output_cost,
                         financial_metrics_json,financial_model,finan_input_cost+finan_output_cost, finan_input_cost, finan_output_cost,
                         sent_score,sentiment_model,sent_input_cost+sent_output_cost, sent_input_cost, sent_output_cost)
    

    #updating embeddings for document
    update_embeddings(document_id,None,embedding_model,current_date,all_embeddings,total_doc_embedd_cost)

    #updating embeddings for summary
    update_embeddings(document_id,summary_id,embedding_model,current_date,summary_embedd,summary_embedd_cost)

    # Append to the list of DataFrames of summary
    df = pd.DataFrame({"ticker": [txt_file], "summary": [final_summary]})

    # Convert the list of rows to a DataFrame
    financial_metrices_df = pd.DataFrame(rows)

    return df, financial_metrices_df


def process_all_files():
    # Update the default user status in the database
    summary_directory=update_default_user_status(active_user_id)
    # Initialize the generative model
    financial_model, summary_model,sentiment_model = get_default_generative_models()
    
    # Print model names
    print(f"Financial Model: {financial_model}")
    print(f"Summary Model: {summary_model}")
    print(f"Sentiment Model: {sentiment_model}")
    # Get the list of file locations with null process_date
    file_locs, titles, document_ids, years, quarters  = extract_file_locs_with_unique_combinations()
    
    # Check if file_locs is a list and not empty
    for file_loc, title, document_id, year, quarter in zip(file_locs, titles, document_ids, years, quarters):
            
            # Check if the path exists and is a file with a .txt extension
            if os.path.exists(file_loc) and os.path.isfile(file_loc) and file_loc.endswith('.txt'):
                try:
                    # Process the file
                    df, financial_metrices_df = automate_process_file(file_loc, financial_model, summary_model,sentiment_model,summary_directory, document_id)
                    print(f"Autogeneration done for {file_loc}")
                    
                    # You can also do something with df and financial_metrices_df here if needed
                except Exception as e:
                    print(f"Error processing file {file_loc}: {e}")
            else:
                print(f"Skipping invalid path files: {file_loc}")
    else:
        print("Process Completed")
        