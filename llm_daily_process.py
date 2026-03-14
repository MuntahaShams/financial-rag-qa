import os
import psycopg2
from datetime import date
import json
import subprocess

from work_flow import (    count_sentences,update_default_user_status,
                           sentiments_score, financial_metrix, process_list,
                           summary, get_default_generative_models)
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

print(f"Active environment: {env}")
print(f"Using database: {conn_params['dbname']}")
print(f"Active user ID: {active_user_id}")

def run_ollama_prompt(prompt):
    """
    Sends a prompt to Llama 3.2 using the Ollama API and returns the response.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2"],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"


def sentiments_score_llama(summary):
    """
    Analyzes sentiment and assigns a numeric score for financial performance using Llama 3.2.
    """
    prompt = f"""
    As a specialized chatbot tasked with analyzing company conference transcriptions {summary}, your objective is to assess the company's performance in terms of sales, profit, and debt.\
    Provide a numeric score on a scale of 1 to 100 based on the transcript content. Assign a score between 1 to 30 for a poor report, 31-60 for a neutral report, 61-90 for a good report, \
    and 91-100 for an excellent report.
    Output only the integer score without additional information, for example: 30
    """
    return run_ollama_prompt(prompt)

def summary_llama(text_input):
    """
    Generates a financial summary using Llama 3.2.
    """
    prompt = f"""
      Generate a comprehensive summary for the text {text_input}, highlighting crucial financial metrics and providing insights into key events and contextual considerations.\
      The focus should be on essential financial indicators, including total sales and sales growth rate, net profit and profit growth rate, as well as total debt, debt-to-equity\
      ratio, and debt growth rate. Additionally, incorporate information on significant dates such as the fiscal year start and end dates, earnings release dates, and major product\
      launches or events. It is essential to offer insights into the broader context, covering market trends influencing sales and profit, noteworthy financial achievements or \
      challenges, and strategic decisions impacting debt management.

      Note:
      Do not include any information about Motley Fool
      Do not hallucinate any information
     
    """
    return run_ollama_prompt(prompt)

def extract_file_locs_with_unique_combinations():
    try:
        global conn_params
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        query = """
        SELECT file_loc, title, document_id, year, quarter
        FROM documents.documents
        WHERE process_date IS NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        unique_combinations = set()
        file_locs, titles, document_ids, years, quarters = [], [], [], [], []

        for row in results:
            file_loc, title, document_id, year, quarter = row
            combination = (year, quarter, title)

            if combination not in unique_combinations:
                file_locs.append(file_loc)
                titles.append(title)
                document_ids.append(document_id)
                years.append(year)
                quarters.append(quarter)
                unique_combinations.add(combination)

        cursor.close()
        conn.close()

        return file_locs, titles, document_ids, years, quarters

    except Exception as e:
        print(f"Error extracting file locations: {e}")
        return [], [], [], [], []


def automate_update_process_date(document_id, current_date):
    try:
        global conn_params
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute('SET search_path TO documents;')

        update_query = """
            UPDATE documents
            SET process_date = %s
            WHERE document_id = %s;
        """
        cursor.execute(update_query, (current_date, document_id))
        conn.commit()

        cursor.close()
        conn.close()
        print(f"Successfully updated process date for document_id {document_id}")
        
    except Exception as e:
        print(f"Error updating process date: {e}")

   
def update_summary_table(document_id,final_summary,summary_filename,summary_model,total_summ_cost, summ_in_cost, summ_out_cost,
                         financial_metrices,financial_model,total_finc_cost, in_finan_cost, out_finan_cost):
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
                cost_fm, input_cost_fm, output_cost_fm

            ) VALUES (%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s)
            RETURNING summary_id;
            """


        # Execute the update query with the provided parameters
        cursor.execute(update_query, (
            document_id,final_summary,summary_filename,summary_model,total_summ_cost, summ_in_cost, summ_out_cost,
            financial_metrices,financial_model, total_finc_cost, in_finan_cost, out_finan_cost
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
from datetime import datetime

def update_sentiment_table(document_id, summary_id, sent_score, sentiment_model, total_sent_cost, in_sent_cost, out_sent_cost):
    try:
        global conn_params
        # Establish the connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set the schema to use
        cursor.execute('SET search_path TO documents;')

        # Prepare the SQL query to update the sentiments table
        update_query = """
            INSERT INTO sentiments (
                document_id,
                summary_id,
                sentiment_score,
                scoring_model,
                cost,
                input_cost,
                output_cost,
                analysis_date
            ) VALUES (%s,%s, %s, %s, %s, %s, %s, %s)
        """

        # Provide the current timestamp for analysis_date
        analysis_date = datetime.now()

        # Execute the update query with the provided parameters
        cursor.execute(update_query, (
           document_id, summary_id, sent_score, sentiment_model, total_sent_cost, in_sent_cost, out_sent_cost, analysis_date
        ))

        # Commit the changes to the database
        conn.commit()

        print(f"Successfully updated sentiment table with summary_id {summary_id}")

        # Close the cursor and connection
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error updating sentiment table in the database: {e}")

def check_summary_exist(document_id, summary_model):
    try:
        global conn_params
        # Establish the connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Set the schema to use
        cursor.execute('SET search_path TO documents;')

        # Step 1: Query for summaries by document_id
        query = """
            SELECT summary_id, summary_text, summary_model
            FROM summaries
            WHERE document_id = %s
        """
        cursor.execute(query, (document_id,))
        results = cursor.fetchall()

        # Check if results are empty
        if not results:
            print(f"No summaries found for document_id {document_id}.")
            return None

        # Step 2: Handle single result
        if len(results) == 1:
            summary_id, summary_text, _ = results[0]
            print(f"Found a single summary with summary_id {summary_id} for document_id {document_id}.")
            return summary_id,  summary_text

        # Step 3: Handle duplicates
        print(f"Found {len(results)} summaries for document_id {document_id}. Checking for matching summary_model.")

        # Filter duplicates for the given summary_model
        for row in results:
            if row[2] == summary_model:  # Check if summary_model matches
                summary_id, summary_text, _ = row
                print(f"Found matching summary_model {summary_model} with summary_id {summary_id}.")
                return summary_id,  summary_text

      # If no summary matches the given model, return the first summary
        first_summary_id, first_summary_text, _ = results[0]
        print(f"No specific match for summary_model {summary_model}. Returning the first summary.")
        return first_summary_id,  first_summary_text

    except Exception as e:
        print(f"Error checking summaries table: {e}")
        return None
    finally:
        # Ensure resources are cleaned up
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def automate_process_file(file_path, financial_model, summary_model, sentiment_model, summary_directory, document_id, 
                          run_summary, run_financial_metrics, run_sentiment_score):
    summary_id=0
    current_date = date.today()
    txt_file = os.path.basename(file_path)
   
    # Load the clean transcript document
    with open(file_path, 'r') as f:
        text = f.read()

    # Update process date in the document table
    automate_update_process_date(document_id, current_date)

    # Initialize accumulators for summary, financial metrics, and sentiment
    final_summary, total_summary_cost = None, None
    summary_input_cost, summary_output_cost = 0, 0
    finan_input_cost, finan_output_cost = 0, 0
    sent_score, sent_input_cost, sent_output_cost = None, 0, 0

    # Generate Summary if flagged
    if run_summary:
        new_text, sentences = count_sentences(text, min_words=7)
        max_characters = 112000
        list_elements = 1700
        new_sen_results = process_list(new_text, max_characters, list_elements)

        sub_summaries, subsummary_input_cost, subsummary_output_cost = [], [], []
        for items in new_sen_results:
            if summary_model == "LLAMA":
                subsummary = summary_llama(text_input=items)
                sub_summaries.append(subsummary)
            else:
                subsummary, subsumm_in_cost, subsumm_out_cost = summary(text_input=items, model=summary_model)
                sub_summaries.append(subsummary)
                subsummary_input_cost.append(subsumm_in_cost)
                subsummary_output_cost.append(subsumm_out_cost)

            sub_summaries_len = ''.join(str(item) for item in sub_summaries)
            if summary_model == "LLAMA":
                final_summary = summary_llama(text_input=items)
            else:
                final_summary, final_summ_inp_cost, final_summ_out_cost = summary(sub_summaries_len, model=summary_model)
            
                # Ensure final_summ_inp_cost and final_summ_out_cost are lists
                if isinstance(final_summ_inp_cost, float):
                    final_summ_inp_cost = [final_summ_inp_cost]
                if isinstance(final_summ_out_cost, float):
                    final_summ_out_cost = [final_summ_out_cost]

                subsummary_input_cost.extend(final_summ_inp_cost)
                subsummary_output_cost.extend(final_summ_out_cost)

                summary_input_cost = sum(subsummary_input_cost)
                summary_output_cost = sum(subsummary_output_cost)
                total_summary_cost = summary_input_cost + summary_output_cost

            print("summary has been generated successfully")

        # Save the summary to a file
        summary_filename = os.path.join(summary_directory, f"{txt_file}_summary.txt")
        with open(summary_filename, 'w') as summary_file:
            summary_file.write(final_summary)

    # Generate Financial Metrics if flagged
    if run_financial_metrics:
            if run_summary is not False:
                financial_metrices, finan_input_cost, finan_output_cost = financial_metrix(final_summary, financial_model)
                print("Financial Metrics has been generated successfully")
            else:
                _, get_existing_summary=check_summary_exist(document_id, summary_model)
                financial_metrices, finan_input_cost, finan_output_cost = financial_metrix(get_existing_summary, financial_model)
                print("Financial Metrics has been generated successfully")

    # Generate Sentiment Score if flagged
    if run_sentiment_score :
        if sentiment_model == "LLAMA":
            if run_summary is not False:
                sent_score = sentiments_score_llama(final_summary)
                print("Sentiment scores has been generated successfully with llama")
            else:
                summary_id, get_existing_summary=check_summary_exist(document_id, summary_model)
                sent_score = sentiments_score_llama(get_existing_summary)
                print("Sentiment scores has been generated successfully with llama")

        else:
            if run_summary is not False:
                sent_score, sent_input_cost, sent_output_cost = sentiments_score(final_summary, sentiment_model)
                print("Sentiment scores has been generated successfully")
            else:
                summary_id, get_existing_summary=check_summary_exist(document_id, summary_model)
                sent_score, sent_input_cost, sent_output_cost = sentiments_score(get_existing_summary, sentiment_model)
                print("Sentiment scores has been generated successfully")


    # At the end, update everything in the database in one go
    update_summary_table(
        document_id,
        final_summary if run_summary else None,
        summary_filename if run_summary else None,
        summary_model if run_summary else None,
        total_summary_cost if run_summary else 0,
        summary_input_cost if run_summary else 0,
        summary_output_cost if run_summary else 0,
        json.dumps(financial_metrices) if run_financial_metrics else None,
        financial_model if run_financial_metrics else None,
        finan_input_cost + finan_output_cost if run_financial_metrics else 0,
        finan_input_cost if run_financial_metrics else 0,
        finan_output_cost if run_financial_metrics else 0
    )
    
    update_sentiment_table(
        document_id, summary_id, 
        sent_score if run_sentiment_score else None,
        sentiment_model if run_sentiment_score else None,
        sent_input_cost + sent_output_cost if run_sentiment_score else 0,
        sent_input_cost if run_sentiment_score else 0,
        sent_output_cost if run_sentiment_score else 0)
    
    

def process_all_files(run_summary=True, run_financial_metrics=True, run_sentiment_score=True):
    financial_model, summary_model, sentiment_model = get_default_generative_models()
    print(f"Financial Model: {financial_model}")
    print(f"Summary Model: {summary_model}")
    print(f"Sentiment Model: {sentiment_model}")

    summary_directory = update_default_user_status(active_user_id)
    print("summary_directory:",summary_directory)

    file_locs, titles, document_ids, years, quarters = extract_file_locs_with_unique_combinations()

    for file_loc, title, document_id, year, quarter in zip(file_locs, titles, document_ids, years, quarters):
        if os.path.exists(file_loc) and os.path.isfile(file_loc) and file_loc.endswith('.txt'):
            try:
                automate_process_file(file_loc, financial_model, summary_model, 
                                                                  sentiment_model, summary_directory, document_id,
                                                                  run_summary, run_financial_metrics, run_sentiment_score)
                print(f"Autogeneration done for {file_loc}")
            except Exception as e:
                print(f"Error processing file {file_loc}: {e}")
    #     else:
    #         print(f"Skipping invalid path: {file_loc}")

    # print("Process Completed")

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Use action='store_true' to create flags that default to False if not passed
    parser.add_argument('--summary', action='store_true', help='Run Summary Generation (default: False)')
    parser.add_argument('--financial', action='store_true', help='Run Financial Metrics (default: False)')
    parser.add_argument('--sentiment', action='store_true', help='Run Sentiment Analysis (default: False)')

    args = parser.parse_args()

    # Proceed with the process if the condition is met
    process_all_files(run_summary=args.summary, 
                      run_financial_metrics=args.financial, 
                      run_sentiment_score=args.sentiment)
