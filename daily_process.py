import os
import psycopg2
from datetime import date
import json
import textwrap
import sys

from work_flow import (get_default_embedding_model, generate_embedding, count_sentences,update_default_user_status,
                           sentiments_score, financial_metrix, update_summary_table, process_list,
                           summary, update_embeddings, get_default_generative_models)
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


def automate_process_file(file_path, financial_model, summary_model, sentiment_model, summary_directory, document_id, 
                          run_summary, run_financial_metrics, run_sentiment_score):
    
    current_date = date.today()
    txt_file = os.path.basename(file_path)
    embedd_model = get_default_embedding_model(conn_params)
    
    # Load the clean transcript document
    with open(file_path, 'r') as f:
        text = f.read()

    # Split the text into manageable chunks
    text_chunks = textwrap.wrap(text, 6000 * 4, break_long_words=False)

    all_embeddings, docu_embedd_cost = [], []
    for chunk in text_chunks:
        doc_embedd, chunk_cost = generate_embedding(chunk, embedd_model)
        all_embeddings.extend(doc_embedd)
        docu_embedd_cost.append(chunk_cost)

    total_doc_embedd_cost = sum(docu_embedd_cost)

    # Update process date in the document table
    automate_update_process_date(document_id, current_date)

    # Initialize accumulators for summary, financial metrics, and sentiment
    final_summary, summary_embedd_cost, total_summary_cost = None, None, None
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
            subsummary, subsumm_in_cost, subsumm_out_cost = summary(text_input=items, model=summary_model)
            sub_summaries.append(subsummary)
            subsummary_input_cost.append(subsumm_in_cost)
            subsummary_output_cost.append(subsumm_out_cost)

        sub_summaries_len = ''.join(str(item) for item in sub_summaries)
        final_summary, final_summ_inp_cost, final_summ_out_cost = summary(sub_summaries_len, model=summary_model)
        print("summary has been generated successfully")

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

        # Generate summary embeddings
        summary_embedd, summary_embedd_cost = generate_embedding(final_summary, embedd_model)

        # Save the summary to a file
        summary_filename = os.path.join(summary_directory, f"{txt_file}_summary.txt")
        with open(summary_filename, 'w') as summary_file:
            summary_file.write(final_summary)

    # Generate Financial Metrics if flagged
    if run_financial_metrics:
        financial_metrices, finan_input_cost, finan_output_cost = financial_metrix(final_summary, financial_model)
        print("Financial Metrics has been generated successfully")

    # Generate Sentiment Score if flagged
    if run_sentiment_score:
        sent_score, sent_input_cost, sent_output_cost = sentiments_score(final_summary, sentiment_model)
        print("Sentiment scores has been generated successfully")

    # At the end, update everything in the database in one go
    summary_id=update_summary_table(
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
        finan_output_cost if run_financial_metrics else 0,
        sent_score if run_sentiment_score else None,
        sentiment_model if run_sentiment_score else None,
        sent_input_cost + sent_output_cost if run_sentiment_score else 0,
        sent_input_cost if run_sentiment_score else 0,
        sent_output_cost if run_sentiment_score else 0
    )

    # Update embeddings for document
    update_embeddings(document_id, None, embedd_model, current_date, all_embeddings, total_doc_embedd_cost)

    # Update embeddings for summary if it was generated
    if run_summary:
        update_embeddings(document_id, summary_id, embedd_model, current_date, summary_embedd, summary_embedd_cost)


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
        else:
            print(f"Skipping invalid path: {file_loc}")

    print("Process Completed")

import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    
    # Use action='store_true' to create flags that default to False if not passed
    parser.add_argument('--summary', action='store_true', help='Run Summary Generation (default: False)')
    parser.add_argument('--financial', action='store_true', help='Run Financial Metrics (default: False)')
    parser.add_argument('--sentiment', action='store_true', help='Run Sentiment Analysis (default: False)')

    args = parser.parse_args()

    # Condition: financial and sentiment can only be True if summary is True
    if not args.summary and (args.financial or args.sentiment):
        print("Error: 'financial' and 'sentiment' cannot be True if 'summary' is False.")
        sys.exit(1)  # Exit with an error code

    # Proceed with the process if the condition is met
    process_all_files(run_summary=args.summary, 
                      run_financial_metrics=args.financial, 
                      run_sentiment_score=args.sentiment)
