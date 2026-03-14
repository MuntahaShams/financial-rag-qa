import subprocess
from test_openai import OpenAI
import os
import textwrap
import re
import json
import pandas as pd

client = OpenAI(api_key = "")

import subprocess

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
      Do not exceed summary more than 250 words.
    """
    return run_ollama_prompt(prompt)

def financial_metrix_llama(summary):
    """
    Extracts financial metrics from a summary using Llama 3.2.
    """
    prompt = f"""
    Given the following summary {summary}, extract the financial metrics and forecasts that are found in the summary. 
    Return the extracted values in a structured JSON format.
    Financial metrics that are not found, do not return them.

    Metrics may include:
    pe_ratio, fpe_ratio, profit_margin, revenue_growth, earnings_growth, cash, debt, market_value, book_value, cash_flow,\
    levered_cash_flow, dividend_yield, ev, ev_to_ebitda, peg_ratio, roa, roe

    Do not return extra text or JSON formatting markers like ```json. Just give a plain JSON dictionary with one key-value pair per line.
    """
    return run_ollama_prompt(prompt)

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
    return response.choices[0].message.content

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
   
    return response.choices[0].message.content

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
    return response.choices[0].message.content


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


def process_file(file_path, selected_summary_model,selected_finan_model,selected_sent_model):
    rows, rows_llama = [],[]
    
    txt_file = os.path.basename(file_path)
    file_url = file_path

    # Load the clean transcript doc
    with open(file_path, 'r') as f:
        text = f.read()

    # Split the text into chunks based on the token limit
    text_chunks = textwrap.wrap(text, 6000 * 4, break_long_words=False)

    # Process the text
    new_text, sentences = count_sentences(text, min_words=7)
    max_characters = 112000 # max tokens for gpt-4o-mini
    list_elements = 1700 
    new_sen_results = process_list(new_text, max_characters, list_elements)

    sub_summaries, sub_summaries_llama, sub_finan_met_llama = [],[],[]
    for items in new_sen_results:
        subsummary = summary(text_input=items, model=selected_summary_model)
        subsummary_llama=summary_llama(text_input=items)
        sub_finan_metric_llama= financial_metrix_llama(items)

        sub_summaries.append(subsummary)
        sub_summaries_llama.append(subsummary_llama)
        sub_finan_met_llama.append(sub_finan_metric_llama)
       

    # Final Summary
    sub_summaries_len = ''.join(str(item) for item in sub_summaries)
    sub_summaries_len_llama=''.join(str(item) for item in sub_summaries_llama)
    sub_finan_met_len_llama=''.join(str(item) for item in sub_finan_met_llama)

    final_summary = summary(sub_summaries_len, model=selected_summary_model)
    final_summary_llama= summary_llama(text_input=sub_summaries_len_llama)
    final_finan_metrics_llama= summary_llama(text_input=sub_finan_met_len_llama)
  
    # Generating financial metrics
    financial_metrices = financial_metrix(final_summary, selected_finan_model, temperature=0)
    
    # Convert financial_metrices to a dictionary if it's a JSON string
    if isinstance(financial_metrices, str):
        financial_metrices = json.loads(financial_metrices)

    # Add rows for each financial metric, filter out None or empty values
    for key, value in financial_metrices.items():
        if isinstance(value, str):
            if value.strip():  # Check if the string value is not empty after stripping whitespace
                rows.append({
                    'Metric Key': key,
                    'Metric Value': value
                })
    
    # # Convert financial_metrices to a dictionary if it's a JSON string
    # if isinstance(final_finan_metrics_llama, str):
    #     financial_metrices_llama = json.loads(final_finan_metrics_llama)

    # # Add rows for each financial metric, filter out None or empty values
    # for key, value in financial_metrices_llama.items():
    #     if isinstance(value, str):
    #         if value.strip():  # Check if the string value is not empty after stripping whitespace
    #             rows_llama.append({
    #                 'Metric Key': key,
    #                 'Metric Value': value
    #             })
    # generating sentiments score
    sent_score= sentiments_score(final_summary, selected_sent_model)
    sent_score_llama=sentiments_score_llama(final_summary_llama)

    # Append to the list of DataFrames of summary
    df = pd.DataFrame({"ticker": [txt_file], "GPT_summary": [final_summary], "GPT_financial_metrices": [rows], "GPT_scores": [sent_score],
                       "llamma_summary":[final_summary_llama],"Llama_financial_metrics": [final_finan_metrics_llama], "Llama_sentiment_score":[sent_score_llama]})

    return df

def generate_summaries(input_path, selected_summary_model,selected_finan_model,selected_sent_model,clean_up=True):
    dfs_list = []
    if os.path.exists(input_path):
        if os.path.isdir(input_path):
            # Traverse the directory tree
            for root, dirs, files in os.walk(input_path):
                # Filter only the text files (assuming they have a .txt extension)
                txt_files = [file for file in files if file.endswith('.txt')]

                for txt_file in txt_files:
                    file_path = os.path.join(root, txt_file)
                    df = process_file(file_path, selected_summary_model,selected_finan_model,selected_sent_model)
                    dfs_list.append(df)
            

        elif os.path.isfile(input_path) and input_path.endswith('.txt'):
            df = process_file(input_path, selected_summary_model,selected_finan_model,selected_sent_model)
            dfs_list.append(df)
          
        else:
            raise ValueError(f"The input path {input_path} is neither a directory nor a valid text file.")
    else:
        raise FileNotFoundError(f"The path {input_path} does not exist.")
    
    if dfs_list: 
        # Concatenate all DataFrames into a single DataFrame
        final_df = pd.concat(dfs_list, ignore_index=True) 
        
    return final_df

input_path= "/Users/vinodnair/Documents/Apps/Bi2Ai/DATA/Docs"
selected_summary_model="gpt-4o-mini"
selected_finan_model="gpt-4o-mini"
selected_sent_model="gpt-4o-mini"
file_name="/Users/vinodnair/Documents/Apps/Bi2Ai/DATA/Docs/LLM_gen.csv"
# file_name="/Users/muntahashams/Desktop/stock_app/generated_summaries/LLM_gen.csv"
final_df=generate_summaries(input_path, selected_summary_model,selected_finan_model,selected_sent_model)
final_df.to_csv(file_name)

