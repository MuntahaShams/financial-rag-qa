import subprocess
import os
import textwrap
import re
import pandas as pd

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

def summary(text_input):
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

def count_tokens_in_strings(file_contents):
    """
    Counts tokens in a list of file content strings and prints the results.

    Args:
        file_contents (list): A list of strings, where each string represents the content of a file.
    """
    for i, content in enumerate(file_contents):
        # Tokenize the content (basic method: split on whitespace)
        tokens = content.split()
        token_count = len(tokens)

        print(f"File {i + 1}, Tokens: {token_count}")
    return token_count

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


def process_file(file_path):
   
    txt_file = os.path.basename(file_path)
    # print txt_file
    # Load the clean transcript doc
    with open(file_path, 'r') as f:
        text = f.read()

    # Split the text into chunks based on the token limit
    text_chunks = textwrap.wrap(text, 6000 * 4, break_long_words=False)

    # Process the text
    new_text, sentences = count_sentences(text, min_words=7)
    max_characters = 125000 
    list_elements = 1700 
    new_sen_results = process_list(new_text, max_characters, list_elements)
   
    doc_length=count_tokens_in_strings(new_sen_results)
    if doc_length > 127000:
        sub_summaries = []
        for items in new_sen_results:
            subsummary = summary(text_input=items)
            sub_summaries.append(subsummary)
        
        # # Final Summary
        sub_summaries_len = ''.join(str(item) for item in sub_summaries)
        final_summary = summary(sub_summaries_len)
        sent_score_llama=sentiments_score_llama(final_summary)
    else:
        sent_score_llama=sentiments_score_llama(new_sen_results)

    # Append to the list of DataFrames of summary
    df = pd.DataFrame({"ticker": [txt_file], "Llama_sentiment_score":[sent_score_llama]})

    return df

def generate_summaries(input_path):
    dfs_list = []
    if os.path.exists(input_path):
        if os.path.isdir(input_path):
            # Traverse the directory tree
            for root, dirs, files in os.walk(input_path):
                # Filter only the text files (assuming they have a .txt extension)
                txt_files = [file for file in files if file.endswith('.txt')]

                for txt_file in txt_files:
                    file_path = os.path.join(root, txt_file)
                    df = process_file(file_path)
                    dfs_list.append(df)
            

        elif os.path.isfile(input_path) and input_path.endswith('.txt'):
            df = process_file(input_path)
            dfs_list.append(df)
          
        else:
            raise ValueError(f"The input path {input_path} is neither a directory nor a valid text file.")
    else:
        raise FileNotFoundError(f"The path {input_path} does not exist.")
    
    if dfs_list: 
        # Concatenate all DataFrames into a single DataFrame
        final_df = pd.concat(dfs_list, ignore_index=True) 
        
    return final_df

input_path= "/Users/muntahashams/Desktop/stock_app/uploads"

file_name="/Users/muntahashams/Desktop/stock_app/generated_summaries/LLM_gen.csv"
final_df=generate_summaries(input_path)
final_df.to_csv(file_name)





