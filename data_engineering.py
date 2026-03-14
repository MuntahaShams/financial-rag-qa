import csv
import json
import xml.etree.ElementTree as ET
from typing import List
import psycopg2
from func import get_openai_key
from openai._client import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import xml.etree.ElementTree as ET
import json
from typing import List, Tuple
import streamlit as st
import re
from chardet import detect
# Load environment variables from .env file
load_dotenv()

# # Retrieve connection parameters from environment variables
new_conn_params = {
    "dbname": "bi4fi-db-t",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

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


# OpenAI API parameters
api_key = get_openai_key(conn_params)
print(api_key)
client = OpenAI(api_key=api_key)


def detect_file_type(file_path):
    """
    Detect the type of the file (CSV, JSON, XML, or Excel).
    Handles errors more robustly and attempts more flexible reading strategies.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Detected file type ("json", "xml", "csv", "xlsx") or None if undetectable.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None

    # Try reading as JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        print("File detected as JSON.")
        return "json"
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        pass  # Not a JSON file

    # Try reading as XML
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            ET.parse(f)
        print("File detected as XML.")
        return "xml"
    except (ET.ParseError, UnicodeDecodeError, ValueError):
        pass  # Not an XML file

    # Try reading as CSV with multiple encodings
    encodings_to_try = ["utf-8", "iso-8859-1", "latin1", "ascii"]
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"File detected as CSV with encoding {encoding}.")
            return "csv"
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue  # Try the next encoding

    # Try reading as Excel (Handling both .xls and .xlsx formats)
    try:
        df = pd.read_excel(file_path, engine="openpyxl")  # specify engine for newer Excel formats
        print("File detected as Excel.")
        return "xlsx"
    except Exception as e:
        print(f"Excel file detection failed: {e}")
        pass  # Not an Excel file

    # Fallback to automatic detection for CSV if all else fails
    # Use chardet to detect encoding and attempt CSV read
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)  # Read first 10 KB
            detected_encoding = detect(raw_data)["encoding"]
            print(f"Automatically detected encoding: {detected_encoding}")
        
        # Manual handling of the CSV file to print skipped lines
        skipped_lines = []  # List to store skipped lines
        valid_rows = []     # List to store valid rows
        expected_columns = None  # To keep track of expected column count

        with open(file_path, "r", encoding=detected_encoding) as f:
            for i, line in enumerate(f):
                # Skip the header row if necessary
                if i == 0:
                    valid_rows.append(line.strip().split(","))
                    expected_columns = len(valid_rows[0])  # Set expected column count from the header
                    continue

                # Check if the number of columns matches the expected number
                line_columns = line.strip().split(",")
                if len(line_columns) != expected_columns:
                    skipped_lines.append(line.strip())  # Log skipped lines
                else:
                    valid_rows.append(line_columns)  # Add valid rows

        # Print skipped lines
        if skipped_lines:
            print(f"Skipped {len(skipped_lines)} lines due to column mismatch:")
            for skipped_line in skipped_lines:
                print(f"Skipped line: {skipped_line}")

        # After handling skipped lines, load valid data into DataFrame
        df = pd.DataFrame(valid_rows[1:], columns=valid_rows[0])
        print(f"File detected as CSV with auto-detected encoding '{detected_encoding}'.")

        return "csv"
    except Exception as e:
        print(f"Failed to read as CSV with auto-detected encoding. Error: {e}")
        pass  # Not a CSV file

    print("Could not detect file type.")
    return None


def openai_api_table_extract(uploaded_data_headers,first_three_rows, table_data: pd.DataFrame):
    """
    Calls the OpenAI API with a prompt and table data.

    Parameters:
        prompt (str): The prompt to send to OpenAI.
        table_data (pd.DataFrame): Data from the database to be passed to OpenAI.

    Returns:
        str: Response from OpenAI.
    """
    # Convert the table data to a text format (e.g., CSV) to include in the prompt
    table_text = table_data.to_csv(index=False)
    
    # Define the full prompt, appending the table data
    full_prompt = f"""
    you are provided header and data  of a file: Headers as follows {uploaded_data_headers}\n\n 
    and the few rows of data {first_three_rows}
    and here is the table information:\n{table_text}. 
    You have to tell me provided file alongs to which of the table.( Give only one best match table for the file)
    output format should json with table_id, schema_name, table_name as keys"""
    
    
 # Prepare the messages for the chat model
    messages = [{"role": "user", "content": full_prompt}]

    try:
        # Make the API call to the chat completions endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        
        # Extract the response content (the generated message from the assistant)
        response_content = response.choices[0].message.content


        return response_content

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return None
    


def openai_api_columns_extract(uploaded_data_headers, col_data: pd.DataFrame):

    # Convert the table data to a text format (e.g., CSV) to include in the prompt
    col_data = col_data.to_csv(index=False)
    
    # Define the full prompt, appending the table data
    full_prompt = f"""
    You are provided with a list of file headers: {uploaded_data_headers}, 
    and a list of DB table column names: {col_data}. 
    Your task is to map each file header to the most appropriate DB table column name. 
    Return the result in JSON format.
    Return the result in JSON format with the following structure:
    [
        {{ "header1": "column1" }},
        {{ "header2": "column2" }},
        {{ "header3": "column3" }}
    ]

    Replace 'header1', 'header2', etc., with the actual file headers, and 'column1', 'column2', etc., with the corresponding DB table column names.

    """

    # Prepare the messages for the chat model
    messages = [{"role": "user", "content": full_prompt}]

    try:
        # Make the API call to the chat completions endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        
        # Extract the response content (the generated message from the assistant)
        response_content = response.choices[0].message.content


        # Clean up response to ensure proper JSON formatting
        # Remove any trailing commas after each key-value pair in the array or object
        response_content = re.sub(r',\s*}', '}', response_content)  # Remove trailing comma before closing curly brace
        response_content = re.sub(r',\s*\]', ']', response_content)  # Remove trailing comma before closing square bracket

        # Further clean-up if there are other potential issues
        # Ensure the response is stripped of unnecessary spaces
        response_content = response_content.strip()

        # Try to parse the content as JSON
        gpt_col_response = json.loads(response_content)
        
        return gpt_col_response
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Response content:", response_content)  # Output the raw content for further debugging
        return {}  # Handle error gracefully by returning an empty dictionary or appropriate response
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}  # Handle other potential errors gracefully




def fetch_table_data_from_postgresql():
    """
    Connects to the PostgreSQL database and fetches data from the metadata.md_metadata table.

    Returns:
        pd.DataFrame: Data from the metadata.md_metadata table.
    """
 
    global new_conn_params
    # SQL query to fetch data
    query = """
            SELECT table_id, schema_name, table_name, description
             FROM metadata.md_tables;"""

    
    try:
        # Connect to the database
        conn =psycopg2.connect(**new_conn_params)
        
        # Fetch data into a DataFrame
        df = pd.read_sql(query, conn)
        
        # Close the connection
        conn.close()
        
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def fetch_col_data_from_postgresql(extracted_table_id):
    """
    Connects to the PostgreSQL database and fetches data from the metadata.md_metadata table.

    Returns:
        pd.DataFrame: Data from the metadata.md_metadata table.
    """
 
    global new_conn_params
    query = """
        SELECT column_name
        FROM metadata.md_metadata
        WHERE table_id = %s;
    """

    try:
        # Connect to the database using a context manager
        with psycopg2.connect(**new_conn_params) as conn:
            # Use a cursor context manager
            with conn.cursor() as cur:
                # Fetch data into a DataFrame
                df = pd.read_sql_query(query, conn, params=(extracted_table_id,))
        
        return df

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        raise  # Re-raise the exception after logging

def extract_headers(file_path: str, file_extension: str) -> Tuple[List[str], List[List[str]]]:
    """
    Extract headers and the first three rows from a file (CSV, JSON, XML, XLSX, and delimited text).
    
    Parameters:
        file_path (str): Path to the file.
        file_extension (str): Extension of the file (e.g., 'csv', 'json', 'xml', 'xlsx', 'txt').

    Returns:
        Tuple[List[str], List[List[str]]]: Headers as a list and the first three rows as a list of lists.
    """
    headers = []
    first_three_rows = []
    
    try:
        if file_extension == "csv":
            with open(file_path, mode="r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader)  # First row as header
                first_three_rows = [row for _, row in zip(range(3), csv_reader)]  # Next 3 rows
            
        elif file_extension == "json":
            with open(file_path, mode="r", encoding="utf-8") as file:
                data = json.load(file)
                
                # Case 1: JSON is a list of dictionaries
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    headers = list(data[0].keys())  # Get headers from the keys of the first dictionary
                    first_three_rows = [
                        [entry.get(key, "") for key in headers] for entry in data[:3]
                    ]
                
                # Case 2: JSON is a single dictionary
                elif isinstance(data, dict):
                    # Case 2.1: Values are dictionaries (nested JSON with tabular structure)
                    if all(isinstance(value, dict) for value in data.values()):
                        # Flatten the keys from nested dictionaries
                        headers = list(data[list(data.keys())[0]].keys())  # Take keys from the first nested dictionary
                        first_three_rows = [
                            [value.get(key, "") if isinstance(value, dict) else value for key in headers]
                            for value in data.values()
                        ][:3]
                    else:
                        # Case 2.2: Flat dictionary case
                        headers = list(data.keys())  # Use dictionary keys as headers
                        first_three_rows = [[data.get(key, "") for key in headers]]
                
        elif file_extension == "xml":
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract XML headers and first three rows (assuming XML data can be tabular)
            headers, first_three_rows = extract_headers_from_xml(root)
                
        elif file_extension == "xlsx":
            # Read Excel file and get headers from the first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            headers = df.columns.tolist()  # Extract column headers
            first_three_rows = df.head(3).values.tolist()  # Extract the first three rows as lists
        
        else:
            raise ValueError("Unsupported file extension")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return headers, first_three_rows

def extract_headers_from_xml(root) -> Tuple[List[str], List[List[str]]]:
    """
    Extracts the first three rows from XML data in any structure and calls OpenAI to infer headers.

    Parameters:
        root (Element): The root of the XML tree.

    Returns:
        Tuple[List[str], List[List[str]]]: List of inferred headers and the first three rows as lists.
    """
    first_three_rows = []
    all_rows = []

    # Step 1: Assume a repetitive row structure; try to find a likely tabular pattern
    # Collect potential rows assuming repetitive child elements under the same parent
    for child in root:
        rows = child.findall("./*")  # Modify based on XML depth (example assumes repetitive siblings)
        if rows:
            all_rows = rows
            break

    # Step 2: Process rows into a tabular format (row-based list of lists)
    if all_rows:
        first_three_rows = []
        for row in all_rows[:3]:  # Limit to the first three rows
            row_data = [elem.text.strip() if elem.text else "" for elem in row]
            first_three_rows.append(row_data)

    # Step 3: Call OpenAI API to infer headers
    headers_response = get_headers_from_openai(first_three_rows)
   
    # Parse OpenAI's response to extract headers
    headers = []
    try:
        response_json = json.loads(headers_response)
        headers = response_json.get("inferred_headers", [])
    except Exception as e:
        print(f"Error parsing headers from OpenAI response: {e}")

    return headers, first_three_rows


def get_headers_from_openai(first_three_rows):
    """
    Calls the OpenAI API with a prompt and the first three rows of data. OpenAI will infer the headers.

    Parameters:
        first_three_rows (List[List[str]]): The first three rows of data (no headers).

    Returns:
        str: Response from OpenAI.
    """
    # Define the full prompt, only including the first three rows of data
    full_prompt = f"""
    You are provided with the first three rows of a xml file. The rows are as follows: {first_three_rows}.
    Based on the provided rows, please infer the headers of the file.
    The output format should be json with inferred_headers."""
    
    # Prepare the messages for the chat model
    messages = [{"role": "user", "content": full_prompt}]

    try:
        # Make the API call to the chat completions endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        
        # Extract the response content (the generated message from the assistant)
        response_content = response.choices[0].message.content

        return response_content

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return None


def fetch_table_id(json_data):
    """
    Extracts the table_id from the given JSON-like dictionary.

    Args:
        json_data (dict): Dictionary containing the table metadata.

    Returns:
        int: The extracted table_id, or None if the key is not found.
    """
    try:
        # Extract table_id
        table_id = json_data.get("table_id")
        
        # Ensure table_id is valid
        if table_id is None:
            raise ValueError("table_id is missing in the input data.")
        
        return table_id
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def update_mapping_db(target_schema, target_table, source_file, source_file_type, source_field, target_field):
    global new_conn_params

    try:
        # Connect to the database
        conn = psycopg2.connect(**new_conn_params)
        cursor = conn.cursor()

        # Delete any existing row with the same target_schema, target_table, and source_field
        delete_query = """
            DELETE FROM metadata.md_field_mappings
            WHERE target_schema = %s AND target_table = %s AND source_file = %s AND source_file_type = %s AND source_field = %s;
        """
        cursor.execute(delete_query, (target_schema, target_table, source_file, source_file_type, source_field))

        # Insert the new mapping
        insert_query = """
            INSERT INTO metadata.md_field_mappings (target_schema, target_table, source_file, source_file_type, source_field, target_field)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, (target_schema, target_table, source_file, source_file_type, source_field, target_field))

        # Commit the transaction
        conn.commit()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return "Mapping updated successfully in the database."

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"



def create_mapping_ui(keys):
    # Initialize session state
    if "keys_table" not in st.session_state:
        st.session_state["keys_table"] = [list(k.keys())[0] for k in keys]
        st.session_state["values_table"] = [list(k.values())[0] for k in keys]
        st.session_state["mappings"] = []
        st.session_state["selected_key"] = None
        st.session_state["selected_value"] = None

    # CSS for hover and row styles
    st.markdown(
        """
        <style>
        .hover-row:hover {
            background-color: #187cbd !important;
            color: white !important;
        }
        .hover-row {
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            cursor: pointer;
            width: 100%;
            display: block;
        }
        .green-row {
            background-color: green !important;
            color: white !important;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            width: 100%;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Extract tables for display
    keys_table = st.session_state["keys_table"]
    values_table = st.session_state["values_table"]
    mappings = st.session_state["mappings"]

    # Display tables side by side
    col1, col_mid, col2 = st.columns([4, 2, 4])

    # Keys Table in col1
    with col1:
        st.markdown("### Uploaded Fields")
        for key in keys_table:
            if st.session_state["selected_key"] == key:
                st.markdown(
                    f"<div class='hover-row' style='background-color: #187cbd; color: white;'>{key}</div>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button(f"Select {key}", key=f"key_{key}"):
                    st.session_state["selected_key"] = key

    # Actions Column in col_mid
    with col_mid:
        st.markdown("### Actions")
        st.markdown("<br>" * 2, unsafe_allow_html=True)  # Add spacing
        map_button = st.button("Map")
        reset_button = st.button("Reset All")
        confirm_button = st.button("Confirm")

    # Values Table in col2
    with col2:
        st.markdown("### Database Fields")
        for value in values_table:
            if st.session_state["selected_value"] == value:
                st.markdown(
                    f"<div class='hover-row' style='background-color: #187cbd; color: white;'>{value}</div>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button(f"Select {value}", key=f"value_{value}"):
                    st.session_state["selected_value"] = value

    # Mapped Items Section (Below Table)
    st.markdown("### Mapped Fields")
    for idx, (key, value) in enumerate(mappings):
        # Render each mapping as a green-row
        st.markdown(
            f"<div class='green-row'>{key} → {value}</div>",
            unsafe_allow_html=True,
        )

    # Mapping Logic
    if map_button and st.session_state["selected_key"] and st.session_state["selected_value"]:
        st.session_state["mappings"].append(
            (st.session_state["selected_key"], st.session_state["selected_value"])
        )
        st.session_state["keys_table"].remove(st.session_state["selected_key"])
        st.session_state["values_table"].remove(st.session_state["selected_value"])
        st.session_state["selected_key"] = None
        st.session_state["selected_value"] = None

    # Confirm Logic: Map all keys to corresponding values
    if confirm_button and keys_table and values_table:
        for key, value in zip(keys_table, values_table):
            st.session_state["mappings"].append((key, value))
        st.session_state["keys_table"].clear()
        st.session_state["values_table"].clear()

    # Reset Logic: Return all mappings to unmapped tables
    if reset_button:
        for key, value in st.session_state["mappings"]:
            st.session_state["keys_table"].append(key)
            st.session_state["values_table"].append(value)
        st.session_state["mappings"].clear()
        st.session_state["selected_key"] = None
        st.session_state["selected_value"] = None

    # Save Button
    if st.button("Save Mappings", help="Save all mappings"):
        st.success("Mappings saved successfully!")




def process_and_call_openai(uploaded_data_headers,first_three_rows, file_name, file_type):
    """
    Fetches data from PostgreSQL, then sends it along with a prompt to OpenAI.

    Parameters:
        prompt (str): The prompt to send to OpenAI.

    Returns:
        str: Response from OpenAI.
    """
    # Fetch data from PostgreSQL
    table_data = fetch_table_data_from_postgresql()
    gpt_table_response=openai_api_table_extract(uploaded_data_headers,first_three_rows,table_data)
    gpt_table_response = json.loads(gpt_table_response)
    table_id = fetch_table_id(gpt_table_response)

    col_data = fetch_col_data_from_postgresql(table_id)
   
    if col_data is not None:
        # Call OpenAI API with prompt and table data
        gpt_col_response = openai_api_columns_extract(uploaded_data_headers, col_data)
       
        # Iterate over each dictionary in gpt_col_response
        for mapping in gpt_col_response:
            for source_field, target_field in mapping.items():
                # Call update_mapping_db for each key-value pair
                result = update_mapping_db(
                    source_file=file_name,
                    source_file_type=file_type, 
                    source_field=source_field,
                    target_schema=gpt_table_response["schema_name"],
                    target_table=gpt_table_response["table_name"],
                    target_field=target_field
                )
                print(result)
       
    return gpt_table_response,gpt_col_response
        
        
      