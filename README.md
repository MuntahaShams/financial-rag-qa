# **Streamlit Data Analytics and Management Application**

This Streamlit app is a comprehensive tool for data management, analysis, visualization, and AI-driven insights. It integrates with a database, supports file uploads, and provides user-friendly interfaces for multiple workflows such as chatbot interactions, data engineering, analysis, and workflow management.

---

## **Table of Contents**
- [Overview](#overview)  
- [Features](#features)  
- [File Structure](#file-structure)  
- [Environment Setup](#environment-setup)  
- [Tabs and Functionalities](#tabs-and-functionalities)  
  - [Chatbot](#1-chatbot)  
  - [Data Engineering](#2-data-engineering)  
  - [Data Analysis & Economic Data Analysis](#3-data-analysis--economic-data-analysis)  
  - [Data Management](#4-data-management)  
  - [Workflow](#5-workflow)  
- [Daily Cron Jobs](#daily-cron-jobs)  
- [How to Run](#how-to-run)  
- [Requirements](#requirements)  

---

## **Overview**
This application provides a unified platform for:  
1. Uploading, managing, and analyzing data.  
2. Training AI models for predictive analysis.  
3. Summarizing and scoring financial and sentiment data.  
4. Seamless integration with a database for real-time updates and insights.  
5. A user-friendly interface built using Streamlit.

---

## **Features**
### **1. Chatbot**  
- Interactive chatbot connected to a database and powered by LLMs.  
- Side bar options to select chatbot models.  
- Retrieves insights based on:
  - Citation summaries  
  - Document details  
  - Financial metrics  

### **2. Data Engineering**  
- Upload CSV, XLSX, JSON, or XML files.  
- Extracts columns from the uploaded file.  
- Uses GPT to map columns between the uploaded file and database tables.  
- Displays mapped results on the UI.  

### **3. Data Analysis & Economic Data Analysis**  
- Sidebar options for:
  - **Prediction Models**: List of available models for training and predictions.  
  - **Features for Visualization**: Visualize features on the UI.  
  - **Features for Analysis**: Normalize and prepare data for AI model training.  
  - **Choose History Dates**: Train AI models based on 1-day, 7-day, or 30-day historical data.  
- Tab options include:
  - **Features Analysis**: Display data in a dataframe format.  
  - **Data Exploration**: Pair plot visualization between selected features and predictors.  
  - **AI Model**: Train AI models and display resulting graphs.  
  - **Predicted vs Actual Results**: Bar graph comparison.  
  - **Evaluation Metrics**: Model performance metrics.  

### **4. Data Management**  
- Sidebar options:  
  - **Select Year(s)**: Automatically fetched and editable.  
  - **Select Quarter(s)**: Automatically fetched and editable.  
  - **Search Ticker**: Filter results by ticker.  
  - **Search Security Name**: Filter results by name.  
- Tab options:
  - **Features Analysis**: Display data in a dataframe with row selection.  
  - **Summary**: Display summary files based on selected rows (if file paths exist).  
  - **Document Data**: Display associated document files (if file paths exist).  

### **5. Workflow**  
- Sidebar options:  
  - **List of Models** for:
    - Summary generation.  
    - Financial metrics.  
    - Sentiment scoring.  
  - **Input Type Selection**:
    - None.  
    - File upload.  
    - Folder upload.  
- Functionality:
  - After uploading files/folders, click **Generate Summary** to:
    - Update the database with summary, sentiment scores, and financial metrics.  
    - Display results on the UI.  
  - Use **Autogenerate** to update the database without changing the UI.  

---

## **File Structure**
```plaintext
📂 project-directory  
├── .env                     # Contains environment variables for database connection  
├── config.toml              # Configuration settings (e.g., axMessageSize = 500MB)  
├── func.py                  # Common functions used across scripts  
├── QrTrDr.py                # Main app code (frontend with Streamlit)  
├── chatbot.py               # Backend for Chatbot tab  
├── data_analysis.py         # Backend for Data Analysis tab  
├── data_engineering.py      # Backend for Data Engineering tab  
├── data_management.py       # Backend for Data Management tab  
├── Economic_data_analysis.py# Backend for Economic Data Analysis tab  
├── work_flow.py             # Backend for Workflow tab  
├── prediction_models.py     # AI models for analysis tabs  
├── daily_process.py         # Cron job for summaries, financial metrics, and sentiment scoring  
├── llm_daily_process.py     # Cron job for Llama/GPT sentiment scoring on existing summaries  
├── llama-score.py           # Test script for Llama model  
├── test_llama.py            # Test script for Llama performance  
├── requirements.txt         # Python dependencies  
├── guide.txt                # Instructions to run the app and cron jobs  
└── process_files.log        # Log file for daily process activities  
```

---

## **Environment Setup**
1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Create and activate a virtual environment:  
   ```bash
   python3 -m venv env  
   source env/bin/activate  # For Linux/Mac  
   env\Scripts\activate     # For Windows  
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```

4. Configure the `.env` file for database connection and update `config.toml` if necessary.

---

## **Tabs and Functionalities**

### **1. Chatbot**  
- **Sidebar Options**:  
  - List of chatbot models.  
- **Main Features**:  
  - Answers user queries using citation summaries, documents, and financial metrics retrieved from the database.  

### **2. Data Engineering**  
- **Sidebar Options**:  
  - Upload CSV, XLSX, JSON, or XML files.  
- **Main Features**:  
  - Automatically map columns between the uploaded file and database table using GPT.  
  - Display mapped results for review.  

### **3. Data Analysis & Economic Data Analysis**  
- **Sidebar Options**:  
  - Select prediction models and features for analysis or visualization.  
  - Choose history dates (1-day, 7-day, 30-day) for training.  
- **Main Features**:  
  - Data exploration with pair plots.  
  - Train AI models and display predicted vs actual results and evaluation metrics.  

### **4. Data Management**  
- **Sidebar Options**:  
  - Select year(s), quarter(s), ticker, or security name to filter data.  
- **Main Features**:  
  - Display data in a dataframe with row selection.  
  - View summaries and associated documents.  

### **5. Workflow**  
- **Sidebar Options**:  
  - Choose models and input type (None, File, Folder).  
- **Main Features**:  
  - Upload files or folders to generate summaries, sentiment scores, and financial metrics.  
  - Update the database with results and display them on the UI.  

---

## **Daily Cron Jobs**
- **`daily_process.py`**:  
  - Generates summaries, financial metrics, and sentiment scores.  
  - Outputs logs to `process_files.log`.  
- **`llm_daily_process.py`**:  
  - Uses Llama/GPT models to calculate sentiment scores for existing summaries.  

---

## **How to Run**
1. Run the Streamlit app:  
   ```bash
   streamlit run QrTrDr.py
   ```
2. For Cron Jobs:  
   - Add entries in the system's crontab using commands in `guide.txt`.  

---

## **Requirements**
Refer to the `requirements.txt` file for Python dependencies. Install them using:  
```bash
pip install -r requirements.txt
```

---



readme with functions
data management code easy
UI good