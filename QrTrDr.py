import streamlit as st
from func import pair_plot
from io import BytesIO
import os,shutil
import pandas as pd
from pathlib import Path
from work_flow import generate_summaries,process_all_files
from chatbot import answer_query, chatbot,generate_embedding,update_questions_table,find_best_matching_summary,document_reference
from data_analysis import display_data,display_feature
from economic_data_analysis import economic_feature,get_economic_features,get_tickers,display_economic_data,visualize_economic_data,economic_preprocess_data,select_and_normalize_eco_columns,split_eco_data,prepare_ml_data_eco_data,map_to_short_names,inverse_transform_and_plot_predictions_eco_data
from prediction_models import visualize_decision_tree_regression,visualize_xgboost_regression,visualize_random_forest_results, get_models,get_features,visalize_data,preprocess_data,select_and_normalize_columns, split_data,prepare_ml_data,build_and_train_lr_model,build_and_train_dt_model,build_and_train_rf_model,build_and_train_xgb_model,model_plot,inverse_transform_and_plot_predictions
from data_engineering import detect_file_type,extract_headers,process_and_call_openai
from data_management import create_filtered_dataframe,get_unique_values,filter_selected_col,selected_row_summary,selected_row_document

# Initialize variables to store model and history
model = None
history = None

# Initialize session state variables
if 'best_match' not in st.session_state:
    st.session_state.best_match = ""
    st.session_state.title = ""
    st.session_state.file_loc = ""


def download_button(fig, filename, label="Download"):
    # Convert the Matplotlib figure to bytes
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    # Add a download button for the plot
    st.download_button(
        label=label,
        data=buffer,
        file_name=filename,
        mime="image/png",
    )

# Function to load data incrementally
def load_more_data(df, rows_per_load=5):
    # Initialize session state for tracking the number of rows loaded
    if 'loaded_rows' not in st.session_state:
        st.session_state.loaded_rows = 0  # Start from 0 loaded rows

    # Button to load more data
    if st.button("Load more"):
        st.session_state.loaded_rows += rows_per_load  # Increment the loaded rows by the batch size

    # Determine the end index for slicing the DataFrame
    end_index = st.session_state.loaded_rows if st.session_state.loaded_rows < len(df) else len(df)

    # Display the loaded portion of the DataFrame
    st.dataframe(df.iloc[:end_index])

    # If all rows are loaded, inform the user
    if end_index >= len(df):
        st.write("All data loaded.")


def main():
    # Navigation bar with tabs
    current_tab = st.sidebar.selectbox("Select Tab:", ["Chatbot","Data Engineering","Data Analysis","Economic Data Analysis", "Data Management", "Work Flow"])

    # Display content based on the selected tab
    
    if current_tab == "Chatbot":
        st.title("QrTrDr")
    
        with st.sidebar:
            st.header("List of Chatbot Models")
            models_options = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            selected_model = st.selectbox(
                label="Choose your preferred model:",
                options=["None"] + models_options,
                index=models_options.index("gpt-4o-mini")
            )
        
        center, right = st.columns([7, 3])

        with center:
            response_text=""
            st.subheader("Chatbot")
            st.write("Example Questions:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("What is Apple's profit growth in the third quarter of 2023?")
            with col2:
                st.write("How did Microsoft's revenue change in the second quarter of 2024?")
            with col3:
                st.write("What was Google's net income growth in the first quarter of 2022?")
            # Input message box
            input_message = st.text_area("Type your question here:", key="input_message")

            # Send button
            send_button_clicked = st.button("Send")

            # Check if the Send button is clicked
            if send_button_clicked:
                ques_embedding=generate_embedding(input_message)
                update_questions_table(input_message,ques_embedding)
                st.session_state.best_match,document_id = find_best_matching_summary(ques_embedding)
                st.session_state.title, st.session_state.file_loc=document_reference(document_id)
               
                assistant, thread = chatbot(input_message, selected_model, st.session_state.best_match)
                response_text = answer_query(assistant, thread)
                
                # Update the text area with the new response
                st.text_area("Chatbot Response:", response_text )


                   
        with right:
            st.subheader("Citation")
            with st.expander("Summary"):
                st.write(
                f"""
                <div style='
                    text-align: justify; 
                    overflow-y: auto; 
                    max-height: 400px; 
                    padding-right: 10px;'>
                    {st.session_state.best_match}
                </div>
                """,
                unsafe_allow_html=True
            )
            # Initialize doc_content before the if statement
            doc_content = ""

            if st.session_state.file_loc and st.session_state.file_loc != "":
                if os.path.exists(st.session_state.file_loc):
                    # Open the file and read its contents
                    with open(st.session_state.file_loc, "r") as file:
                        doc_content = file.read()
                else:
                    doc_content="The specified file path does not exist."


            with st.expander("Document"):
                # Display the title and file location
                st.write(f"**Title:** {st.session_state.title}")
                st.write(f"**File Location:** {st.session_state.file_loc}")
                # Add a heading for Document Content
                st.write("**Document Content:**")

                # Use CSS to style the document content
                st.write(
                    f"""
                    <div style='
                        text-align: justify; 
                        overflow-y: auto; 
                        max-height: 400px; 
                        padding-right: 10px;'>
                        {doc_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                
            with st.expander("Financial Metrics"):
                st.write("""
                **Section 3:** Future Projections

                Looking ahead, the company expects continued growth in the digital advertising space, with projections suggesting...
                """)
    elif current_tab =="Data Engineering":
        # Create the directory if it doesn't exist
        upload_dir = Path("uploaded_etl")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Streamlit app title
        st.title("Merge data to DB")

        # Streamlit file upload and processing
        uploaded_file = st.file_uploader("Upload a CSV, JSON, or XML file", type=["csv","xlsx", "json", "xml"])

        if uploaded_file is not None:
            # Save the uploaded file to the directory
            file_path = upload_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Detect the file type based on content
            file_extension = detect_file_type(file_path)

            if file_extension is not None:
                headers,first_three_rows = extract_headers(file_path, file_extension)
            
                # Display success message and file information
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                st.write(f"Detected file type: {file_extension}")
                st.write(f"File saved at: {file_path}")
   
                # Process the headers and call GPT function
                gpt_table_response,gpt_col_response = process_and_call_openai(headers,first_three_rows,uploaded_file.name,file_extension )
                schema_name = gpt_table_response["schema_name"]
                table_name= gpt_table_response['table_name']

                st.markdown(
                f"Uploaded file belongs to **{schema_name}** schema and **{table_name}** table")

                # Prepare data for the table with reversed mapping
                data = [{"Source Field (Uploaded File)": key, "Target Field (Database)": value} for mapping in gpt_col_response for key, value in mapping.items()]

                # Convert to a DataFrame
                df = pd.DataFrame(data)

                # Display table in Streamlit
                st.write("### Mapping Fields")
                st.table(df)
                # # Run the mapping UI function
                # create_mapping_ui(gpt_col_response)

                
               
        else:
            st.info("Please upload a file to proceed.")
    elif current_tab == "Data Analysis":
        # st.title("QrTrDr")
        
        # Create tabs
        tab1, tab2, tab3,tab4, tab5= st.tabs(["Features Analysis", "Data Exploration","AI Model", "Predicted VS Actual Results", "Evaluation Metrics"])

        with st.sidebar:

            # Fetch models from PostgreSQL
            models_options, default_model = get_models()
            st.header("List of Predictive Models")
            # Set default model index
            default_index = models_options.index(default_model) if default_model else 0
            # Add "None" as an option and display the selectbox
            selected_model = st.selectbox(
                label="Choose your preferred Model:",
                options=["None"] + models_options,
                index=default_index + 1  # Add 1 because "None" is the first option
            )
           
            feature_options, fixed_features = display_feature()
            st.header("List of Features for Visualization")
            selected_features = st.multiselect(
            label="Choose your preferred feature for analysis:",
            options=feature_options,
            default=fixed_features
        )          
            # Fetch features from PostgreSQL
            analysis_feature_options, analysis_fixed_features = get_features()
            st.header("List of Features for Analysis")
            # Multiselect for choosing features, with default selected
            analysis_selected_features = st.multiselect(
                label="Choose your preferred feature:",
                options=analysis_feature_options,
                default=analysis_fixed_features
            )
                        
            st.header("History Date")
            duration_options = ['1 Day', '7 Days', '30 Days']
            selected_duration = st.selectbox(
                label="Choose History date to be trained:",
                options=duration_options
        
        )  
            selected_predictor = ["price", "price_change"]
            # Find the index of "price_change" in the list
            default_index = selected_predictor.index("price_change")
            # Use the index in the selectbox
            stock_feature = st.selectbox(
                label="Select the predictor:",
                options=selected_predictor,
                index=default_index  # Pass the integer index here
            )


        # Initialize session state variables if they don't exist
        if 'df_final' not in st.session_state:
            st.session_state.df_final = None
        if 'all_data' not in st.session_state:
            st.session_state.all_data = None
        if 'df_final_normalized' not in st.session_state:
            st.session_state.df_final_normalized = None
        if 'mse' not in st.session_state:
            st.session_state.mse = None
        if 'r2' not in st.session_state:
            st.session_state.r2 = None
        if 'intercept' not in st.session_state:
            st.session_state.intercept =None
        if 'coefficients' not in st.session_state:
            st.session_state.coefficients=None
        if 'scaler' not in st.session_state:
            st.session_state.scaler = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'test_df' not in st.session_state:
            st.session_state.test_df = None
 
        # Content for Tab 1
        with tab1:
            st.subheader("Features Analysis")

            # Initialize session state variables if they don't exist
            if 'all_data' not in st.session_state:
                st.session_state.all_data = None
            if 'df_final_normalized' not in st.session_state:
                st.session_state.df_final_normalized = None

            # Display existing data if it is already present
            if st.session_state.all_data is not None:
                st.write("Actual Data")
                st.dataframe(st.session_state.all_data)

            if st.button("Visualize data"):
                if selected_features and analysis_selected_features:
                    # Display actual data
                    st.session_state.all_data = display_data(selected_features)  # Update session state
                    st.write("Actual Data")
                    st.dataframe(st.session_state.all_data)

                    # Process the data for visualization
                    visalize_data_df = visalize_data(analysis_selected_features)
                    preprocessed_df = preprocess_data(visalize_data_df)

                    # Normalize the data if the model and stock feature are selected
                    if selected_model != "None" and stock_feature != "None":
                        st.session_state.df_final_normalized, st.session_state.df_final, st.session_state.scaler = select_and_normalize_columns(
                            preprocessed_df, analysis_selected_features, stock_feature, selected_duration
                        )

            # Optionally, display normalized data if it exists
            if st.session_state.df_final_normalized is not None:
                st.write("Normalized Data")
                st.dataframe(st.session_state.df_final_normalized)


        # Content for Tab 2
        with tab2:
            st.subheader("Pair Plot")
            if st.session_state.df_final_normalized is not None:
                print(len(st.session_state.df_final_normalized))
                graph = pair_plot(st.session_state.df_final_normalized)
                st.pyplot(graph.gcf(), clear_figure=True)  # Pass the current figure to st.pyplot
                download_button(graph, filename="pair_plot.png")
            else:
                st.warning("Please visualize the model first in Data Visualization Tab to view pair plot.")

        # Content for Tab 3
        with tab3:
            # If 'None' is not selected, load and process the data
            if 'None' not in analysis_selected_features and selected_duration:
                if st.button("Train model"):
                    # Split the data
                    # st.set_option('deprecation.showPyplotGlobalUse', False)
                    train_df, st.session_state.test_df = split_data(st.session_state.df_final) #contains lag data as well

                    X_train, y_train, X_test, y_test = prepare_ml_data(train_df, st.session_state.test_df, stock_feature)

                    if selected_model == "Linear Regression Model":
                        # Train the model and store predictions
                        st.session_state.model, y_pred, st.session_state.mse, st.session_state.r2, st.session_state.intercept, st.session_state.coefficients = build_and_train_lr_model(X_train, y_train, X_test, y_test)
                        
                        # First plot: Without removing outliers
                        st.session_state.fig = model_plot(X_test, y_test, y_pred, scaler=st.session_state.scaler,remove_outliers_flag=False)
                        st.pyplot(st.session_state.fig)
                        
                        # Second plot: With removing outliers
                        st.session_state.fig2 = model_plot(X_test, y_test, y_pred, scaler=st.session_state.scaler,remove_outliers_flag=True)
                        st.pyplot(st.session_state.fig2)


                    elif selected_model == "Random Forest Regressor":
                        st.session_state.model, y_pred, st.session_state.mse, st.session_state.r2 = build_and_train_rf_model(X_train, y_train, X_test, y_test)
                        st.write("RF is trained")
                        # Assuming you have your Random Forest model, X_test, y_test, and y_pred ready
                        visualize_random_forest_results(st.session_state.model, X_test, y_test, y_pred, analysis_selected_features)

                    elif selected_model == "XGBoost Regressor":
                        st.session_state.model, y_pred, st.session_state.mse, st.session_state.r2 = build_and_train_xgb_model(X_train, y_train, X_test, y_test)
                        st.write("Xgboost is trained")
                        # Assuming you've trained your models and obtained predictions:
                        # For XGBoost
                        visualize_xgboost_regression(st.session_state.model, X_test, y_test, y_pred, analysis_selected_features)

                    elif selected_model == "Decision Tree":
                        st.session_state.model, y_pred, st.session_state.mse, st.session_state.r2 = build_and_train_dt_model(X_train, y_train, X_test, y_test)
                        st.write("Decision Tree is trained")
                        # For Decision Tree
                        visualize_decision_tree_regression(st.session_state.model, X_test, y_test, y_pred, analysis_selected_features)

        
        # Content for Tab 4 (Plot predictions)
        with tab4:
            if st.session_state.model is not None and st.session_state.test_df is not None:
                inverse_transform_and_plot_predictions(st.session_state.model, st.session_state.test_df, st.session_state.test_df[['ticker', 'date',stock_feature]], st.session_state.scaler, stock_feature)
            else:
                st.warning("Please train the model first in AI model Tab to view actual VS predicted results.")

        # Content for Tab 5 (Display metrics)
        with tab5:
            if (st.session_state.mse is not None or 
                st.session_state.r2 is not None or 
                st.session_state.intercept is not None or 
                st.session_state.coefficients is not None):
                
                st.write("### Model Performance")
                
                if st.session_state.mse is not None:
                    st.write(f"**Mean Squared Error (MSE):** {st.session_state.mse:.5f}")
                
                if st.session_state.r2 is not None:
                    st.write(f"**R² Score:** {st.session_state.r2:.5f}")
                
                if st.session_state.intercept is not None:
                    st.write(f"**Intercept:** {st.session_state.intercept:.5f}")
                
                if st.session_state.coefficients is not None:
                    # Assuming coefficients is a list, you might want to display all coefficients
                    st.write(f"**Coefficients:** {', '.join(f'{coef:.5f}' for coef in st.session_state.coefficients)}")


            else:
                st.warning("Please train the model first in AI model Tab to view the performance metrics.")

    elif current_tab == "Economic Data Analysis":

        # Create tabs
        tab1, tab2, tab3,tab4, tab5= st.tabs(["Features Analysis", "Data Exploration","AI Model", "Predicted VS Actual Results", "Evaluation Metrics"])

        with st.sidebar:

            # Fetch models from PostgreSQL
            models_options, default_model = get_models()
            st.header("List of Predictive Models")
            # Set default model index
            default_index = models_options.index(default_model) if default_model else 0
            # Add "None" as an option and display the selectbox
            selected_model = st.selectbox(
                label="Choose your preferred Model:",
                options=["None"] + models_options,
                index=default_index + 1  # Add 1 because "None" is the first option
            )
           
            feature_options = economic_feature()
            st.header("List of Economic Features for Visualization")
            vis_selected_features = st.multiselect(
            label="Choose your preferred feature for analysis:",
            options=feature_options
        )          
            # Fetch features from PostgreSQL
            analysis_feature_options = get_economic_features()
            st.header("List of Economic Features for Analysis")
            # Multiselect for choosing features, with default selected
            analysis_selected_features = st.multiselect(
                label="Choose your preferred feature:",
                options=analysis_feature_options,
    
            )
                        
            st.header("History Date")
            duration_options = ['1 Day', '7 Days', '30 Days']
            selected_duration = st.selectbox(
                label="Choose History date to be trained:",
                options=duration_options
        
        )   
            ticker_list = get_tickers()
            default_index = "SPY"

            # Check if "SPY" exists in the ticker_list and get its index, else use 0 as the default
            if default_index in ticker_list:
                index_position = ticker_list.index(default_index)
            else:
                index_position = 0  # Default to the first item if "SPY" is not found

            # Use the index in the selectbox
            selected_ticker = st.selectbox(
                label="Select the ticker:",
                options=ticker_list,
                index=index_position  # Pass the integer index here
            )

        # Initialize session state variables if they don't exist
        if 'df_final' not in st.session_state:
            st.session_state.df_final = None
        if 'all_data' not in st.session_state:
            st.session_state.all_data = None
        if 'df_final_normalized' not in st.session_state:
            st.session_state.df_final_normalized = None
        if 'mse' not in st.session_state:
            st.session_state.mse = None
        if 'r2' not in st.session_state:
            st.session_state.r2 = None
        if 'intercept' not in st.session_state:
            st.session_state.intercept =None
        if 'coefficients' not in st.session_state:
            st.session_state.coefficients=None
        if 'scaler' not in st.session_state:
            st.session_state.scaler = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'test_df' not in st.session_state:
            st.session_state.test_df = None
 
        # Content for Tab 1
        with tab1:
            st.subheader("Features Analysis")

            # Initialize session state variables if they don't exist
            if 'all_data' not in st.session_state:
                st.session_state.all_data = None
            if 'df_final_normalized' not in st.session_state:
                st.session_state.df_final_normalized = None

            # Display existing data if it is already present
            if st.session_state.all_data is not None:
                st.write("Actual Data")
                st.dataframe(st.session_state.all_data)

            if st.button("Visualize data"):
                if vis_selected_features and analysis_selected_features:
                    # Display actual data
                    st.session_state.all_data = display_economic_data(vis_selected_features,selected_ticker)  # Update session state
                     # Check if any data was found
                    if st.session_state.all_data.empty:
                        # Show a warning message and exit to avoid further code execution
                        st.warning("No data found for the selected parameters.")
                        return  # Stop further code execution
                    
                    st.write("Actual Data")
                    st.dataframe(st.session_state.all_data)

                    # Process the data for visualization
                    visalize_data_df = visualize_economic_data(analysis_selected_features,selected_ticker)
                    if visalize_data_df.empty:
                        # Show a warning message and exit to avoid further code execution
                        st.warning("No data found for the selected parameters.")
                        return  # Stop further code execution
                    preprocessed_df = economic_preprocess_data(visalize_data_df)
                   
                    # Normalize the data if the model and stock feature are selected
                    if selected_model != "None" and selected_ticker != "None":
                        st.session_state.df_final_normalized, st.session_state.df_final, st.session_state.scaler = select_and_normalize_eco_columns(
                            preprocessed_df, analysis_selected_features, selected_ticker, selected_duration
                        )

            # Optionally, display normalized data if it exists
            if st.session_state.df_final_normalized is not None:
                st.write("Normalized Data")
                st.dataframe(st.session_state.df_final_normalized)


        # Content for Tab 2
        with tab2:
            st.subheader("Pair Plot")
            if st.session_state.df_final_normalized is not None:
                graph = pair_plot(st.session_state.df_final_normalized)
                st.pyplot(graph.gcf(), clear_figure=True)  # Pass the current figure to st.pyplot
                download_button(graph, filename="pair_plot.png")
            else:
                st.warning("Please visualize the model first in Data Visualization Tab to view pair plot.")

        # Content for Tab 3
        with tab3:
            # If 'None' is not selected, load and process the data
            if 'None' not in analysis_selected_features and selected_duration:
                if st.button("Train model"):
                    # Split the data
                    # st.set_option('deprecation.showPyplotGlobalUse', False)
                    train_df, st.session_state.test_df = split_eco_data(st.session_state.df_final) #contains lag data as well

                    X_train, y_train, X_test, y_test = prepare_ml_data_eco_data(train_df, st.session_state.test_df, selected_ticker)

                    if selected_model == "Linear Regression Model":
                        # Train the model and store predictions
                        st.session_state.model, y_pred, st.session_state.mse, st.session_state.r2, st.session_state.intercept, st.session_state.coefficients = build_and_train_lr_model(X_train, y_train, X_test, y_test)
                        # First plot: Without removing outliers
                        st.session_state.fig = model_plot(X_test, y_test, y_pred, scaler=st.session_state.scaler,remove_outliers_flag=False)
                        st.pyplot(st.session_state.fig)
                        
                    elif selected_model == "Random Forest Regressor":
                        st.session_state.model, y_pred, st.session_state.mse, st.session_state.r2 = build_and_train_rf_model(X_train, y_train, X_test, y_test)
                        st.write("RF is trained")
                        selected_features = map_to_short_names(analysis_selected_features)
                        visualize_random_forest_results(st.session_state.model, X_test, y_test, y_pred, selected_features)

                    elif selected_model == "XGBoost Regressor":
                        st.session_state.model, y_pred, st.session_state.mse, st.session_state.r2 = build_and_train_xgb_model(X_train, y_train, X_test, y_test)
                        st.write("Xgboost is trained")
                        selected_features = map_to_short_names(analysis_selected_features)
                        visualize_xgboost_regression(st.session_state.model, X_test, y_test, y_pred, selected_features)

                    elif selected_model == "Decision Tree":
                        st.session_state.model, y_pred, st.session_state.mse, st.session_state.r2 = build_and_train_dt_model(X_train, y_train, X_test, y_test)
                        st.write("Decision Tree is trained")
                        selected_features = map_to_short_names(analysis_selected_features)
                        visualize_decision_tree_regression(st.session_state.model, X_test, y_test, y_pred, selected_features)

        
        # Content for Tab 4 (Plot predictions)
        with tab4:
            if st.session_state.model is not None and st.session_state.test_df is not None:
                inverse_transform_and_plot_predictions_eco_data(st.session_state.model, st.session_state.test_df, st.session_state.test_df[['date',selected_ticker]], st.session_state.scaler, selected_ticker)
            else:
                st.warning("Please train the model first in AI model Tab to view actual VS predicted results.")

        # Content for Tab 5 (Display metrics)
        with tab5:
            if (st.session_state.mse is not None or 
                st.session_state.r2 is not None or 
                st.session_state.intercept is not None or 
                st.session_state.coefficients is not None):
                
                st.write("### Model Performance")
                
                if st.session_state.mse is not None:
                    st.write(f"**Mean Squared Error (MSE):** {st.session_state.mse:.5f}")
                
                if st.session_state.r2 is not None:
                    st.write(f"**R² Score:** {st.session_state.r2:.5f}")
                
                if st.session_state.intercept is not None:
                    st.write(f"**Intercept:** {st.session_state.intercept:.5f}")
                
                if st.session_state.coefficients is not None:
                    # Assuming coefficients is a list, you might want to display all coefficients
                    st.write(f"**Coefficients:** {', '.join(f'{coef:.5f}' for coef in st.session_state.coefficients)}")


            else:
                st.warning("Please train the model first in AI model Tab to view the performance metrics.")


    elif current_tab == "Data Management":
        # Initialize session state for pagination and selected data
        if "selected_document_ids" not in st.session_state:
            st.session_state["selected_document_ids"] = []
        if "selected_rows" not in st.session_state:
            st.session_state["selected_rows"] = []
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = 1  # Start from page 1

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Features Analysis", "Summary", "Document Data"])

        # Function to get rows for the current page
        def get_page_data(df, page, rows_per_page=10):
            start_row = (page - 1) * rows_per_page
            end_row = page * rows_per_page
            return df.iloc[start_row:end_row]
        
        # Fetch unique years and quarters for filters
        unique_years, unique_quarters = get_unique_values()
        # Tab 1: Data Filtering & Row Selection
        with tab1:
            df = create_filtered_dataframe()

            # Sidebar filters
            year_filter = st.sidebar.multiselect("Select year(s)", options=unique_years, default=unique_years)
            quarter_filter = st.sidebar.multiselect("Select quarter(s)", options=unique_quarters, default=unique_quarters)
            ticker_search = st.sidebar.text_input("Search ticker", value="")
            security_name_search = st.sidebar.text_input("Search security name", value="")

            # Fetch filtered data based on filters
            filtered_df = filter_selected_col(
                years=year_filter,
                quarters=quarter_filter,
                ticker_search=ticker_search,
                security_name_search=security_name_search
            )
            # Paginate the filtered data and get rows for the current page
            page_data = get_page_data(filtered_df, st.session_state["current_page"])

            # Add 'select' column dynamically if it doesn't exist
            if "select" not in page_data.columns:
                page_data.insert(0, "select", False)

            # Let the user select rows
            edited_df = st.data_editor(
                page_data,
                column_config={
                    "select": st.column_config.CheckboxColumn("Select", help="Check to select rows")
                },
                disabled=[col for col in page_data.columns if col != "select"],  # Make other columns read-only
                hide_index=True,
                
            )

            # Capture the selected document_ids (store only the necessary data)
            selected_document_ids = edited_df[edited_df["select"] == True]["document_id"].tolist()

            # Store selected document_ids in session state
            if selected_document_ids != st.session_state["selected_document_ids"]:
                st.session_state["selected_document_ids"] = selected_document_ids

            # Automatically update the session state with selected rows
            st.session_state["selected_rows"] = edited_df[edited_df["select"] == True]

            # Pagination controls
            total_pages = (len(filtered_df) // 10) + (1 if len(filtered_df) % 10 != 0 else 0)
            st.write(f"Page {st.session_state['current_page']} of {total_pages}")

            # Navigation buttons
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.session_state["current_page"] > 1:
                    if st.button("Previous", key="previous"):
                        st.session_state["current_page"] -= 1
            with col3:
                if st.session_state["current_page"] < total_pages:
                    if st.button("Next", key="next"):
                        st.session_state["current_page"] += 1

       
        # Tab 2: Display Summary of Selected Rows
        with tab2:
            selected_rows = st.session_state["selected_rows"]
            
            if not selected_rows.empty:
                # Extract document_id of selected rows
                selected_document_ids = selected_rows["document_id"].tolist()
                
                # Loop through each document_id and display the summary
                for document_id in selected_document_ids:
                    summary_data = selected_row_summary(document_id)
                    
                    st.write(f"**Document ID**: {summary_data['document_id']}")
                    st.write(f"**Summary URL**: {summary_data['summary_url']}")
                    st.write(f"**Summary Content**: {summary_data['summary_content']}")
                    st.write("---")  # Separator
            else:
                st.write("No rows selected. Please select rows in Tab 1.")


        with tab3:
            selected_rows = st.session_state["selected_rows"]
            
            if not selected_rows.empty:
                # Extract document_id of selected rows
                selected_document_ids = selected_rows["document_id"].tolist()
                
                # Loop through each document_id and display the summary
                for document_id in selected_document_ids:
                    doc_data = selected_row_document(document_id)
                    
                    st.write(f"**Document ID**: {doc_data['document_id']}")
                    st.write(f"**Document URL**: {doc_data['document_url']}")
                    st.write(f"**Document Content**: {doc_data['document_content']}")
                    st.write("---")  # Separator
            else:
                st.write("No rows selected. Please select rows in Tab 1.")

         
    elif current_tab =="Work Flow":
        st.title("QrTrDr")

        with st.sidebar:
            st.header("List of Models For Summary")
            summ_models_options = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            default_summ_index = summ_models_options.index("gpt-4o-mini")
            selected_summary_model = st.selectbox(
                label="Choose your preferred model for summary:",
                options=["None"] + summ_models_options,
                index=default_summ_index,
                key="summary_model_selectbox"  # Unique key for this selectbox
            )
            
            st.header("List of Models For Financial Metrics")
            finan_models_options = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            default_index = finan_models_options.index("gpt-4o-mini")
            selected_finan_model = st.selectbox(
                label="Choose your preferred model for financial metrics:",
                options=["None"] + finan_models_options,
                index=default_index,
                key="financial_model_selectbox"  # Unique key for this selectbox
            )
            st.header("List of Models For Sentiment Scoring")
            senti_models_options = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            default_index = senti_models_options.index("gpt-4o-mini")
            selected_sent_model = st.selectbox(
                label="Choose your preferred model for sentiment scoring:",
                options=["None"] + senti_models_options,
                index=default_index,
                key="senti_model_selectbox"  # Unique key for this selectbox
            )

            st.header("Select input type")
            summary_option = st.radio(
            label="Choose input type for summary generation:",
            options=["None", "File", "Folder"],
            index=0
        )
    
        input_path = None  # To store the path of the uploaded file or folder
        
        if summary_option == "File":
            uploaded_file = st.file_uploader("Upload a file", type=["txt"])
            if uploaded_file:
                save_path = os.path.join("uploads", uploaded_file.name)  # Save in "uploads" directory
                os.makedirs("uploads", exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                input_path = save_path
                st.write(f"File `{uploaded_file.name}` has been saved to uploads.")
        
        elif summary_option == "Folder":
            uploaded_folder = st.file_uploader("Upload a folder (zipped)", type=["zip"])
            if uploaded_folder:
                save_directory = "saved_folders"
                folder_name = uploaded_folder.name.split(".")[0]
                destination_path = os.path.join(save_directory, folder_name)
                
                # Create the destination directory if it doesn't exist
                os.makedirs(save_directory, exist_ok=True)
                
                # Save the uploaded zip file
                zip_path = os.path.join(save_directory, uploaded_folder.name)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_folder.getbuffer())
                
                # Extract the zip file
                shutil.unpack_archive(zip_path, destination_path)
                
                # Recursively find all .txt files
                txt_files = []
                for root, dirs, files in os.walk(destination_path):
                    for file in files:
                        if file.endswith('.txt'):
                            txt_files.append(os.path.join(root, file))
                
                if txt_files:
                    input_path = destination_path
                    st.write(f"Folder `{folder_name}` has been successfully extracted to `{destination_path}`.")
                    st.write("The folder contains the following .txt files:")
                    for txt_file in txt_files:
                        st.write(f"- {os.path.relpath(txt_file, destination_path)}")
                else:
                    st.write(f"No .txt files found in the uploaded folder `{folder_name}`.")

        # Generate Summary Button
        if st.button("Generate Summary"):
            if (selected_summary_model and selected_finan_model and selected_sent_model != "None") and input_path:
                df_summary,financial_metrices_df=generate_summaries(input_path, selected_summary_model,selected_finan_model,selected_sent_model)
                st.dataframe(df_summary)
                st.dataframe(financial_metrices_df)
            else:
                st.error("Please select a model and upload a file or folder before generating the summary.")
        
        if st.button("Autogenerate Summary, financial metrics and Sentiment scores"):
            process_all_files()



if __name__ == "__main__":
    # Set a custom theme for the Streamlit app
    st.set_page_config(
        page_title="QrTrDr", page_icon=":chart_with_upwards_trend:",layout="wide",
        initial_sidebar_state="expanded"
    )
    main()
    