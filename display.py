import streamlit as st
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
import pandas as pd
from model_agent import DataScienceAgent
from model_agent import AutoFeatureImplementer
import time

def render_data_analysis():
    """Render the data analysis section if data exists"""
    if st.session_state.data is not None:
        # Data Preview
        with st.expander("📊 Data Preview", expanded=False) as preview_expanded:
            st.dataframe(st.session_state.data.head())
            st.write("Dataset Shape:", st.session_state.data.shape)
            st.write("Columns:", ", ".join(st.session_state.data.columns))

        # Generate AI insights if not already present
        if st.session_state.data_analysis['ai_insights'] is None:
            with st.spinner("Analyzing data with AI..."):
                data_description = f"""
                Dataset Summary:
                - Columns: {', '.join(st.session_state.data.columns)}
                - Shape: {st.session_state.data.shape}
                - Data types: {st.session_state.data.dtypes.to_dict()}
                """
                try:
                    st.session_state.data_analysis['ai_insights'] = \
                        st.session_state.agent.gemini_assistant.analyze_data_context(data_description)
                except Exception as e:
                    st.error(f"Error generating AI insights: {str(e)}")
                    st.session_state.data_analysis['ai_insights'] = "Unable to generate AI insights."

        # Display AI insights
        st.markdown("### 🤖 AI Insights")
        st.markdown(
            f"<div class='ai-insight'>{st.session_state.data_analysis['ai_insights']}</div>",
            unsafe_allow_html=True
        )

def handle_file_upload():
    """Handle file upload and update session state"""
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format)",
        type=['csv'],
        key='file_uploader'
    )

    if uploaded_file is not None:
        # Only load new data if file has changed
        if st.session_state.last_file_name != uploaded_file.name:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.session_state.last_file_name = uploaded_file.name
                st.session_state.data_analysis['ai_insights'] = None  # Reset AI insights for new data
                st.session_state.model = None
                st.session_state.model_development['training_completed'] = False
                st.session_state.evaluation['results'] = None
                st.session_state.auto_feature_summary = {}
                st.rerun()  # Rerun to update the UI with new data
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

def initialize_session_state():
    if 'agent' not in st.session_state:
        st.session_state.agent = DataScienceAgent()
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data_analysis' not in st.session_state:
        st.session_state.data_analysis = {
            'preview_expanded': False,
            'ai_insights': None
        }
    if 'last_file_name' not in st.session_state:
        st.session_state.last_file_name = None
    if 'model_development' not in st.session_state:
        st.session_state.model_development = {
            'target_column': None,
            'feature_importance': None,
            'training_completed': False
        }
    if 'feature_engineering' not in st.session_state:
        st.session_state.feature_engineering = {
            'ai_suggestions': None
        }
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = {
            'results': None,
            'metrics': None,
            'interpretation': None
        }
    if 'auto_feature_summary' not in st.session_state:
        st.session_state.auto_feature_summary = {}

def display_sidebar():
    st.header("⚙️ User Settings")
    expertise_level = st.select_slider(
        "Select your expertise level:",
        options=["Beginner", "Intermediate", "Expert"],
        value="Intermediate"
    )

    st.header("🧠 AI Assistant")
    user_question = st.text_input("Ask me anything about your data:")
    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.agent.gemini_assistant.chat.send_message(user_question)
            st.markdown(f"**AI Response:**\n{response.text}")

    st.header("🎛️ Navigation")
    pages = ["Data Upload & Analysis", "Feature Engineering", "Model Development", "Model Evaluation"]
    current_page = st.radio("Select Page:", pages)
    return current_page

def display_data_upload_page():
    st.header("📈 Data Upload & Analysis")

    # Handle file upload
    handle_file_upload()

    # Show existing data analysis if data exists
    render_data_analysis()

    # Add a reset button if needed
    if st.session_state.data is not None:
        if st.button("Reset Data"):
            st.session_state.data = None
            st.session_state.last_file_name = None
            st.session_state.data_analysis['ai_insights'] = None
            st.session_state.feature_engineering['ai_suggestions'] = None
            st.session_state.model = None
            st.session_state.auto_feature_summary = {}
            st.session_state.current_results = None
            st.rerun()

def display_feature_engineering_page():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return

    st.header("🛠️ Feature Engineering")

    # Add AI feature suggestions
    with st.expander("🧠 Get AI Feature Engineering Suggestions"):
        if st.button("Generate Suggestions") or st.session_state.feature_engineering['ai_suggestions'] is not None:
            if st.session_state.feature_engineering['ai_suggestions'] is None:
                with st.spinner("Generating suggestions..."):
                    try:
                        suggestions = st.session_state.agent.get_feature_suggestions(st.session_state.data)
                        if suggestions:
                            st.session_state.feature_engineering['ai_suggestions'] = suggestions
                        else:
                            st.error("Unable to generate suggestions. Please try again.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

            if st.session_state.feature_engineering['ai_suggestions']:
                st.markdown(f"<div class='ai-insight'>{st.session_state.feature_engineering['ai_suggestions']}</div>",
                          unsafe_allow_html=True)

def display_model_development_page():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return

    st.header("🎯 Model Development")

    # Initialize session states
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'feature_implementer' not in st.session_state:
        st.session_state.feature_implementer = AutoFeatureImplementer()

    if (not st.session_state.auto_feature_summary and
        st.session_state.feature_engineering['ai_suggestions'] is None):
        with st.spinner("Generating feature engineering suggestions..."):
            try:
                suggestions = st.session_state.agent.get_feature_suggestions(st.session_state.data)
                st.session_state.feature_engineering['ai_suggestions'] = suggestions
            except Exception as e:
                st.error(f"Error generating suggestions: {str(e)}")

    with st.expander("🔍 Automatic Feature Engineering Process", expanded=True):
        if (not st.session_state.auto_feature_summary):
            with st.spinner("Implementing feature engineering..."):
                try:
                    transformed_data, feature_summary = st.session_state.feature_implementer.implement_suggestions(
                        st.session_state.data,
                        st.session_state.feature_engineering['ai_suggestions']
                    )

                    st.session_state.transformed_data = transformed_data
                    st.session_state.auto_feature_summary = feature_summary

                except Exception as e:
                    st.error(f"Error in feature engineering: {str(e)}")

        # Display feature engineering summary
        st.subheader("Feature Engineering Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Features", st.session_state.auto_feature_summary['original_features'])
        with col2:
            st.metric("New Features", st.session_state.auto_feature_summary['new_features'])
        with col3:
            st.metric("Features Added",
                     st.session_state.auto_feature_summary['new_features'] -
                     st.session_state.auto_feature_summary['original_features'])

        # Display transformation log
        st.subheader("Transformation Steps")
        for step in st.session_state.auto_feature_summary['transformation_log']:
            st.write(f"✓ {step}")

        # Preview transformed data
        st.subheader("Transformed Data Preview")
        st.dataframe(st.session_state.transformed_data.head())

    # Create tabs for different sections
    train_tab, history_tab = st.tabs(["Train Model", "Experiment History"])

    with train_tab:
        display_train_tab()

    with history_tab:
        display_history_tab()

def display_train_tab():
    col1, col2 = st.columns([2, 1])

    with col1:
        target_column = st.selectbox(
            "Select target variable",
            options=st.session_state.transformed_data.columns
        )
    with col2:
        experiment_name = st.text_input(
            "Experiment Name",
            value=f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    train_clicked = st.button("Train Model")

    # Display current results if they exist
    if st.session_state.current_results:
        display_current_results()

    if train_clicked:
        handle_model_training(target_column, experiment_name)

def display_history_tab():
    if st.session_state.experiment_history:
        st.subheader("Previous Experiments")

        experiment_options = {
            f"{exp['name']} | Target: {exp['target']} | Score: {exp['accuracy']:.4f}": idx
            for idx, exp in enumerate(st.session_state.experiment_history)
        }

        selected_exp_key = st.selectbox(
            "Select experiment to view results:",
            options=list(experiment_options.keys())
        )

        if selected_exp_key:
            display_experiment_details(experiment_options[selected_exp_key])
    else:
        st.info("No experiments run yet. Train a model to see results here.")

def display_model_evaluation_page():
    if st.session_state.model is None:
        st.warning("Please train a model first!")
    elif 'X_test' not in st.session_state or 'y_test' not in st.session_state:
        st.warning("Please retrain the model to generate evaluation data!")
    else:
        st.header("📊 Model Evaluation")

        if 'evaluation_history' not in st.session_state:
            st.session_state.evaluation_history = []

        current_tab, history_tab = st.tabs(["Current Evaluation", "Evaluation History"])

        with current_tab:
            display_current_evaluation_tab()

        with history_tab:
            display_evaluation_history_tab()

def display_current_evaluation_tab():
    # Show which model is being evaluated
    if st.session_state.current_results:
        st.info(f"Currently evaluating model: {st.session_state.current_results['name']} "
               f"(Target: {st.session_state.current_results['target']})")

        # Basic model metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", st.session_state.current_results['model_type'])
        with col2:
            st.metric("Features", st.session_state.current_results['feature_count'])
        with col3:
            st.metric("Base Accuracy", f"{st.session_state.current_results['accuracy']:.4f}")

    # Evaluation options
    evaluation_name = st.text_input(
        "Evaluation Name",
        value=f"Eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Evaluate button
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating model..."):
            try:
                # Make predictions
                y_pred = st.session_state.model.predict(st.session_state.X_test)

                # Calculate metrics based on problem type
                evaluation_results = {
                    'name': evaluation_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_name': st.session_state.current_results['name'],
                    'model_type': st.session_state.current_results['model_type'],
                    'target': st.session_state.current_results['target'],
                    'predictions': y_pred
                }

                if st.session_state.model.problem_type == 'classification':
                    evaluation_results.update({
                        'accuracy': accuracy_score(st.session_state.y_test, y_pred),
                        'precision': precision_score(st.session_state.y_test, y_pred, average='weighted'),
                        'recall': recall_score(st.session_state.y_test, y_pred, average='weighted'),
                        'f1': f1_score(st.session_state.y_test, y_pred, average='weighted'),
                        'confusion_matrix': confusion_matrix(st.session_state.y_test, y_pred).tolist()
                    })
                else:  # regression
                    evaluation_results.update({
                        'mse': mean_squared_error(st.session_state.y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_pred)),
                        'mae': mean_absolute_error(st.session_state.y_test, y_pred),
                        'r2': r2_score(st.session_state.y_test, y_pred)
                    })

                # Get AI interpretation
                results_desc = (f"Model: {evaluation_results['model_name']}\n"
                              f"Type: {evaluation_results['model_type']}\n")

                if st.session_state.model.problem_type == 'classification':
                    results_desc += (f"Accuracy: {evaluation_results['accuracy']:.2%}\n"
                                   f"Precision: {evaluation_results['precision']:.2%}\n"
                                   f"Recall: {evaluation_results['recall']:.2%}\n"
                                   f"F1 Score: {evaluation_results['f1']:.2%}")
                else:
                    results_desc += (f"MSE: {evaluation_results['mse']:.4f}\n"
                                   f"RMSE: {evaluation_results['rmse']:.4f}\n"
                                   f"MAE: {evaluation_results['mae']:.4f}\n"
                                   f"R² Score: {evaluation_results['r2']:.4f}")

                evaluation_results['interpretation'] = st.session_state.agent.gemini_assistant.explain_results(results_desc)

                # Add to history
                st.session_state.evaluation_history.append(evaluation_results)

                # Display results
                st.success("Evaluation complete!")

                # Display metrics
                st.subheader("📈 Model Metrics")
                if st.session_state.model.problem_type == 'classification':
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{evaluation_results['accuracy']:.2%}")
                    with col2:
                        st.metric("Precision", f"{evaluation_results['precision']:.2%}")
                    with col3:
                        st.metric("Recall", f"{evaluation_results['recall']:.2%}")
                    with col4:
                        st.metric("F1 Score", f"{evaluation_results['f1']:.2%}")

                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(evaluation_results['confusion_matrix'],
                              annot=True, fmt='d', ax=ax)
                    st.pyplot(fig)
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MSE", f"{evaluation_results['mse']:.4f}")
                    with col2:
                        st.metric("RMSE", f"{evaluation_results['rmse']:.4f}")
                    with col3:
                        st.metric("MAE", f"{evaluation_results['mae']:.4f}")
                    with col4:
                        st.metric("R² Score", f"{evaluation_results['r2']:.4f}")

                # Display AI interpretation
                st.subheader("🧠 AI Interpretation")
                st.markdown(f"<div class='ai-insight'>{evaluation_results['interpretation']}</div>",
                          unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during evaluation: {str(e)}")

def display_evaluation_history_tab():
    if st.session_state.evaluation_history:
        st.subheader("Previous Evaluations")

        # Create evaluation selection
        evaluation_options = {
            f"{eval['name']} | Model: {eval['model_name']} | {eval['timestamp']}": idx
            for idx, eval in enumerate(st.session_state.evaluation_history)
        }

        selected_eval_key = st.selectbox(
            "Select evaluation to view results:",
            options=list(evaluation_options.keys())
        )

        if selected_eval_key:
            selected_idx = evaluation_options[selected_eval_key]
            eval_result = st.session_state.evaluation_history[selected_idx]

            # Display evaluation details
            st.divider()
            st.subheader(f"Results: {eval_result['name']}")

            # Basic information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", eval_result['model_name'])
            with col2:
                st.metric("Model Type", eval_result['model_type'])
            with col3:
                st.metric("Target", eval_result['target'])

            # Display metrics based on problem type
            st.subheader("📈 Model Metrics")
            if 'accuracy' in eval_result:  # Classification
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{eval_result['accuracy']:.2%}")
                with col2:
                    st.metric("Precision", f"{eval_result['precision']:.2%}")
                with col3:
                    st.metric("Recall", f"{eval_result['recall']:.2%}")
                with col4:
                    st.metric("F1 Score", f"{eval_result['f1']:.2%}")

                # Display confusion matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(eval_result['confusion_matrix'],
                          annot=True, fmt='d', ax=ax)
                st.pyplot(fig)
            else:  # Regression
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{eval_result['mse']:.4f}")
                with col2:
                    st.metric("RMSE", f"{eval_result['rmse']:.4f}")
                with col3:
                    st.metric("MAE", f"{eval_result['mae']:.4f}")
                with col4:
                    st.metric("R² Score", f"{eval_result['r2']:.4f}")

            # Display AI interpretation
            st.subheader("🧠 AI Interpretation")
            st.markdown(f"<div class='ai-insight'>{eval_result['interpretation']}</div>",
                      unsafe_allow_html=True)
    else:
        st.info("No evaluations run yet. Run an evaluation to see results here.")

def handle_model_training(target_column, experiment_name):
    # Reset model_loaded flag when training new model
    st.session_state.model_loaded = False

    # Create progress containers
    progress_container = st.empty()
    status_container = st.empty()
    metric_container = st.container()

    try:
        with progress_container:
            progress_bar = st.progress(0)

        steps = [
            "Preparing data...",
            "Identifying features...",
            "Engineering features...",
            "Training models...",
            "Evaluating performance...",
            "Finalizing model..."
        ]

        for idx, step in enumerate(steps):
            status_container.info(step)
            progress_bar.progress((idx + 1) * (100 // len(steps)))
            time.sleep(0.5)

            if idx == 2:
                with metric_container:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Features Processed", f"{len(st.session_state.data.columns)} columns")
                    with col2:
                        st.metric("Rows Analyzed", f"{len(st.session_state.data)} rows")
                    with col3:
                        st.metric("Missing Values", f"{st.session_state.data.isnull().sum().sum()} total")

        # Fresh instance for each training
        model_agent = DataScienceAgent()

        # Create a copy of the data
        data_copy = st.session_state.transformed_data.copy()

        # Process data and train model
        model = model_agent.process_data(data_copy, target_column)

        # Get feature importance if available
        feature_importance = None
        try:
            if hasattr(model.best_model.named_steps['model'], 'feature_importances_'):
                feature_names = data_copy.drop(columns=[target_column]).columns
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.best_model.named_steps['model'].feature_importances_
                }).sort_values('importance', ascending=False)
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            feature_importance = None

        # Create results dictionary
        current_results = {
            'name': experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target': target_column,
            'model_type': type(model.best_model.named_steps['model']).__name__,
            'problem_type': model.problem_type,
            'feature_count': len(data_copy.drop(columns=[target_column]).columns),
            'accuracy': model.best_model.score(model.X_test, model.y_test),
            'model': model,
            'X_test': model.X_test,
            'y_test': model.y_test,
            'feature_importance': feature_importance
        }

        # Update current results and add to history
        st.session_state.current_results = current_results
        st.session_state.experiment_history.append(current_results)

        # Update current model and test data
        st.session_state.model = model
        st.session_state.X_test = model.X_test
        st.session_state.y_test = model.y_test

        # Clear progress indicators
        progress_container.empty()
        status_container.success("Model training complete!")
        st.rerun()  # Rerun to update the UI with new results

    except Exception as e:
        progress_container.empty()
        status_container.error(f"An error occurred: {str(e)}")

def display_current_results():
    st.divider()
    if st.session_state.model_loaded:
        st.info("Currently viewing model loaded from history")
    st.subheader("Current Model Results")

    # Display basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", st.session_state.current_results['model_type'])
    with col2:
        st.metric("Target Variable", st.session_state.current_results['target'])
    with col3:
        st.metric("Accuracy/Score", f"{st.session_state.current_results['accuracy']:.4f}")

    # Display feature importance if available
    if ('feature_importance' in st.session_state.current_results and
        st.session_state.current_results['feature_importance'] is not None and
        not st.session_state.current_results['feature_importance'].empty):
        st.subheader("Feature Importance")
        feature_importance_df = st.session_state.current_results['feature_importance']
        st.bar_chart(feature_importance_df.set_index('feature'))
    else:
        st.info("Feature importance visualization not available for this model type.")

def display_experiment_details(selected_idx):
    exp = st.session_state.experiment_history[selected_idx]

    # Display experiment details
    st.divider()
    st.subheader(f"Results: {exp['name']}")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", exp['model_type'])
    with col2:
        st.metric("Target Variable", exp['target'])
    with col3:
        st.metric("Accuracy/Score", f"{exp['accuracy']:.4f}")

    # Feature importance
    if ('feature_importance' in exp and
        exp['feature_importance'] is not None and
        not exp['feature_importance'].empty):
        st.subheader("Feature Importance")
        st.bar_chart(exp['feature_importance'].set_index('feature'))
    else:
        st.info("Feature importance visualization not available for this model type.")

    # Load model button
    if st.button("Load This Model"):
        st.session_state.current_results = exp
        st.session_state.model = exp['model']
        st.session_state.X_test = exp['X_test']
        st.session_state.y_test = exp['y_test']
        st.session_state.model_loaded = True  # Set flag when loading model
        st.success(f"Loaded model from experiment: {exp['name']}")
        st.rerun()  # Force a rerun to update both tabs