import streamlit as st
from display import (
    initialize_session_state,
    display_sidebar,
    display_data_upload_page,
    display_feature_engineering_page,
    display_model_development_page,
    display_model_evaluation_page
)

def main():
    # Set page config
    st.set_page_config(page_title="DataScientist.AI", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .stProgress > div > div > div {
            background-color: #1f77b4;
        }
        .ai-insight {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Main title
    st.markdown("<h1 style='color: #0066CC;'>🔬DataScientist.AI: Intelligent Data Science Assistant💡</h1>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        current_page = display_sidebar()

    # Main content area based on selected page
    if current_page == "Data Upload & Analysis":
        display_data_upload_page()
    elif current_page == "Feature Engineering":
        display_feature_engineering_page()
    elif current_page == "Model Development":
        display_model_development_page()
    elif current_page == "Model Evaluation":
        display_model_evaluation_page()

    # Add chat history display at the bottom
    if st.checkbox("Show Chat History"):
        st.header("💬 AI Chat History")
        for message in st.session_state.agent.gemini_assistant.chat.history:
            role = "🤖 AI" if message.role == "model" else "👤 You"
            st.markdown(f"**{role}:** {message.parts[0].text}")

if __name__ == "__main__":
    main()
