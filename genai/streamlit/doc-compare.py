# imports
import streamlit as st
import PyPDF2
import io
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#supported-models
model = GenerativeModel("gemini-1.5-pro-001")

############################################## project setup
GCP_PROJECT = 'finnhub-pipeline-ba882'
GCP_REGION = "us-central1"

vertexai.init(project=GCP_PROJECT, location=GCP_REGION)

# Streamlit setup
st.image("https://cdn.mos.cms.futurecdn.net/EFRicr9v8AhNXhw2iW3qfJ-1600-80.jpg.webp")
st.title("Financial Risk Analysis")
st.markdown("Analyze financial documents for key risks, opportunities, and overall sentiment.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to analyze financial risks and sentiment using Vertex AI
def analyze_financial_risks(text):
    try:
        prompt = f"""
        Analyze the following financial document for:
        
        1. Key financial risks mentioned (e.g., market risk, credit risk, operational risk).
        2. Opportunities or strengths highlighted in the document.
        3. The overall sentiment of the document (positive, neutral, or negative).
        4. Any actionable recommendations based on the content.

        Financial Document:
        {text[:2000]}  # Limiting text length to avoid token limits
        
        Provide the response in a clear, structured format.
        """
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        st.error(f"Error in Vertex AI analysis: {str(e)}")
        return None

# File uploader
file = st.sidebar.file_uploader("Upload a Financial Document (PDF)", type=['pdf'])

if file:
    st.success("PDF uploaded successfully! Click the button below to start the analysis.")
    with st.spinner("Processing PDF..."):
        # Extract text from PDF
        text = extract_text_from_pdf(file)
        
    if text:
        if st.button("Analyze Financial Risks"):
            with st.spinner("Analyzing document..."):
                analysis_result = analyze_financial_risks(text)
                
                if analysis_result:
                    st.subheader("Analysis Results")
                    st.markdown(analysis_result)
                    
                    # Option to download the analysis
                    st.download_button(
                        label="Download Analysis Results",
                        data=analysis_result,
                        file_name="financial_risk_analysis.txt",
                        mime="text/plain"
                    )

                    # Show the prompt used
                    st.markdown("""

                    ### The Prompt Used for Analysis:
                    
                    Analyze the following financial document for:
                    
                    1. Key financial risks mentioned (e.g., market risk, credit risk, operational risk).
                    2. Opportunities or strengths highlighted in the document.
                    3. The overall sentiment of the document (positive, neutral, or negative).
                    4. Any actionable recommendations based on the content.
                    
                    Financial Document:
                    {text[:2000]}  # Limiting text length to avoid token limits
                    
                    Provide the response in a clear, structured format.
                    """)
    else:
        st.error("Error processing the PDF. Please try again.")
