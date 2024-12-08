# <img src = "https://sigmoid-image.s3.amazonaws.com/wp-content/uploads/2022/02/22112809/Build-a-Winning-Data-Pipeline-Architecture-on-the-Cloud-for-CPG-1.gif" alt = "Moving Header" width="1100px">

# BA882: Predictive End-to-End Analytics Pipeline for Financial APIs Group 6
## Project Overview: Predictive Financial Analytics APIs

The Predictive Financial Analytics APIs project is a comprehensive end-to-end solution designed to provide actionable insights into daily stock data for selected technology companies. This pipeline integrates multiple components to deliver a robust financial analysis toolset:

#### **Pipeline Features**:
1. **Daily Stock Data Acquisition**:  
   The pipeline automatically extracts, processes, and updates daily stock data for key technology companies (e.g., Apple, Microsoft, Nvidia, Amazon, Netflix). 

2. **News Integration and Sentiment Analysis**:  
   - Aggregates the latest news articles related to the selected companies.
   - Utilizes sentiment analysis to score news data, providing insights into market sentiment for each company.

3. **Predictive Modeling**:  
   - Implements traditional machine learning models like **XGBoost Regressor** and **Random Forest** for reliable predictions.
   - Employs advanced neural network models, including **LSTM** (Long Short-Term Memory) and **RNN** (Recurrent Neural Networks), to predict key financial metrics such as:
     - **Closing Prices**: Forecasts daily stock closing prices.
     - **Trade Volumes**: Predicts trade volumes with high accuracy.

4. **Interactive User Interface**:  
   A **Streamlit-based application** serves as the user interface, offering:
   - **BigQuery Integration**: Enables users to text-to-sql query real-time and historical data from BigQuery.
   - **Large Language Model (LLM) Integration**: Allows users to interact with financial data via natural language queries.
   - **News Insights**: Displays the latest news articles alongside sentiment scores for easy interpretation.

5. **10-K Document Analysis**:  
   - Users can upload **10-K reports** of the selected companies directly into the interface.
   - An integrated **LLM-based chatbot** analyzes the uploaded documents, enabling users to ask specific questions about the financial conditions and performance metrics of each company.

#### **Key Highlights**:
- **Automation**: Fully automated data pipeline orchestrates stock data extraction, transformation, and loading (ETL), coupled with news updates.
- **Machine Learning and AI**: Combines traditional and neural network-based predictive models for enhanced accuracy.
- **User-Centric Design**: The Streamlit application ensures accessibility, making it easy for users to obtain predictions, analyze sentiment, and engage with financial reports interactively.
- **Comprehensive Insights**: By integrating financial metrics, predictive analytics, sentiment analysis, and document review, the project delivers a holistic view of each company’s financial health and market outlook.

## **Sources and Tools Utilized in the Project**

#### **Data Source**  
- **Alpha Vantage**  
  - Website: [Alpha Vantage](https://www.alphavantage.co/#page-top)  
  - Documentation: [API Documentation](https://www.alphavantage.co/documentation/)  

#### **Technology Stocks Used**  
1. Apple (AAPL)  
2. Microsoft (MSFT)  
3. Nvidia (NVDA)  
4. Netflix (NFLX)  
5. Amazon (AMZN)  

#### **Tools**  
1. Google Cloud  
2. Prefect Orchestration & Prefect Cloud  
3. Google BigQuery  
4. Flask  
5. Google Secret Manager  
6. Pinecone  
7. Langchain  
8. Vertex AI  
9. Streamlit

## Streamlit User-Interface: 
   - https://streamlit-genai-apps-676257416424.us-central1.run.app

## Looker Visualization: 
   - https://lookerstudio.google.com/s/kDvwmHdQ76g
