# <img src = "https://sigmoid-image.s3.amazonaws.com/wp-content/uploads/2022/02/22112809/Build-a-Winning-Data-Pipeline-Architecture-on-the-Cloud-for-CPG-1.gif" alt = "Moving Header" width="1100px">

# BA882: Predictive End-to-End Analytics Pipeline for Financial APIs Group 6
## Project Overview

### Data Feeds Utilized in the Project
- Alpha Vantage is a website that offers real-time and historical financial market data through APIs and spreadsheets. The API provides data on asset classes such as stocks, ETFs, mutual funds, and foreign commodities. Additionally, it covers pre-market and post-market hours.
  - Website Link: https://www.alphavantage.co/#page-top
  - Documentation of API: https://www.alphavantage.co/documentation/

### Data Frequency 
- Alpha Vantage data are updated every second, allowing users to access real-time US market data.
  For the purpose of our analysis we extract data at the market close time to obtain a comprehensive overview of the daily movement of our tech stocks.

Technology Stocks Used during the Analysis
1. Apple (AAPL)
2. Microsoft (MSFT)
3. Nvidia (NVDA)
4. Netflix (NFLX)
5. Amazon (AMZN)

### Looking Ahead:
  - Our team plans to develop two regression models, one utilizing traditional Machine Learning Algorithms and another one developing a Neural Network (Recurrent Neural Network) to predict closing prices per stock and trading volumes.
  - Aside from this, we plan on integrating a Retrieval Augmented Generative AI Framework that allows potential investors to ask questions about the year end reports from the above companies gaining a comprehensive understanding of the current state of the companies for potential investment opportunities. 

### Tools utilized:
1. Google Cloud
2. Prefect Orchestration & Prefect Cloud
3. Google BigQuery
4. Flask
5. Google Secret Manager

### Pipeline Flow

```mermaid
graph TD;
    A[Extract Data] --> B[Parse Data];
    B --> C[Loading Data into BigQuery];
    C --> D[Orchestration with Prefect];



