# BA882-Group6

BA882 - Deploying Analytics Pipelines 

1. What data feed(s) are you considering for project?

The Data Feeds that we are considering for our project are the following: 
- FinHub is a website that offers democratized financial data, providing access to real-time stocks, currencies, and cryptocurrencies through an API. FinHub has a limit of 30 API calls per second, ensuring that we will have continuous access to real-time stock data throughout the entire project.
a.	Website Link: https://finnhub.io/dashboard
b.	Documentation of API: https://finnhub.io/docs/api/introduction 
c.	Github: https://github.com/Finnhub-Stock-API/finnhub-python
- Alpha Vantage is a website that offers real-time and historical financial market data through APIs and spreadsheets. The API provides data on asset classes such as stocks, ETFs, mutual funds, and foreign commodities. Additionally, it covers pre-market and post-market hours.
d.	Website Link: https://www.alphavantage.co/#page-top
e.	Documentation of API: https://www.alphavantage.co/documentation/

2. How often do the data get updated (i.e. new records are available and need to be added to your data warehouse)

- FinHub and Alpha Vantage data are updated every second, allowing users to access real-time US market data.

3. Why are you interested in exploring these data feed(s) for your project?

- We are developing an end-to-end pipeline for financial market data, driven by our passion for pursuing careers in the financial services industry after completing our Master's program. This project will not only help us automate processes but also allow us to apply advanced deep learning models, such as RNNs, to adapt to new financial information continuously. By leveraging machine learning techniques from previous courses, we aim to build a robust system that provides a comprehensive understanding of current U.S. market conditions with updated data. 

  Using Alpha Vantage's data feeds, which offer real-time and historical data for stocks, and cryptocurrencies, as well as a wide range of technical indicators, we will perform detailed market analysis. This project aligns perfectly with our career goals and provides an excellent opportunity to implement pipelines that match our goals.

4. What are the business problem(s) you will be exploring once your data are being collected?

- The business problem we are addressing focuses on improving decision-making for financial advisors. This will enable them to better serve their customers by providing timely access to market data. By designing an efficient, end-to-end pipeline for financial data analysis, financial institutions can reduce investment risk, improve responsiveness, and gain a competitive advantage. This automated process will allow advisors to make faster, data-driven decisions to optimize their portfolios.

Key business challenges we aim to explore include analyzing specific portfolios based on risk/return metrics, evaluating industries, assessing volatility and other risk factors, and interpreting events such as major corporate actions. We also aim to predict market trends, including future volume and price movements, which will assist investors in making informed buy/sell decisions.

Once data is collected from sources like the Alpha Vantage API, we will focus on optimizing pipeline efficiency. We will leverage machine learning models to forecast stock prices and market trends based on real-time and historical data. Furthermore, we plan to filter and segment data into meaningful feature clusters, allowing financial advisors to formulate more accurate investment strategies.


