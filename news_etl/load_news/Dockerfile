FROM python:3.9-slim

WORKDIR /app

COPY load_news_main.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "load_news_main.py"]
