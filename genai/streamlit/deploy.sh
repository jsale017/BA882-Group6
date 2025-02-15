
# setup the project
gcloud config set project finnhub-pipeline-ba882

echo "======================================================"
echo "build (no cache)"
echo "======================================================"

docker build --no-cache -t gcr.io/finnhub-pipeline-ba882/streamlit-genai-apps .

echo "======================================================"
echo "push"
echo "======================================================"

docker push gcr.io/finnhub-pipeline-ba882/streamlit-genai-apps

echo "======================================================"
echo "deploy run"
echo "======================================================"


gcloud run deploy streamlit-genai-apps \
    --image gcr.io/finnhub-pipeline-ba882/streamlit-genai-apps \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account 676257416424-compute@developer.gserviceaccount.com \
    --memory 1Gi