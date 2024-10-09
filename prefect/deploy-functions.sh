######
## Simple script for now to deploy functions
## Deploys all, which may not be necessary for unchanged resources
######

# Set up the project
gcloud config set project financial-pipeline-group-6

PROJECT_ID="financial-pipeline-group-6"
BUCKET_NAME="financial-pipeline-group-6-bucket"
SERVICE_ACCOUNT="financial-pipeline-group-6@appspot.gserviceaccount.com"
REGION="us-east1"
BASE_DIR="/home/jsale017/Predictive-Financial-Analytics-Financial-APIs/functions"
SCHEMA_SETUP_DIR="/home/jsale017/Predictive-Financial-Analytics-Financial-APIs/schema-setup"

# Deploy schema setup (since it's outside the functions folder)
echo "======================================================"
echo "Deploying the schema setup"
echo "======================================================"

gcloud functions deploy schema-setup \
    --gen2 \
    --runtime python311 \
    --trigger-http \
    --entry-point task \
    --source $SCHEMA_SETUP_DIR \
    --stage-bucket $BUCKET_NAME \
    --service-account $SERVICE_ACCOUNT \
    --region $REGION \
    --allow-unauthenticated \
    --memory 256MB 

# Deploy extract-rss (within the functions folder)
echo "======================================================"
echo "Deploying the rss extractor"
echo "======================================================"

gcloud functions deploy extract-rss \
    --gen2 \
    --runtime python311 \
    --trigger-http \
    --entry-point task \
    --source $BASE_DIR/extract-rss \
    --stage-bucket $BUCKET_NAME \
    --service-account $SERVICE_ACCOUNT \
    --region $REGION \
    --allow-unauthenticated \
    --memory 256MB 

# Deploy parse-rss (within the functions folder)
echo "======================================================"
echo "Deploying the rss parser"
echo "======================================================"

gcloud functions deploy parse-rss \
    --gen2 \
    --runtime python311 \
    --trigger-http \
    --entry-point task \
    --source $BASE_DIR/parse-rss \
    --stage-bucket $BUCKET_NAME \
    --service-account $SERVICE_ACCOUNT \
    --region $REGION \
    --allow-unauthenticated \
    --memory 512MB 

# Deploy load-rss (within the functions folder)
echo "======================================================"
echo "Deploying the loader"
echo "======================================================"

gcloud functions deploy load-rss \
    --gen2 \
    --runtime python311 \
    --trigger-http \
    --entry-point task \
    --source $BASE_DIR/load-rss \
    --stage-bucket $BUCKET_NAME \
    --service-account $SERVICE_ACCOUNT \
    --region $REGION \
    --allow-unauthenticated \
    --memory 512MB  \
    --timeout 60s
