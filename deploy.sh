#!/bin/bash
set -e

# RAG System Deployment Script
# Builds and deploys backend and frontend to Azure Container Apps

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RAG System - Azure Deployment                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}\n"

# Configuration
RESOURCE_GROUP="rag-dev-rg"
ACR_NAME="ragdevacrfxbt87w3"
LOCATION="eastus"

# Check for arguments
if [ "$#" -lt 1 ]; then
    echo -e "${YELLOW}Usage: $0 <postgres-password> [openai-api-key]${NC}"
    echo ""
    echo "Arguments:"
    echo "  postgres-password  : PostgreSQL admin password (from infrastructure deployment)"
    echo "  openai-api-key     : OpenAI API key (optional, defaults to placeholder)"
    echo ""
    echo "Example:"
    echo "  $0 'MySecurePassword123!' 'sk-...'"
    exit 1
fi

POSTGRES_PASSWORD="$1"
OPENAI_API_KEY="${2:-sk-placeholder-user-will-bring-own-key-via-frontend}"

# Check if logged into Azure
if ! az account show &> /dev/null; then
    echo -e "${RED}Not logged into Azure. Please run 'az login' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration loaded${NC}"

# Login to ACR
echo -e "\n${YELLOW}[1/5] Logging into Azure Container Registry...${NC}"
az acr login --name ${ACR_NAME}
echo -e "${GREEN}✓ Logged into ACR${NC}"

# Build and push backend
echo -e "\n${YELLOW}[2/5] Building and pushing backend image...${NC}"
echo "  Building backend..."
cd backend
docker build -t ${ACR_NAME}.azurecr.io/rag-backend:latest . --quiet
echo "  Pushing backend..."
docker push ${ACR_NAME}.azurecr.io/rag-backend:latest --quiet
cd ..
echo -e "${GREEN}✓ Backend image built and pushed${NC}"

# Get backend URL first (we'll use the predictable URL)
BACKEND_URL="https://rag-backend.purpledesert-cefd71e5.eastus.azurecontainerapps.io"

# Build and push frontend with backend URL
echo -e "\n${YELLOW}[3/5] Building and pushing frontend image...${NC}"
echo "  Building frontend with backend URL: ${BACKEND_URL}"
cd frontend
docker build -f Dockerfile.prod \
  --build-arg REACT_APP_BACKEND_URL="${BACKEND_URL}" \
  -t ${ACR_NAME}.azurecr.io/rag-frontend:latest . --quiet
echo "  Pushing frontend..."
docker push ${ACR_NAME}.azurecr.io/rag-frontend:latest --quiet
cd ..
echo -e "${GREEN}✓ Frontend image built and pushed${NC}"

# Deploy using Bicep
echo -e "\n${YELLOW}[4/5] Deploying container apps with Bicep...${NC}"
DEPLOYMENT_NAME="rag-deployment-$(date +%Y%m%d-%H%M%S)"

az deployment group create \
  --resource-group ${RESOURCE_GROUP} \
  --template-file azure/deploy-apps.bicep \
  --parameters \
    environment=dev \
    location=${LOCATION} \
    backendImage="${ACR_NAME}.azurecr.io/rag-backend:latest" \
    frontendImage="${ACR_NAME}.azurecr.io/rag-frontend:latest" \
    postgresAdminPassword="${POSTGRES_PASSWORD}" \
    openAIApiKey="${OPENAI_API_KEY}" \
  --name ${DEPLOYMENT_NAME} \
  --output json > deployment-output.json

echo -e "${GREEN}✓ Deployment complete${NC}"

# Get deployment outputs
echo -e "\n${YELLOW}[5/5] Getting deployment information...${NC}"
BACKEND_URL=$(cat deployment-output.json | jq -r '.properties.outputs.backendUrl.value')
FRONTEND_URL=$(cat deployment-output.json | jq -r '.properties.outputs.frontendUrl.value')
CORS_ORIGINS=$(cat deployment-output.json | jq -r '.properties.outputs.corsOrigins.value')

# Save URLs to file
cat > deployment-urls.txt <<EOF
Deployment Time: $(date)
Backend URL:  ${BACKEND_URL}
Frontend URL: ${FRONTEND_URL}
CORS Origins: ${CORS_ORIGINS}
EOF

echo -e "${GREEN}✓ Deployment information saved to deployment-urls.txt${NC}"

# Summary
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Deployment Successful!                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${YELLOW}Application URLs:${NC}"
echo -e "  Frontend: ${GREEN}${FRONTEND_URL}${NC}"
echo -e "  Backend:  ${GREEN}${BACKEND_URL}${NC}"

echo -e "\n${YELLOW}CORS Configuration:${NC}"
echo -e "  ${GREEN}✓ Automatic CORS enabled${NC}"
echo -e "  ${GREEN}✓ Frontend origin allowed${NC}"
echo -e "  ${GREEN}✓ localhost:3000 allowed (development)${NC}"

echo -e "\n${YELLOW}Features Enabled:${NC}"
echo "  ✓ Vector Search (PGVector)"
echo "  ✓ Memory Management"
echo "  ✗ Graph Search (disabled for performance)"
echo "  ✗ Elasticsearch Filter Search (disabled for performance)"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "  1. Open ${FRONTEND_URL} in your browser"
echo "  2. Upload a PDF document"
echo "  3. Ask questions using vector search"

echo -e "\n${GREEN}Deployment complete! 🚀${NC}\n"
