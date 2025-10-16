# Azure Container Apps Deployment Guide

This guide walks you through deploying your RAG application to Azure Container Apps.

## Prerequisites

1. **Azure Account**: You need an active Azure subscription
   - Sign up at https://azure.microsoft.com/free/

2. **Azure CLI**: Install the Azure CLI
   ```bash
   # macOS
   brew install azure-cli

   # Windows
   # Download from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows

   # Linux
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

3. **Docker**: Ensure Docker is installed and running
   ```bash
   docker --version
   ```

4. **OpenAI API Key**: Get your API key from https://platform.openai.com/api-keys

## Architecture Overview

Your application will be deployed as multiple Container Apps:

- **Frontend**: React application served via Nginx
- **Backend**: FastAPI application
- **Databases**: PostgreSQL, PGVector, Neo4j, Elasticsearch, Redis

All services run in the same Azure Container Apps Environment with internal networking.

## Deployment Steps

**Important**: This guide uses a workflow where you build and test images locally before deploying to Azure. For a quick start, see [QUICKSTART.md](QUICKSTART.md). For detailed workflow information, see [DEPLOYMENT-WORKFLOW.md](DEPLOYMENT-WORKFLOW.md).

### Step 1: Build Production Images Locally

```bash
cd azure
./build-images.sh
```

This builds production-optimized Docker images on your local machine.

**Duration**: 2-5 minutes

### Step 2: Test Images Locally (Recommended)

```bash
./test-images-local.sh
```

This spins up all services using production images with local databases for testing.

- Frontend: http://localhost:3000
- Backend: http://localhost:8000

Test thoroughly, then stop:
```bash
docker-compose -f docker-compose.prod.yml down
```

### Step 3: Configure Azure Settings

Edit `push-images.sh` and `deploy.sh` to set your `ACR_NAME` (must be globally unique):

```bash
ACR_NAME="ragappyourname123"  # Must be globally unique across all Azure
```

**Important**: Use the same `ACR_NAME` in both files.

### Step 4: Login to Azure and Set API Key

```bash
az login
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Step 5: Push Images to Azure Container Registry

```bash
./push-images.sh
```

This script will:
1. Create Azure Container Registry (if it doesn't exist)
2. Tag your local images for ACR
3. Push verified images to ACR

**Duration**: 3-10 minutes

### Step 6: Deploy to Azure Container Apps

```bash
./deploy.sh
```

This script will:
1. Verify images exist in ACR
2. Create Azure Resource Group
3. Create Container Apps Environment
4. Deploy all database containers (Postgres, PGVector, Neo4j, Elasticsearch, Redis)
5. Deploy backend and frontend using pre-pushed images

**Duration**: 10-15 minutes

### Step 7: Access Your Application

After deployment completes, you'll see:

```
==========================================
Deployment completed successfully!
==========================================
Frontend URL: https://rag-frontend.xxx.azurecontainerapps.io
Backend URL: https://rag-backend.xxx.azurecontainerapps.io
==========================================
```

Open the Frontend URL in your browser to access your application.

## Updating Your Application

When you make code changes and want to redeploy:

### Quick Update (3 Commands)

```bash
cd azure
./build-images.sh      # Build new images locally
./push-images.sh       # Push to ACR
./update-images.sh     # Update deployment
```

### Thorough Update (With Testing)

```bash
cd azure
./build-images.sh                    # Build
./test-images-local.sh               # Test locally
docker-compose -f docker-compose.prod.yml down
./push-images.sh                     # Push verified images
./update-images.sh                   # Deploy
```

## Monitoring and Debugging

### View Application Logs

```bash
# Backend logs
az containerapp logs show \
  --name rag-backend \
  --resource-group rag-app-rg \
  --follow

# Frontend logs
az containerapp logs show \
  --name rag-frontend \
  --resource-group rag-app-rg \
  --follow
```

### View Container App Status

```bash
az containerapp show \
  --name rag-backend \
  --resource-group rag-app-rg \
  --query properties.runningStatus
```

### Scale Your Application

```bash
# Scale backend
az containerapp update \
  --name rag-backend \
  --resource-group rag-app-rg \
  --min-replicas 2 \
  --max-replicas 5
```

### Update Environment Variables

```bash
az containerapp update \
  --name rag-backend \
  --resource-group rag-app-rg \
  --set-env-vars "MAX_PERFORMANCE=true"
```

## Cost Management

### Estimated Monthly Costs

Based on minimal usage (1 replica per service):

- Container Apps Environment: ~$0
- Container Apps (7 apps × ~$0.000024/vCPU-s): ~$50-100/month
- Container Registry (Basic): ~$5/month
- Egress traffic: Variable based on usage

**Total**: Approximately $55-105/month for minimal usage

### Cost Optimization Tips

1. **Use Spot Instances**: For non-critical workloads
2. **Set Min Replicas to 0**: For development environments
3. **Use Managed Databases**: Consider Azure Database for PostgreSQL for production (better performance and automatic backups)
4. **Monitor Usage**: Use Azure Cost Management to track spending

### Stop All Services (Development)

To stop services without deleting them:

```bash
# Scale down to 0 replicas
az containerapp update --name rag-backend --resource-group rag-app-rg --min-replicas 0 --max-replicas 0
az containerapp update --name rag-frontend --resource-group rag-app-rg --min-replicas 0 --max-replicas 0
# Repeat for other services...
```

## Cleanup

To delete all Azure resources and stop incurring costs:

```bash
cd azure
./cleanup.sh
```

**Warning**: This permanently deletes all data and resources!

## Production Recommendations

For production deployments, consider these enhancements:

### 1. Use Azure Managed Databases

Instead of containerized databases, use:

- **Azure Database for PostgreSQL**: For better performance and automatic backups
- **Azure Cosmos DB**: Can replace Neo4j for graph operations
- **Azure Cache for Redis**: Managed Redis with high availability
- **Azure Cognitive Search**: Alternative to Elasticsearch

Update the connection strings in `deploy.sh` accordingly.

### 2. Add Custom Domain

```bash
# Add custom domain to frontend
az containerapp hostname add \
  --name rag-frontend \
  --resource-group rag-app-rg \
  --hostname yourdomain.com

# Configure DNS
# Add a CNAME record pointing to: rag-frontend.xxx.azurecontainerapps.io
```

### 3. Enable HTTPS

Container Apps automatically provide HTTPS. For custom domains:

```bash
az containerapp hostname bind \
  --name rag-frontend \
  --resource-group rag-app-rg \
  --hostname yourdomain.com \
  --environment rag-app-env \
  --validation-method CNAME
```

### 4. Set Up Application Insights

```bash
# Create Application Insights
az monitor app-insights component create \
  --app rag-app-insights \
  --location eastus \
  --resource-group rag-app-rg

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app rag-app-insights \
  --resource-group rag-app-rg \
  --query instrumentationKey \
  --output tsv)

# Update container app with instrumentation key
az containerapp update \
  --name rag-backend \
  --resource-group rag-app-rg \
  --set-env-vars "APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY"
```

### 5. Enable Autoscaling

```bash
az containerapp update \
  --name rag-backend \
  --resource-group rag-app-rg \
  --scale-rule-name http-scale \
  --scale-rule-type http \
  --scale-rule-http-concurrency 50
```

### 6. Set Up CI/CD with GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure Container Apps

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Login to Azure
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Build Images
        run: |
          cd azure
          ./build-images.sh

      - name: Push to ACR
        run: |
          cd azure
          ./push-images.sh
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Update Deployment
        run: |
          cd azure
          ./update-images.sh
```

## Troubleshooting

### Problem: Container Registry name already exists

**Solution**: Change `ACR_NAME` in `deploy.sh` to a unique value.

### Problem: Deployment fails with "QuotaExceeded"

**Solution**: Request quota increase or deploy to a different region.

### Problem: Backend can't connect to databases

**Solution**: Check that database services are running:
```bash
az containerapp list --resource-group rag-app-rg --output table
```

### Problem: Frontend shows "Cannot connect to backend"

**Solution**: Verify the `REACT_APP_BACKEND_URL` environment variable is set correctly.

## Additional Resources

- [Azure Container Apps Documentation](https://docs.microsoft.com/en-us/azure/container-apps/)
- [Azure Container Registry Documentation](https://docs.microsoft.com/en-us/azure/container-registry/)
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/)
- [Container Apps Pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/)

## Support

If you encounter issues:

1. Check the application logs using `az containerapp logs`
2. Verify all environment variables are set correctly
3. Ensure your Azure subscription has sufficient quota
4. Review the Azure Portal for any error messages

## Next Steps

After successful deployment:

1. Test your application thoroughly
2. Set up monitoring and alerts
3. Configure backup strategies for databases
4. Implement a CI/CD pipeline
5. Review and optimize costs
6. Set up a staging environment for testing
