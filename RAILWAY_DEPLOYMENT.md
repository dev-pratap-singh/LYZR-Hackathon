# 🚂 Railway Full-Stack Deployment Guide

Complete guide to deploy both backend and frontend on Railway.

---

## 📋 Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Railway CLI**: Install globally
   ```bash
   npm i -g @railway/cli
   ```
3. **Git Repository**: Code committed and pushed to GitHub
4. **OpenAI API Key**: Optional (users can provide their own via UI)

---

## 🚀 Quick Deploy (Automated)

Use the provided deployment script:

```bash
# Make script executable (if not already)
chmod +x deploy-railway.sh

# Run deployment script
./deploy-railway.sh
```

The script will guide you through:
1. Backend deployment
2. Frontend deployment
3. Environment variable configuration
4. CORS setup

---

## 📝 Manual Deployment (Step-by-Step)

### Part 1: Deploy Backend

#### Step 1: Login to Railway

```bash
railway login
```

This will open a browser window for authentication.

#### Step 2: Create New Project

```bash
# Initialize Railway project in root directory
railway init
```

Select "Create new project" and give it a name (e.g., "rag-system-backend").

#### Step 3: Link to Service

```bash
# Add a new service
railway add
```

Select "Empty Service" and name it "backend".

#### Step 4: Set Environment Variables

```bash
# Essential variables
railway variables set POSTGRES_PASSWORD="your_secure_password"
railway variables set NEO4J_AUTH="neo4j/your_secure_password"
railway variables set OPENAI_API_KEY="sk-proj-your-key"  # Optional

# Performance settings
railway variables set MAX_PERFORMANCE="false"
railway variables set GRAPHRAG_ENABLED="true"
railway variables set GRAPHRAG_ENABLE_MULTIPASS="true"

# Search settings
railway variables set ENABLE_VECTOR_SEARCH="true"
railway variables set ENABLE_GRAPH_SEARCH="true"
railway variables set ENABLE_FILTER_SEARCH="true"

# RAG settings
railway variables set CHUNK_SIZE="1200"
railway variables set CHUNK_OVERLAP="500"
railway variables set TOP_K_RESULTS="20"

# Memory settings
railway variables set MEMORY_ENABLED="true"
```

#### Step 5: Deploy Backend

```bash
# Deploy using Dockerfile
railway up
```

This will:
- Build the Docker image
- Deploy all 5 services (PostgreSQL, PGVector, Neo4j, Elasticsearch, Redis)
- Start the FastAPI backend

#### Step 6: Generate Backend URL

```bash
# Generate public domain
railway domain
```

Save this URL! You'll need it for the frontend.

Example: `rag-backend-production.up.railway.app`

---

### Part 2: Deploy Frontend

#### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

#### Step 2: Initialize Railway for Frontend

```bash
# Create new service for frontend
railway init
```

Select "Create new service" or link to existing project and add new service.

#### Step 3: Set Frontend Environment Variables

Replace `YOUR_BACKEND_URL` with the backend URL from Part 1, Step 6:

```bash
railway variables set REACT_APP_API_URL="https://YOUR_BACKEND_URL"
```

Example:
```bash
railway variables set REACT_APP_API_URL="https://rag-backend-production.up.railway.app"
```

#### Step 4: Deploy Frontend

```bash
# Deploy frontend
railway up
```

This will:
- Build React app with production optimizations
- Create optimized nginx container
- Deploy to Railway

#### Step 5: Generate Frontend URL

```bash
# Generate public domain for frontend
railway domain
```

Example: `rag-frontend-production.up.railway.app`

---

### Part 3: Update Backend CORS

#### Step 1: Go Back to Root Directory

```bash
cd ..
```

#### Step 2: Link to Backend Service

```bash
# Link to backend service
railway link
```

Select your backend service.

#### Step 3: Update CORS Settings

Replace `YOUR_FRONTEND_URL` with the frontend URL from Part 2, Step 5:

```bash
railway variables set ALLOWED_ORIGINS="https://YOUR_FRONTEND_URL,http://localhost:3000"
```

Example:
```bash
railway variables set ALLOWED_ORIGINS="https://rag-frontend-production.up.railway.app,http://localhost:3000"
```

#### Step 4: Redeploy Backend (if needed)

Railway will automatically redeploy when environment variables change. You can verify with:

```bash
railway logs
```

---

## ✅ Post-Deployment Verification

### 1. Check Backend Health

```bash
curl https://YOUR_BACKEND_URL/health
```

Expected response:
```json
{
  "status": "healthy",
  "storage_path": "/app/storage",
  "openai_configured": true
}
```

### 2. Check Frontend

Visit your frontend URL in a browser:
```
https://YOUR_FRONTEND_URL
```

You should see the RAG System interface.

### 3. Test Complete Flow

1. **Configure API Key** (if not set in backend):
   - Click Settings (⚙️) in top-right
   - Enter your OpenAI API key
   - Click Save

2. **Upload Document**:
   - Drag and drop a PDF
   - Wait for processing to complete

3. **Query**:
   - Ask a question about the document
   - Verify you get a response

4. **Check Graph**:
   - Click "🕸️ Show Knowledge Graph"
   - Verify graph visualization loads

---

## 🔍 Troubleshooting

### Backend won't start

**Check logs:**
```bash
railway logs
```

**Common issues:**
- Database passwords not set
- Insufficient memory (upgrade Railway plan)
- Docker build failed (check Dockerfile)

### Frontend shows blank page

**Check browser console** for errors.

**Common issues:**
- `REACT_APP_API_URL` not set correctly
- CORS errors (update `ALLOWED_ORIGINS`)
- Build failed (check Railway logs)

### CORS Errors

**Symptom:** Browser console shows CORS policy errors.

**Fix:**
```bash
# Make sure ALLOWED_ORIGINS includes your frontend URL
cd /path/to/project/root
railway link  # Link to backend service
railway variables set ALLOWED_ORIGINS="https://your-frontend-url,http://localhost:3000"
```

### Database Connection Errors

**Symptom:** Backend logs show database connection failures.

**Fix:**
1. Check that all services are running:
   ```bash
   railway logs
   ```

2. Verify environment variables are set:
   ```bash
   railway variables
   ```

3. Restart services:
   ```bash
   railway restart
   ```

### Frontend Can't Fetch from Backend

**Symptom:** Network errors in browser console.

**Fix:**
1. Verify backend URL is accessible:
   ```bash
   curl https://YOUR_BACKEND_URL/health
   ```

2. Check `REACT_APP_API_URL` is set correctly:
   ```bash
   cd frontend
   railway variables
   ```

3. Verify CORS is configured:
   ```bash
   cd ..
   railway variables | grep ALLOWED_ORIGINS
   ```

---

## 💰 Cost Estimation

### Railway Pricing

- **Hobby Plan**: $5/month
  - $5 free credit
  - Good for testing
  
- **Pro Plan**: $20/month (Recommended)
  - Includes $20 free credit
  - Better performance
  - More resources

### Resource Usage

**Backend** (with 5 databases):
- ~1-2 GB RAM
- ~$10-15/month

**Frontend** (nginx):
- ~100-200 MB RAM
- ~$2-5/month

**Total Estimated Cost**: $12-20/month

### Cost Optimization Tips

1. **Use User API Keys**: Don't set `OPENAI_API_KEY` in environment
2. **Disable MAX_PERFORMANCE**: Set to `false` to reduce API calls
3. **Enable Memory**: Reduce redundant queries with caching
4. **Monitor Usage**: Check Railway dashboard regularly

---

## 🔄 Updating Your Deployment

### Option 1: Automatic Deployment (Recommended)

Connect Railway to GitHub for automatic deployments:

1. Go to Railway dashboard
2. Click on your service
3. Settings → Connect to GitHub
4. Select repository and branch
5. Enable "Auto-deploy on push"

Now every `git push` will trigger a deployment!

### Option 2: Manual Deployment

```bash
# After making changes
git add .
git commit -m "Your changes"
git push

# Deploy backend
cd /path/to/project/root
railway up

# Deploy frontend
cd frontend
railway up
```

---

## 📊 Monitoring

### View Logs

```bash
# Real-time logs
railway logs

# Follow logs
railway logs -f
```

### Check Status

```bash
railway status
```

### Open Dashboard

```bash
railway open
```

---

## 🎯 Best Practices

1. **Environment Variables**: Never commit secrets to Git
2. **Monitoring**: Check logs regularly for errors
3. **Backups**: Railway doesn't backup data automatically
4. **Testing**: Test locally before deploying
5. **User Keys**: Encourage users to use their own API keys

---

## 🆘 Getting Help

**Railway Discord**: [discord.gg/railway](https://discord.gg/railway)

**Railway Docs**: [docs.railway.app](https://docs.railway.app)

**Project Issues**: [GitHub Issues](https://github.com/dev-pratap-singh/LYZR-Hackathon/issues)

---

## ✨ Success!

Your RAG System is now live on Railway! 🎉

**Share your deployment:**
```
Frontend: https://YOUR_FRONTEND_URL
Backend:  https://YOUR_BACKEND_URL
API Docs: https://YOUR_BACKEND_URL/docs
```

Enjoy your production RAG system!
