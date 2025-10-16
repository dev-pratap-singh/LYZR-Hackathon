# 🚀 Railway Deployment Quick Start

**Get your RAG System live in 15 minutes!**

---

## Option 1: Automated Script (Recommended)

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login to Railway
railway login

# 3. Run deployment script
./deploy-railway.sh

# 4. Follow the prompts and you're done! 🎉
```

---

## Option 2: Manual Commands

### Backend Deployment

```bash
# Login
railway login

# Initialize project
railway init

# Set environment variables (replace with your values)
railway variables set POSTGRES_PASSWORD="YOUR_PASSWORD"
railway variables set NEO4J_AUTH="neo4j/YOUR_PASSWORD"
railway variables set OPENAI_API_KEY="sk-proj-YOUR_KEY"
railway variables set MAX_PERFORMANCE="false"
railway variables set GRAPHRAG_ENABLED="true"

# Deploy
railway up

# Get backend URL
railway domain
# Save this URL! Example: rag-backend.up.railway.app
```

### Frontend Deployment

```bash
# Navigate to frontend
cd frontend

# Initialize frontend service
railway init

# Set backend URL (replace with YOUR backend URL from above)
railway variables set REACT_APP_API_URL="https://YOUR_BACKEND_URL"

# Deploy
railway up

# Get frontend URL
railway domain
# Example: rag-frontend.up.railway.app
```

### Update Backend CORS

```bash
# Go back to root
cd ..

# Link to backend service
railway link

# Update CORS (replace with YOUR frontend URL)
railway variables set ALLOWED_ORIGINS="https://YOUR_FRONTEND_URL,http://localhost:3000"
```

---

## ✅ Verify Deployment

```bash
# Check backend health
curl https://YOUR_BACKEND_URL/health

# Open frontend in browser
open https://YOUR_FRONTEND_URL
```

---

## 🆘 Troubleshooting

| Issue | Fix |
|-------|-----|
| CORS errors | `railway variables set ALLOWED_ORIGINS="https://your-frontend-url"` |
| Backend won't start | Check logs: `railway logs` |
| Frontend blank page | Verify `REACT_APP_API_URL` is set |
| Database errors | Restart: `railway restart` |

---

## 📚 Full Documentation

For detailed instructions, see:
- **[RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)** - Complete step-by-step guide
- **[README.md](README.md)** - Project overview and local setup

---

**Need help?** Join [Railway Discord](https://discord.gg/railway)
