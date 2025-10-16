#!/bin/bash

# Quick Railway Deployment Script
# Run this to deploy your backend immediately

set -e

echo "🚀 Deploying Backend to Railway..."
echo ""

# Check if in correct directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found. Make sure you're in the project root directory."
    exit 1
fi

echo "✅ Dockerfile found!"
echo ""

# Deploy to Railway
echo "📦 Deploying to Railway..."
railway up

echo ""
echo "✅ Backend deployment initiated!"
echo ""
echo "⚠️  IMPORTANT: You need to configure environment variables:"
echo ""
echo "1. Add PostgreSQL database:"
echo "   railway add"
echo "   (Select PostgreSQL)"
echo ""
echo "2. Add Redis:"
echo "   railway add"
echo "   (Select Redis)"
echo ""
echo "3. Set environment variables:"
echo "   railway variables set NEO4J_HOST='your-neo4j-aura-host'"
echo "   railway variables set NEO4J_AUTH='neo4j/your-password'"
echo "   railway variables set ELASTICSEARCH_HOST='your-elastic-host'"
echo "   railway variables set ELASTICSEARCH_PORT='9200'"
echo "   railway variables set ALLOWED_ORIGINS='https://your-frontend-url'"
echo "   railway variables set OPENAI_API_KEY='sk-proj-xxx'  # Optional"
echo ""
echo "4. Get your backend URL:"
echo "   railway domain"
echo ""
echo "📚 See RAILWAY_FIX.md for detailed database setup instructions"
