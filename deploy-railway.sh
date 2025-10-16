#!/bin/bash

# ==============================================================================
# Railway Full-Stack Deployment Script
# ==============================================================================
# This script automates the deployment of both backend and frontend to Railway
# 
# Prerequisites:
# 1. Railway CLI installed: npm i -g @railway/cli
# 2. Railway account and logged in: railway login
# 3. Git repository initialized and committed
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Railway CLI is installed
check_railway_cli() {
    if ! command -v railway &> /dev/null; then
        log_error "Railway CLI is not installed"
        log_info "Install it with: npm i -g @railway/cli"
        exit 1
    fi
    log_success "Railway CLI is installed"
}

# Check if user is logged in to Railway
check_railway_login() {
    if ! railway whoami &> /dev/null; then
        log_error "You are not logged in to Railway"
        log_info "Login with: railway login"
        exit 1
    fi
    log_success "Logged in to Railway as $(railway whoami)"
}

# Create new Railway project
create_railway_project() {
    log_info "Creating new Railway project..."
    railway init
    log_success "Railway project created"
}

# Deploy Backend
deploy_backend() {
    log_info "========================================="
    log_info "  Deploying Backend to Railway"
    log_info "========================================="
    
    cd "$(dirname "$0")"
    
    log_info "Creating backend service..."
    railway add
    
    log_info "Setting backend environment variables..."
    read -p "Enter your PostgreSQL password (or press Enter to skip): " POSTGRES_PASSWORD
    read -p "Enter your Neo4j password (or press Enter to skip): " NEO4J_PASSWORD
    read -p "Enter your OpenAI API key (optional, press Enter to skip): " OPENAI_KEY
    
    # Set critical environment variables
    if [ ! -z "$POSTGRES_PASSWORD" ]; then
        railway variables set POSTGRES_PASSWORD="$POSTGRES_PASSWORD"
    fi
    
    if [ ! -z "$NEO4J_PASSWORD" ]; then
        railway variables set NEO4J_AUTH="neo4j/$NEO4J_PASSWORD"
    fi
    
    if [ ! -z "$OPENAI_KEY" ]; then
        railway variables set OPENAI_API_KEY="$OPENAI_KEY"
    fi
    
    # Set default environment variables
    railway variables set MAX_PERFORMANCE="false"
    railway variables set GRAPHRAG_ENABLED="true"
    railway variables set GRAPHRAG_ENABLE_MULTIPASS="true"
    railway variables set ENABLE_VECTOR_SEARCH="true"
    railway variables set ENABLE_GRAPH_SEARCH="true"
    railway variables set ENABLE_FILTER_SEARCH="true"
    railway variables set CHUNK_SIZE="1200"
    railway variables set CHUNK_OVERLAP="500"
    railway variables set TOP_K_RESULTS="20"
    railway variables set MEMORY_ENABLED="true"
    
    log_info "Deploying backend..."
    railway up
    
    log_info "Getting backend URL..."
    BACKEND_URL=$(railway domain)
    
    if [ -z "$BACKEND_URL" ]; then
        log_warning "Backend URL not available yet. Generating domain..."
        railway domain
        BACKEND_URL=$(railway domain)
    fi
    
    log_success "Backend deployed successfully!"
    log_info "Backend URL: https://$BACKEND_URL"
    
    # Save backend URL for frontend deployment
    echo "$BACKEND_URL" > .backend_url
}

# Deploy Frontend
deploy_frontend() {
    log_info "========================================="
    log_info "  Deploying Frontend to Railway"
    log_info "========================================="
    
    cd "$(dirname "$0")/frontend"
    
    # Read backend URL
    if [ -f "../.backend_url" ]; then
        BACKEND_URL=$(cat ../.backend_url)
    else
        read -p "Enter your backend URL: " BACKEND_URL
    fi
    
    log_info "Creating frontend service..."
    railway init
    railway add
    
    log_info "Setting frontend environment variables..."
    railway variables set REACT_APP_API_URL="https://$BACKEND_URL"
    
    log_info "Deploying frontend..."
    railway up
    
    log_info "Getting frontend URL..."
    FRONTEND_URL=$(railway domain)
    
    if [ -z "$FRONTEND_URL" ]; then
        log_warning "Frontend URL not available yet. Generating domain..."
        railway domain
        FRONTEND_URL=$(railway domain)
    fi
    
    log_success "Frontend deployed successfully!"
    log_info "Frontend URL: https://$FRONTEND_URL"
    
    # Update backend CORS
    log_info "Updating backend CORS settings..."
    cd ..
    railway variables set ALLOWED_ORIGINS="https://$FRONTEND_URL,http://localhost:3000"
    
    log_success "CORS updated successfully!"
}

# Main deployment flow
main() {
    echo ""
    log_info "========================================="
    log_info "  Railway Full-Stack Deployment"
    log_info "========================================="
    echo ""
    
    # Pre-flight checks
    check_railway_cli
    check_railway_login
    
    echo ""
    log_info "What would you like to deploy?"
    echo "  1) Backend only"
    echo "  2) Frontend only"
    echo "  3) Full-stack (Backend + Frontend)"
    echo ""
    read -p "Enter your choice (1-3): " CHOICE
    
    case $CHOICE in
        1)
            deploy_backend
            ;;
        2)
            deploy_frontend
            ;;
        3)
            deploy_backend
            echo ""
            sleep 2
            deploy_frontend
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
    
    echo ""
    log_success "========================================="
    log_success "  Deployment Complete!"
    log_success "========================================="
    
    if [ -f ".backend_url" ]; then
        BACKEND_URL=$(cat .backend_url)
        echo ""
        log_info "Backend URL: https://$BACKEND_URL"
        log_info "API Docs:    https://$BACKEND_URL/docs"
    fi
    
    if [ ! -z "$FRONTEND_URL" ]; then
        echo ""
        log_info "Frontend URL: https://$FRONTEND_URL"
    fi
    
    echo ""
    log_info "Next steps:"
    echo "  1. Visit your frontend URL"
    echo "  2. Configure your OpenAI API key in Settings"
    echo "  3. Upload a PDF document"
    echo "  4. Start querying!"
    echo ""
}

# Run main function
main
