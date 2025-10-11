# RAG System

Advanced Retrieval-Augmented Generation system with Vector, Graph, and Filter search capabilities.

## Quick Start

1. Clone the repository

2. **Setup Environment Variables:**
   ```bash
   cp .env-example .env
   ```

3. **Configure Credentials (IMPORTANT!):**
   Edit `.env` and set the following required values:
   ```bash
   # Database credentials
   POSTGRES_USER=your_username
   POSTGRES_PASSWORD=your_secure_password

   # Neo4j credentials (format: username/password)
   NEO4J_AUTH=neo4j/your_secure_password

   # OpenAI API Key (required for embeddings and LLM)
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

   ⚠️ **Security Note:** Never commit `.env` file to version control!

4. Run the system:
   ```bash
   docker-compose up --build
   ```

## Access Points

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Backend Docs: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474
- PostgreSQL: localhost:5434
- PGVector: localhost:5435

## Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Security

### Environment Variables

All sensitive data is stored in `.env` file which is **excluded from version control**.

**Protected Credentials:**
- `POSTGRES_PASSWORD` - Database password
- `NEO4J_AUTH` - Neo4j username/password
- `OPENAI_API_KEY` - OpenAI API key

**Best Practices:**
1. Never commit `.env` file to git (already in `.gitignore`)
2. Use `.env-example` as a template for setup
3. Use strong, unique passwords for production
4. Rotate API keys regularly
5. Use different credentials for development and production

### Production Deployment

For production environments:
1. Use secrets management (AWS Secrets Manager, Azure Key Vault, etc.)
2. Set environment variables via your hosting platform
3. Enable SSL/TLS for all connections
4. Use managed database services with encrypted connections
5. Implement API rate limiting and authentication

## Architecture

See `architecture.md` for detailed system design.
