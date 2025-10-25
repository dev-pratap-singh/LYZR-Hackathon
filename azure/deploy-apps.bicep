// Deploy RAG System Container Apps
// This deploys backend and frontend with automatic CORS configuration

@description('Environment name')
param environment string = 'dev'

@description('Azure region')
param location string = resourceGroup().location

@description('Backend Docker image with tag')
param backendImage string

@description('Frontend Docker image with tag')
param frontendImage string

@description('OpenAI API Key (optional - users can provide their own)')
@secure()
param openAIApiKey string

// ============================================================================
// Reference existing resources
// ============================================================================

resource containerAppsEnv 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: 'rag-${environment}-env'
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: 'ragdevacrfxbt87w3'  // Hardcoded ACR name - replace if different
}

resource postgresServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' existing = {
  name: 'rag-${environment}-postgres'
}

resource pgvectorServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' existing = {
  name: 'rag-${environment}-pgvector'
}

resource redis 'Microsoft.Cache/redis@2023-08-01' existing = {
  name: 'rag-${environment}-redis'
}

// ============================================================================
// Variables
// ============================================================================

var backendAppName = 'rag-backend'
var frontendAppName = 'rag-frontend'
var registryPassword = containerRegistry.listCredentials().passwords[0].value

// Get postgres admin credentials from deployment params
// These should match what was used during initial infrastructure deployment
@secure()
param postgresAdminPassword string

// ============================================================================
// Backend Container App
// ============================================================================

resource backendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: backendAppName
  location: location
  tags: {
    environment: environment
    project: 'rag-system'
  }
  properties: {
    managedEnvironmentId: containerAppsEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
        // Automatic CORS configuration
        corsPolicy: {
          allowedOrigins: [
            'https://${frontendAppName}.${containerAppsEnv.properties.defaultDomain}'
            'http://localhost:3000'
            'http://localhost:8000'
          ]
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH']
          allowedHeaders: ['*']
          exposeHeaders: ['*']
          maxAge: 3600
          allowCredentials: true
        }
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          username: containerRegistry.name
          passwordSecretRef: 'registry-password'
        }
      ]
      secrets: [
        {
          name: 'registry-password'
          value: registryPassword
        }
        {
          name: 'postgres-password'
          value: postgresAdminPassword
        }
        {
          name: 'redis-key'
          value: redis.listKeys().primaryKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: backendAppName
          image: backendImage
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            // CORS Configuration
            {
              name: 'CORS_ORIGINS'
              value: 'https://${frontendAppName}.${containerAppsEnv.properties.defaultDomain},http://localhost:3000,http://localhost:8000'
            }
            // Database Configuration
            {
              name: 'POSTGRES_HOST'
              value: postgresServer.properties.fullyQualifiedDomainName
            }
            {
              name: 'POSTGRES_PORT'
              value: '5432'
            }
            {
              name: 'POSTGRES_DB'
              value: 'rag_database'
            }
            {
              name: 'POSTGRES_USER'
              value: 'ragadmin'
            }
            {
              name: 'POSTGRES_PASSWORD'
              secretRef: 'postgres-password'
            }
            {
              name: 'POSTGRES_SSL_MODE'
              value: 'require'
            }
            // PGVector Configuration
            {
              name: 'PGVECTOR_HOST'
              value: pgvectorServer.properties.fullyQualifiedDomainName
            }
            {
              name: 'PGVECTOR_PORT'
              value: '5432'
            }
            {
              name: 'PGVECTOR_DB'
              value: 'vector_db'
            }
            {
              name: 'PGVECTOR_USER'
              value: 'ragadmin'
            }
            {
              name: 'PGVECTOR_PASSWORD'
              secretRef: 'postgres-password'
            }
            {
              name: 'PGVECTOR_SSL_MODE'
              value: 'require'
            }
            // Redis Configuration
            {
              name: 'REDIS_HOST'
              value: redis.properties.hostName
            }
            {
              name: 'REDIS_PORT'
              value: '6380'
            }
            {
              name: 'REDIS_SSL'
              value: 'true'
            }
            {
              name: 'REDIS_ACCESS_KEY'
              secretRef: 'redis-key'
            }
            // OpenAI Configuration
            {
              name: 'OPENAI_API_KEY'
              value: openAIApiKey
            }
            {
              name: 'OPENAI_MODEL'
              value: 'gpt-4o-mini'
            }
            {
              name: 'OPENAI_EMBEDDING_MODEL'
              value: 'text-embedding-3-small'
            }
            // Search Configuration
            {
              name: 'ENABLE_VECTOR_SEARCH'
              value: 'true'
            }
            {
              name: 'ENABLE_GRAPH_SEARCH'
              value: 'false'
            }
            {
              name: 'ENABLE_FILTER_SEARCH'
              value: 'false'
            }
            {
              name: 'GRAPHRAG_ENABLED'
              value: 'false'
            }
            // Memory Configuration
            {
              name: 'MEMORY_ENABLED'
              value: 'true'
            }
            {
              name: 'MEMORY_DB_HOST'
              value: postgresServer.properties.fullyQualifiedDomainName
            }
            {
              name: 'MEMORY_DB_PORT'
              value: '5432'
            }
            {
              name: 'MEMORY_DB_NAME'
              value: 'rag_database'
            }
            {
              name: 'MEMORY_DB_USER'
              value: 'ragadmin'
            }
            {
              name: 'MEMORY_DB_PASSWORD'
              secretRef: 'postgres-password'
            }
            {
              name: 'MEMORY_APPROACH'
              value: 'external_llm'
            }
            {
              name: 'MEMORY_MODEL'
              value: 'gpt-4o-mini'
            }
            // App Configuration
            {
              name: 'DEPLOYMENT_ENVIRONMENT'
              value: 'azure'
            }
            {
              name: 'STORAGE_PATH'
              value: '/app/storage'
            }
            {
              name: 'CHUNK_SIZE'
              value: '1200'
            }
            {
              name: 'CHUNK_OVERLAP'
              value: '400'
            }
            {
              name: 'TOP_K_RESULTS'
              value: '5'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: environment == 'prod' ? 10 : 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
}

// ============================================================================
// Frontend Container App
// ============================================================================

resource frontendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: frontendAppName
  location: location
  tags: {
    environment: environment
    project: 'rag-system'
  }
  properties: {
    managedEnvironmentId: containerAppsEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 80
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          username: containerRegistry.name
          passwordSecretRef: 'registry-password'
        }
      ]
      secrets: [
        {
          name: 'registry-password'
          value: registryPassword
        }
      ]
    }
    template: {
      containers: [
        {
          name: frontendAppName
          image: frontendImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'REACT_APP_BACKEND_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: environment == 'prod' ? 10 : 3
      }
    }
  }
}

// ============================================================================
// Outputs
// ============================================================================

output backendUrl string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output frontendUrl string = 'https://${frontendApp.properties.configuration.ingress.fqdn}'
output backendFqdn string = backendApp.properties.configuration.ingress.fqdn
output frontendFqdn string = frontendApp.properties.configuration.ingress.fqdn
output corsOrigins string = 'https://${frontendAppName}.${containerAppsEnv.properties.defaultDomain},http://localhost:3000,http://localhost:8000'
