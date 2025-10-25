// Container Apps deployment with automatic CORS configuration
// This module deploys both frontend and backend with proper CORS settings

@description('Container Apps Environment ID')
param containerAppsEnvironmentId string

@description('Container Registry Login Server')
param containerRegistryLoginServer string

@description('Container Registry Username')
param containerRegistryUsername string

@description('Container Registry Password')
@secure()
param containerRegistryPassword string

@description('Environment name')
param environment string = 'dev'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Backend image tag')
param backendImageTag string = 'latest'

@description('Frontend image tag')
param frontendImageTag string = 'latest'

@description('Environment variables for backend')
param backendEnvVars array

@description('Tags for all resources')
param tags object = {
  environment: environment
  project: 'rag-system'
  managedBy: 'bicep'
}

// ============================================================================
// Variables
// ============================================================================

var backendAppName = 'rag-backend'
var frontendAppName = 'rag-frontend'

// ============================================================================
// Backend Container App
// ============================================================================

resource backendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: backendAppName
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
        // CORS Configuration - automatically allows frontend
        corsPolicy: {
          allowedOrigins: [
            'https://${frontendAppName}.${split(split(containerAppsEnvironmentId, '/')[8], '.')[1]}.${split(split(containerAppsEnvironmentId, '/')[8], '.')[2]}.azurecontainerapps.io'
            'http://localhost:3000'
            'http://localhost:8000'
          ]
          allowedMethods: [
            'GET'
            'POST'
            'PUT'
            'DELETE'
            'OPTIONS'
            'PATCH'
          ]
          allowedHeaders: [
            '*'
          ]
          exposeHeaders: [
            '*'
          ]
          maxAge: 3600
          allowCredentials: true
        }
      }
      registries: [
        {
          server: containerRegistryLoginServer
          username: containerRegistryUsername
          passwordSecretRef: 'registry-password'
        }
      ]
      secrets: [
        {
          name: 'registry-password'
          value: containerRegistryPassword
        }
      ]
    }
    template: {
      containers: [
        {
          name: backendAppName
          image: '${containerRegistryLoginServer}/${backendAppName}:${backendImageTag}'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: concat(backendEnvVars, [
            {
              // Add CORS origins as environment variable for backend code
              name: 'CORS_ORIGINS'
              value: 'https://${frontendAppName}.${split(split(containerAppsEnvironmentId, '/')[8], '.')[1]}.${split(split(containerAppsEnvironmentId, '/')[8], '.')[2]}.azurecontainerapps.io,http://localhost:3000,http://localhost:8000'
            }
          ])
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
  tags: tags
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      ingress: {
        external: true
        targetPort: 80
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: containerRegistryLoginServer
          username: containerRegistryUsername
          passwordSecretRef: 'registry-password'
        }
      ]
      secrets: [
        {
          name: 'registry-password'
          value: containerRegistryPassword
        }
      ]
    }
    template: {
      containers: [
        {
          name: frontendAppName
          image: '${containerRegistryLoginServer}/${frontendAppName}:${frontendImageTag}'
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

output backendFqdn string = backendApp.properties.configuration.ingress.fqdn
output backendUrl string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output frontendFqdn string = frontendApp.properties.configuration.ingress.fqdn
output frontendUrl string = 'https://${frontendApp.properties.configuration.ingress.fqdn}'
output backendName string = backendApp.name
output frontendName string = frontendApp.name
