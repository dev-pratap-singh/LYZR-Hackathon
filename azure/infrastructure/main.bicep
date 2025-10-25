// Azure Infrastructure for RAG System with Managed Services
// This Bicep template deploys all required Azure resources

@description('Environment name (dev, staging, prod)')
@allowed([
  'dev'
  'staging'
  'prod'
])
param environment string = 'dev'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('PostgreSQL administrator username')
param postgresAdminUsername string = 'ragadmin'

@description('PostgreSQL administrator password')
@secure()
param postgresAdminPassword string

// Removed unused redisAccessKey parameter

@description('OpenAI API Key')
@secure()
param openAIApiKey string

@description('Tags for all resources')
param tags object = {
  environment: environment
  project: 'rag-system'
  managedBy: 'bicep'
}

// ============================================================================
// Variables
// ============================================================================

var resourcePrefix = 'rag-${environment}'
var postgresServerName = '${resourcePrefix}-postgres'
var pgvectorServerName = '${resourcePrefix}-pgvector'
var redisName = '${resourcePrefix}-redis'
// Generate unique storage account name (3-24 chars, lowercase alphanumeric only)
var storageAccountName = replace('${resourcePrefix}st${uniqueString(resourceGroup().id)}', '-', '')
var keyVaultName = '${resourcePrefix}-kv-${uniqueString(resourceGroup().id)}'
var logAnalyticsName = '${resourcePrefix}-logs'
var containerRegistryName = replace('${resourcePrefix}acr${uniqueString(resourceGroup().id)}', '-', '')
var containerAppsEnvName = '${resourcePrefix}-env'
var vnetName = '${resourcePrefix}-vnet'

// Database sizing based on environment
var databaseConfig = environment == 'prod' ? {
  postgresSku: 'Standard_D4s_v3'
  pgvectorSku: 'Standard_D8s_v3'
  postgresStorageSize: 128
  pgvectorStorageSize: 256
  redisSku: 'Standard'
  redisFamily: 'C'
  redisCapacity: 3
} : environment == 'staging' ? {
  postgresSku: 'Standard_D2s_v3'
  pgvectorSku: 'Standard_D4s_v3'
  postgresStorageSize: 64
  pgvectorStorageSize: 128
  redisSku: 'Standard'
  redisFamily: 'C'
  redisCapacity: 1
} : {
  postgresSku: 'Standard_B2s'
  pgvectorSku: 'Standard_D2s_v3'
  postgresStorageSize: 32
  pgvectorStorageSize: 64
  redisSku: 'Basic'
  redisFamily: 'C'
  redisCapacity: 1
}

// ============================================================================
// Virtual Network
// ============================================================================

resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: vnetName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'database-subnet'
        properties: {
          addressPrefix: '10.0.1.0/24'
          serviceEndpoints: [
            {
              service: 'Microsoft.Storage'
            }
          ]
          delegations: []
        }
      }
      {
        name: 'container-apps-subnet'
        properties: {
          addressPrefix: '10.0.2.0/23'
          delegations: [
            {
              name: 'Microsoft.App.environments'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
        }
      }
      {
        name: 'redis-subnet'
        properties: {
          addressPrefix: '10.0.4.0/24'
          serviceEndpoints: []
        }
      }
    ]
  }
}

// ============================================================================
// Log Analytics Workspace
// ============================================================================

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ============================================================================
// Container Registry
// ============================================================================

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: 'Enabled'
  }
}

// ============================================================================
// PostgreSQL Flexible Server (Metadata)
// ============================================================================

resource postgresServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' = {
  name: postgresServerName
  location: location
  tags: tags
  sku: {
    name: databaseConfig.postgresSku
    tier: startsWith(databaseConfig.postgresSku, 'Standard_B') ? 'Burstable' : 'GeneralPurpose'
  }
  properties: {
    version: '15'
    administratorLogin: postgresAdminUsername
    administratorLoginPassword: postgresAdminPassword
    storage: {
      storageSizeGB: databaseConfig.postgresStorageSize
      autoGrow: 'Enabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: environment == 'prod' ? 'Enabled' : 'Disabled'
    }
    highAvailability: environment == 'prod' ? {
      mode: 'ZoneRedundant'
    } : null
  }
}

// PostgreSQL Database
resource postgresDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-03-01-preview' = {
  parent: postgresServer
  name: 'rag_database'
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// Firewall rule to allow Azure services
resource postgresFirewallAzure 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-03-01-preview' = {
  parent: postgresServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// ============================================================================
// PostgreSQL Flexible Server (PGVector)
// ============================================================================

resource pgvectorServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' = {
  name: pgvectorServerName
  location: location
  tags: tags
  sku: {
    name: databaseConfig.pgvectorSku
    tier: 'GeneralPurpose'
  }
  properties: {
    version: '15'
    administratorLogin: postgresAdminUsername
    administratorLoginPassword: postgresAdminPassword
    storage: {
      storageSizeGB: databaseConfig.pgvectorStorageSize
      autoGrow: 'Enabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: environment == 'prod' ? 'Enabled' : 'Disabled'
    }
    highAvailability: environment == 'prod' ? {
      mode: 'ZoneRedundant'
    } : null
  }
}

// PGVector Database
resource pgvectorDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-03-01-preview' = {
  parent: pgvectorServer
  name: 'vector_db'
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// Enable pgvector extension
resource pgvectorExtension 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2023-03-01-preview' = {
  parent: pgvectorServer
  name: 'azure.extensions'
  properties: {
    value: 'VECTOR'
    source: 'user-override'
  }
}

// Firewall rule to allow Azure services
resource pgvectorFirewallAzure 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-03-01-preview' = {
  parent: pgvectorServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// ============================================================================
// Azure Cache for Redis
// ============================================================================

resource redis 'Microsoft.Cache/redis@2023-08-01' = {
  name: redisName
  location: location
  tags: tags
  properties: {
    sku: {
      name: databaseConfig.redisSku
      family: databaseConfig.redisFamily
      capacity: databaseConfig.redisCapacity
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    redisConfiguration: {
      'maxmemory-policy': 'allkeys-lru'
    }
    redisVersion: '6'
  }
}

// ============================================================================
// Storage Account
// ============================================================================

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: environment == 'prod' ? 'Standard_GRS' : 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

// Blob service for document storage
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

// Container for uploaded documents
resource documentsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'documents'
  properties: {
    publicAccess: 'None'
  }
}

// ============================================================================
// Key Vault
// ============================================================================

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: environment == 'prod' ? true : null
  }
}

// Store secrets in Key Vault
resource postgresPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'postgres-admin-password'
  properties: {
    value: postgresAdminPassword
  }
}

resource redisAccessKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'redis-primary-access-key'
  properties: {
    value: redis.listKeys().primaryKey
  }
}

resource openAIKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'openai-api-key'
  properties: {
    value: openAIApiKey
  }
}

// ============================================================================
// Container Apps Environment
// ============================================================================

resource containerAppsEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppsEnvName
  location: location
  tags: tags
  properties: {
    vnetConfiguration: {
      infrastructureSubnetId: vnet.properties.subnets[1].id
    }
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    zoneRedundant: environment == 'prod' ? true : false
  }
}

// ============================================================================
// Outputs
// ============================================================================

output postgresServerFqdn string = postgresServer.properties.fullyQualifiedDomainName
output pgvectorServerFqdn string = pgvectorServer.properties.fullyQualifiedDomainName
output redisHostName string = redis.properties.hostName
output redisPrimaryKey string = redis.listKeys().primaryKey
output storageAccountName string = storageAccount.name
output keyVaultName string = keyVault.name
output containerRegistryName string = containerRegistry.name
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
output containerAppsEnvironmentId string = containerAppsEnv.id
output logAnalyticsWorkspaceId string = logAnalytics.id
