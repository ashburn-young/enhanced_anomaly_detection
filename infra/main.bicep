targetScope = 'resourceGroup'

metadata name = 'Anomaly Detection Application Infrastructure'
metadata description = 'Production-ready infrastructure for anomaly detection application using Container Apps, Azure OpenAI, Storage, and Key Vault'

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string



@description('Id of the user or app to assign application roles')
param principalId string = ''

// Generate a unique token to be used in naming resources
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))

// Define the tags to be applied to all resources
var tags = {
  'azd-env-name': environmentName
}

// Create Log Analytics workspace
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'la-${resourceToken}'
  location: location
  tags: tags
  properties: {
    retentionInDays: 30
    sku: {
      name: 'PerGB2018'
    }
  }
}

// Create Application Insights
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'ai-${resourceToken}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
  }
}

// Create Managed Identity
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'mi-${resourceToken}'
  location: location
  tags: tags
}

// Create Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${resourceToken}'
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enabledForDeployment: false
    enabledForTemplateDeployment: false
    enabledForDiskEncryption: false
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Create Storage Account
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${resourceToken}'
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Create blob container for data storage
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
}

resource dataContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'anomaly-data'
  properties: {
    publicAccess: 'None'
  }
}

// Create Azure OpenAI Service
resource openAIService 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'ai-${resourceToken}'
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: 'ai-${resourceToken}'
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// Deploy GPT-4o model
resource gptDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAIService
  name: 'gpt-4o'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-08-06'
    }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: {
    name: 'Standard'
    capacity: 100
  }
}

// Create Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: 'cr${resourceToken}'
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: 'Enabled'
  }
}

// Create Container App Environment
resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${resourceToken}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// Create Container App
resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${resourceToken}'
  location: location
  tags: union(tags, { 'azd-service-name': 'web' })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    environmentId: containerAppEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8501
        transport: 'http'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
          allowedHeaders: ['*']
          allowCredentials: false
        }
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: managedIdentity.id
        }
      ]
      secrets: [
        {
          name: 'azure-openai-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/azure-openai-key'
          identity: managedIdentity.id
        }
        {
          name: 'storage-account-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/storage-account-key'
          identity: managedIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'anomaly-detection-app'
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: openAIService.properties.endpoint
            }
            {
              name: 'AZURE_OPENAI_KEY'
              secretRef: 'azure-openai-key'
            }
            {
              name: 'AZURE_OPENAI_API_VERSION'
              value: '2024-02-01'
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
              value: gptDeployment.name
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT_NAME'
              value: storageAccount.name
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT_KEY'
              secretRef: 'storage-account-key'
            }
            {
              name: 'AZURE_STORAGE_CONTAINER_NAME'
              value: dataContainer.name
            }
            {
              name: 'AZURE_KEY_VAULT_URL'
              value: keyVault.properties.vaultUri
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: applicationInsights.properties.ConnectionString
            }
            {
              name: 'STREAMLIT_SERVER_PORT'
              value: '8501'
            }
            {
              name: 'ENVIRONMENT'
              value: 'production'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

// Role assignments for managed identity
// Key Vault Secrets User role
resource keyVaultSecretsUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, managedIdentity.id, '4633458b-17de-408a-b874-0445c86b69e6')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Storage Blob Data Contributor role
resource storageBlobDataContributorRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, managedIdentity.id, 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Cognitive Services OpenAI User role
resource cognitiveServicesOpenAIUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAIService.id, managedIdentity.id, '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  scope: openAIService
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// AcrPull role for container registry
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, managedIdentity.id, '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  scope: containerRegistry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// If principalId is provided, assign roles to the user/app
resource userKeyVaultSecretsUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  name: guid(keyVault.id, principalId, '4633458b-17de-408a-b874-0445c86b69e6')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: principalId
    principalType: 'User'
  }
}

resource userCognitiveServicesOpenAIUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  name: guid(openAIService.id, principalId, '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  scope: openAIService
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: principalId
    principalType: 'User'
  }
}

// Store secrets in Key Vault
resource openAIKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'azure-openai-key'
  properties: {
    value: openAIService.listKeys().key1
  }
  dependsOn: [
    keyVaultSecretsUserRole
  ]
}

resource storageKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'storage-account-key'
  properties: {
    value: storageAccount.listKeys().keys[0].value
  }
  dependsOn: [
    keyVaultSecretsUserRole
  ]
}

// Outputs
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.properties.loginServer
output AZURE_CONTAINER_REGISTRY_NAME string = containerRegistry.name
output AZURE_KEY_VAULT_NAME string = keyVault.name
output AZURE_KEY_VAULT_URL string = keyVault.properties.vaultUri
output AZURE_LOCATION string = location
output AZURE_OPENAI_ENDPOINT string = openAIService.properties.endpoint
output AZURE_OPENAI_API_VERSION string = '2024-02-01'
output AZURE_OPENAI_DEPLOYMENT_NAME string = gptDeployment.name
output AZURE_STORAGE_ACCOUNT_NAME string = storageAccount.name
output AZURE_STORAGE_CONTAINER_NAME string = dataContainer.name
output APPLICATIONINSIGHTS_CONNECTION_STRING string = applicationInsights.properties.ConnectionString
output SERVICE_WEB_IDENTITY_PRINCIPAL_ID string = managedIdentity.properties.principalId
output SERVICE_WEB_NAME string = containerApp.name
output SERVICE_WEB_URI string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
