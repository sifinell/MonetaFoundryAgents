# Moneta Integration with AI Agent Service (Foundry)

## Introduction

This project recreates the agents available in the [Moneta repository](https://github.com/Azure-Samples/moneta-agents/tree/main) by leveraging the AI Agent Service (Foundry). Instead of relying on Moneta's native implementation, this repository uses Foundry to instantiate and manage AI agents, enabling a streamlined and scalable approach to financial AI automation. The agents maintain the same functionalities but are deployed and orchestrated through the AI Agent Service.

## Prerequisites

Before configuring and running this repo, ensure that the required Azure services are already deployed from Moneta accelerator:

- **Azure AI Search Service**
- **Azure Cosmos DB**

## Configuration Instructions

To set up and configure this repo, follow these steps:

### 1. Create an AI Foundry Hub and Project

This repo requires an **AI Foundry Hub and Project** in Azure AI Studio. Follow this guide to create a new project:  
[Creating a Project in Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/create-projects?tabs=ai-studio)

### 2. Connect to Azure AI Search

Once your AI Project is created, establish a connection to the existing **Azure AI Search** service:  
[Adding a Connection in Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/connections-add)

### 3. Configure the `.env` File

The `.env` file contains essential configuration details required for this repo to function correctly. The **Project Connection String** can be found in the **Azure AI Foundry portal** under:

- **Project details** → **Project connection string**

![image](https://github.com/user-attachments/assets/3925ad10-05e9-4ccd-a654-9ed429744ce6)

Ensure the `.env` file is correctly placed in the root directory of the project.

## Running the Code

Once configuration is complete, execute the script to initialize and orchestrate AI agents. The script:

1. Loads environment variables  
2. Connects to the Azure AI Project  
3. Creates and manages AI agents  
4. Handles user queries and processes responses  

Run the script with:

```bash
python main.py
```

## Agent Functionality

The Moneta framework includes multiple AI agents specialized for financial tasks:

- **CIO Agent**: Provides insights from the Chief Investment Office (CIO)
- **CRM Agent**: Fetches client-related financial data
- **Funds Agent**: Tracks investment fund performance
- **News Agent**: Aggregates financial news
- **Responder Agent**: Handles general user queries
- **Team Leader Agent**: Orchestrates agent collaboration

Example queries:

```plaintext
Provide me a summary of the portfolio's positions of my client id 123456.
What are our CIO's views on the AI sector?
Can you update me on the UBS 100 Index Switzerland Equity Fund CHF?
```

## License

This project is part of the [Azure Samples](https://github.com/Azure-Samples) collection and follows the associated licensing terms.
