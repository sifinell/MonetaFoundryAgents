from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AzureAISearchTool, FunctionTool, ToolSet
from agents.functions.crm import crm_functions
from agents.functions.news import news_functions
import yaml
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

model=os.environ["OPENAI_API_MODEL"]

class AgentInitializer:
    def __init__(self, project_client: AIProjectClient):
        self.project_client = project_client

    def load_yaml(self, yaml_path: str):
        script_dir = os.path.dirname(__file__)
        absolute_path = os.path.join(script_dir, yaml_path)
        print(f"Absolute path to YAML file: {absolute_path}")
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"File not found: {absolute_path}")
        with open(absolute_path, 'r') as file:
            return yaml.safe_load(file)
        
    def create_agent_cio(self, agent_definition: dict):

        # Retrieves AI Search connection from Project connections 
        # Further checks should be made if more AI Search connections were created in Foundry
        conn_list = self.project_client.connections.list()
        conn_id = ""
        for conn in conn_list:
            if conn.connection_type == "AZURE_AI_SEARCH":
                conn_id = conn.id

        ai_search = AzureAISearchTool(index_connection_id=conn_id, index_name="cio-index")

        agent = self.project_client.agents.create_agent(
            model=model,
            name=agent_definition["name"],
            description=agent_definition["description"],
            instructions=agent_definition["instructions"],
            temperature=agent_definition["temperature"],
            tools=ai_search.definitions,
            tool_resources=ai_search.resources,
        )
        print(f"Created agent, ID: {agent.id}")
        return agent
    
    def create_agent_crm(self, agent_definition: dict):

        # Initialize function tool
        functions = FunctionTool(functions=crm_functions)
        toolset = ToolSet()
        toolset.add(functions)
        
        agent = self.project_client.agents.create_agent(
            model=model,
            name=agent_definition["name"],
            description=agent_definition["description"],
            instructions=agent_definition["instructions"],
            temperature=agent_definition["temperature"],
            toolset=toolset,
        )
        print(f"Created agent, ID: {agent.id}")
        return agent    
        
    def create_agent_funds(self, agent_definition: dict):

        # Retrieves AI Search connection from Project connections 
        # Further checks should be made if more AI Search connections were created in Foundry
        conn_list = self.project_client.connections.list()
        conn_id = ""
        for conn in conn_list:
            if conn.connection_type == "AZURE_AI_SEARCH":
                conn_id = conn.id

        ai_search = AzureAISearchTool(index_connection_id=conn_id, index_name="funds-index")

        agent = self.project_client.agents.create_agent(
            model=model,
            name=agent_definition["name"],
            description=agent_definition["description"],
            instructions=agent_definition["instructions"],
            temperature=agent_definition["temperature"],
            tools=ai_search.definitions,
            tool_resources=ai_search.resources,
        )
        print(f"Created agent, ID: {agent.id}")
        return agent
    
    def create_agent_news(self, agent_definition: dict):

        # Initialize function tool
        functions = FunctionTool(functions=news_functions)
        toolset = ToolSet()
        toolset.add(functions)
        
        agent = self.project_client.agents.create_agent(
            model=model,
            name=agent_definition["name"],
            description=agent_definition["description"],
            instructions=agent_definition["instructions"],
            temperature=agent_definition["temperature"],
            toolset=toolset,
        )
        print(f"Created agent, ID: {agent.id}")
        return agent    
    
    def create_agent_responder(self, agent_definition: dict):

        agent = self.project_client.agents.create_agent(
            model=model,
            name=agent_definition["name"],
            description=agent_definition["description"],
            instructions=agent_definition["instructions"],
        )
        print(f"Created agent, ID: {agent.id}")
        return agent

    def create_agent(self, agent_type:str):

        # Load agent definition from yaml files
        agent_definition = self.load_yaml("./definitions/" + agent_type + ".yaml")

        # List all agents from Project
        agents_list = self.project_client.agents.list_agents()

        # Check if agent already exists 
        # If that's the case, just return it and don't create a new one
        for agent in agents_list.data:
            if agent.name == agent_definition["name"]:
                return self.project_client.agents.get_agent(agent.id)

        # Create agent 
        # This could be improved to have a single create function
        if agent_type == "cio":
            return self.create_agent_cio(agent_definition)
        elif agent_type == "crm":
            return self.create_agent_crm(agent_definition)
        elif agent_type == "funds":
            return self.create_agent_funds(agent_definition)
        elif agent_type == "news":
            return self.create_agent_news(agent_definition)
        elif agent_type == "responder":
            return self.create_agent_responder(agent_definition)