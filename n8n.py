#!/usr/bin/env python3
"""
n8n Workflow Generator using AI
Generates n8n workflows based on user requirements by querying available nodes and their schemas.
Tests multiple AI models to compare performance and cost.
"""

import os
import time
import asyncio
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()


@dataclass
class N8NNode:
    """Represents an n8n node with its metadata."""
    name: str
    display_name: str
    description: str
    version: int


@dataclass
class ModelResult:
    """Results from testing a model."""
    model_name: str
    workflow: str
    response_time: float
    input_tokens: int
    output_tokens: int
    cost: float
    error: Optional[str] = None


class N8NNodeTools:
    """Tools for interacting with n8n node API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DATARAMA_API_KEY")
        self.base_url = "https://api.datarama.ai/api/internal-nodes"
        
        if not self.api_key:
            raise ValueError("DATARAMA_API_KEY not found in environment variables")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key for authentication."""
        return {
            "X-API-Key": f"{self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_available_nodes(self) -> List[Dict]:
        """
        Fetch the list of available n8n nodes.
        
        Returns:
            List of node objects with basic metadata
        """
        try:
            response = requests.get(
                f"{self.base_url}/nodes/",
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch nodes: {str(e)}")
    
    def get_node_details(self, node_name: str) -> Dict:
        """
        Fetch detailed information about a specific node.
        
        Args:
            node_name: The name of the node to fetch details for
            
        Returns:
            Detailed node schema including parameters, inputs, outputs
        """
        try:
            response = requests.get(
                f"{self.base_url}/nodes/{node_name}/",
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch node details for '{node_name}': {str(e)}")


class N8NWorkflowGenerator:
    """AI-powered n8n workflow generator with multi-model testing."""
    
    def __init__(self):
        self.console = Console()
        self.node_tools = N8NNodeTools()
        
        # Initialize OpenAI client with Vercel AI Gateway
        self.client = OpenAI(
            api_key=os.getenv('AI_GATEWAY_API_KEY'),
            base_url='https://ai-gateway.vercel.sh/v1'
        )
        
        self.pricing = {
            "openai/gpt-oss-20b": {"input": 0.07, "output": 0.30},
            "openai/gpt-oss-120b": {"input": 0.10, "output": 0.50},
            "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        }
        
        self.system_prompt = """You are an expert n8n workflow generator for a SaaS developing proxy automation. Your task is to create valid n8n workflow JSON based on user requirements.

**Available Tools:**
- get_available_nodes(): Returns a list of all available n8n nodes you can use
- get_node_details(node_name): Returns detailed schema for a specific node including parameters, inputs, and outputs

**Workflow Generation Process:**
1. Understand the user's requirement
2. Call get_available_nodes() to see what nodes are available
3. For each node you plan to use, call get_node_details(node_name) to understand its configuration
4. Generate a valid n8n workflow JSON structure

NOTES:
CUSTOM.* nodes MUST be used for scraping tasks, parsing, and storing data. Always use these nodes when user wants to scrape, parse, store or compare data.

**n8n Workflow Structure:**
```json
{
  "name": "Workflow Name",
  "nodes": [
    {
      "parameters": {},
      "name": "Node Name",
      "type": "n8n-nodes-base.nodeName",
      "typeVersion": 1,
      "position": [x, y]
    }
  ],
  "connections": {
    "Node1": {
      "main": [[{"node": "Node2", "type": "main", "index": 0}]]
    }
  },
  "settings": {}
}
```

**Important Rules:**
- Undestand what the users wants, break down the nodes.
- After Scraping, almost always create a parser node after the scraper to parse the output, unless the user needs raw html.
- Always validate that nodes exist before using them
- Ensure proper connections between nodes
- Include all required parameters for each node
- Use realistic position coordinates (increment by 200-300 for readability)
- Return only valid JSON that can be imported directly into n8n
- If a node doesn't exist, suggest alternatives or ask for clarification
- ALWAYS use CUSTOM.* nodes for tasks that can be done with them.

**Response Format:**
Always respond with valid JSON workflow structure. Nothing but the valid JSON workflow."""

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost based on token usage and model pricing."""
        pricing = self.pricing.get(model_name, {"input": 0, "output": 0})
        cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
        return cost

    async def generate_workflow(self, user_requirement: str, model_name: str = "openai/gpt-4o") -> ModelResult:
        """
        Generate an n8n workflow based on user requirements using a specific model.
        
        Args:
            user_requirement: Natural language description of the desired workflow
            model_name: The AI model to use for generation
            
        Returns:
            ModelResult with workflow and performance metrics
        """
        try:
            start_time = time.time()
            
            # Prepare tools for function calling
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_available_nodes",
                        "description": "Get a list of all available n8n nodes that can be used in workflows",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_node_details",
                        "description": "Get detailed schema and configuration options for a specific n8n node",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "node_name": {
                                    "type": "string",
                                    "description": "The name of the node to get details for"
                                }
                            },
                            "required": ["node_name"]
                        }
                    }
                }
            ]
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_requirement}
            ]
            
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Conversation loop with function calling
            while True:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                
                # Track token usage
                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens
                
                message = response.choices[0].message
                messages.append(message)
                
                # Check if the model wants to call functions
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = eval(tool_call.function.arguments)
                        
                        # Execute the appropriate function
                        if function_name == "get_available_nodes":
                            result = self.node_tools.get_available_nodes()
                        elif function_name == "get_node_details":
                            result = self.node_tools.get_node_details(function_args["node_name"])
                        else:
                            result = {"error": "Unknown function"}
                        
                        # Add function result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result)
                        })
                else:
                    # No more function calls, return the final response
                    end_time = time.time()
                    response_time = end_time - start_time
                    cost = self.calculate_cost(model_name, total_input_tokens, total_output_tokens)
                    
                    return ModelResult(
                        model_name=model_name,
                        workflow=message.content,
                        response_time=response_time,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        cost=cost
                    )
        
        except Exception as e:
            return ModelResult(
                model_name=model_name,
                workflow="",
                response_time=0,
                input_tokens=0,
                output_tokens=0,
                cost=0,
                error=str(e)
            )

    def display_results(self, results: List[ModelResult]):
        """Display results in a formatted table."""
        table = Table(title="n8n Workflow Generator - Model Performance Comparison")

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Response Time", style="magenta")
        table.add_column("Input Tokens", style="green")
        table.add_column("Output Tokens", style="green")
        table.add_column("Cost ($)", style="yellow")
        table.add_column("Status", style="white")

        for result in results:
            if result.error:
                status = f"[red]Error: {result.error[:30]}...[/red]"
                table.add_row(result.model_name, "N/A", "N/A", "N/A", "N/A", status)
            else:
                table.add_row(
                    result.model_name,
                    f"{result.response_time:.2f}s",
                    str(result.input_tokens),
                    str(result.output_tokens),
                    f"${result.cost:.6f}",
                    "[green]Success[/green]",
                )

        self.console.print(table)
        self.console.print()

    def save_workflows(self, results: List[ModelResult], user_requirement: str):
        """Save the generated workflows to individual files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = "workflows_output"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for result in results:
            if not result.error and result.workflow:
                # Create safe filename from model name
                safe_model_name = result.model_name.replace("/", "_").replace(":", "_")
                filename = f"{output_dir}/workflow_{safe_model_name}_{timestamp}.json"
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(result.workflow)
                    self.console.print(f"[green]✓[/green] Saved workflow from {result.model_name} to {filename}")
                except Exception as e:
                    self.console.print(f"[red]✗[/red] Failed to save {result.model_name}: {str(e)}")
        
        # Save metadata file with comparison
        metadata_file = f"{output_dir}/comparison_{timestamp}.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"n8n Workflow Generation Comparison\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nUser Requirement:\n{user_requirement}\n")
            f.write(f"\n{'='*80}\n\n")
            
            for result in results:
                f.write(f"Model: {result.model_name}\n")
                if result.error:
                    f.write(f"Status: ERROR - {result.error}\n")
                else:
                    f.write(f"Response Time: {result.response_time:.2f}s\n")
                    f.write(f"Input Tokens: {result.input_tokens}\n")
                    f.write(f"Output Tokens: {result.output_tokens}\n")
                    f.write(f"Cost: ${result.cost:.6f}\n")
                f.write(f"\n{'-'*80}\n\n")
        
        self.console.print(f"\n[cyan]Comparison metadata saved to {metadata_file}[/cyan]")

    async def test_models(self, user_requirement: str):
        """Test multiple models and compare their performance."""
        self.console.print("[bold blue]n8n Workflow Generator - Model Testing[/bold blue]")
        self.console.print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.console.print()
        
        self.console.print(Panel(
            f"[bold cyan]User Requirement:[/bold cyan]\n{user_requirement}",
            title="Workflow Request",
            border_style="blue",
        ))
        self.console.print()

        # Use only models that have pricing defined
        models_to_test = list(self.pricing.keys())

        results = []

        with self.console.status("[bold green]Testing models...") as status:
            for model_name in models_to_test:
                status.update(f"[bold green]Testing {model_name}...")
                result = await self.generate_workflow(user_requirement, model_name)
                results.append(result)
                self.console.print(f"✓ Completed {model_name}")

        self.console.print()
        self.display_results(results)
        self.save_workflows(results, user_requirement)

        # Summary
        successful_results = [r for r in results if not r.error]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x.response_time)
            cheapest = min(successful_results, key=lambda x: x.cost)

            self.console.print(
                Panel(
                    f"[green]Fastest Model:[/green] {fastest.model_name} ({fastest.response_time:.2f}s)\n"
                    f"[yellow]Cheapest Model:[/yellow] {cheapest.model_name} (${cheapest.cost:.6f})",
                    title="Summary",
                    border_style="cyan",
                )
            )


async def main():
    """Main function to run the n8n workflow generator with model testing."""
    generator = N8NWorkflowGenerator()
    
    # Check for API keys
    if not os.getenv("AI_GATEWAY_API_KEY"):
        print("Error: AI_GATEWAY_API_KEY not found in environment variables")
        print("Please add it to your .env file")
        return
    
    if not os.getenv("DATARAMA_API_KEY"):
        print("Error: DATARAMA_API_KEY not found in environment variables")
        print("Please add it to your .env file")
        return
    
    # Example: Generate a workflow and test multiple models
    user_request = """
    I need to scrape this url every 10 minutes: https://datarama.ai/pricing and then store the prices for its plans.
    """
    
    await generator.test_models(user_request)


if __name__ == "__main__":
    asyncio.run(main())
