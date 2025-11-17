#!/usr/bin/env python3
"""
AI Model Performance and Cost Tester using Vercel AI Gateway
Tests different AI models through Vercel AI Gateway with fixed prompts and tracks performance metrics.
"""

import os
import time
import asyncio
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelResult:
    model_name: str
    response: str
    response_time: float
    input_tokens: int
    output_tokens: int
    cost: float
    error: Optional[str] = None


class AIGatewayTester:
    def __init__(self):
        self.console = Console()
        
        # Initialize OpenAI client with Vercel AI Gateway
        self.client = OpenAI(
            api_key=os.getenv('AI_GATEWAY_API_KEY'),
            base_url='https://ai-gateway.vercel.sh/v1'
        )

        self.pricing = {
            "openai/gpt-4.1": {"input": 2.00, "output": 8.00},
            "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "openai/gpt-5": {"input": 1.25, "output": 10.00},
            "openai/gpt-5-mini": {"input": 0.25, "output": 2.00},
            "zai/glm-4.6": {"input": 0.45, "output": 1.80},
            "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
            "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
            "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
            
        }

        # Fixed prompts for testing
        self.system_prompt = """You are a Lifestyle AI Coach. Your goal is to help the user design a system that will help them achieve their ideal daily routine and goals by asking thoughtful, structured follow-up questions. 
You will later progressively suggest atomic habits to the user. The habits you suggest later, are not always related to the first response. The user will be tracking their habits through the website.
The user might for example need to improve in a certain area (for example, discipline), but their first response is about a different area (for example, health).
- The user's first response is provided to you.
- Ask clarifying questions to better understand their goals, priorities, and constraints, mainly based on their first response.
- Always ask 5–10 questions total.
- Vary the question type appropriately: [TEXT, NUMBER, DATE, SELECT, CHECKBOX]. SELECT allows for only one answer, checkbox allows for multiple answers.
- Provide answer options for SELECT and CHECKBOX questions. Include an "Other" option where relevant.
- Use TEXT when freeform input is needed.
- Keep questions clear, concise, and user-friendly.

Output format:
Each category starts with | <category> on its own line.
Each question should follow this format:
1.[TYPE]; <question>; [options]

Example:
| Health  
1.[TEXT]; How would you describe your current health?; []  
2.[SELECT]; How active are you on a typical week?; [Very active, Moderately active, Rarely active, Other]  
3.[CHECKBOX]; What areas of health do you want to improve?; [Fitness, Nutrition, Sleep, Stress, Other]
        """
        self.user_prompt = """
        - What's the goal and lifestyle you want to achieve? I want to balance my life so I can learn more skills and grow more.
        - Why do you want to achieve this goal and lifestyle? Because I want to learn more skills, and to earn more money.
        - On a scale of 1–10, how much free time do you feel you realistically have for new habits? 5
        - Do you like structure (fixed schedule) or flexibility? Flexibility.
        - Are there constraints I should know? Money
        - Do you currently have any habits you're proud of? No
        - What are the main obstacles that usually stop you from starting or sticking with a new habit? Lack of time, tiredness.
        - What's one "non-negotiable" thing in your life you don't want disrupted? Work and my fun time.
        """

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate the cost based on token usage and model pricing."""
        pricing = self.pricing.get(model_name, {"input": 0, "output": 0})

        # Cost per 1M tokens, so divide by 1,000,000
        cost = (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

        return cost

    async def test_model(self, model_name: str) -> ModelResult:
        """Test a model through Vercel AI Gateway."""
        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': self.user_prompt}
                ]
            )

            end_time = time.time()
            response_time = end_time - start_time

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.calculate_cost(
                model_name, input_tokens, output_tokens
            )

            response_text = response.choices[0].message.content

            return ModelResult(
                model_name=model_name,
                response=response_text,
                response_time=response_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )

        except Exception as e:
            return ModelResult(
                model_name=model_name,
                response="",
                response_time=0,
                input_tokens=0,
                output_tokens=0,
                cost=0,
                error=str(e),
            )

    def display_prompts(self):
        """Display the prompts being used for testing."""
        self.console.print(
            Panel(
                f"[bold cyan]System Prompt:[/bold cyan]\n{self.system_prompt}\n\n"
                f"[bold cyan]User Prompt:[/bold cyan]\n{self.user_prompt}",
                title="Test Prompts",
                border_style="blue",
            )
        )
        self.console.print()

    def display_results(self, results: list[ModelResult]):
        """Display results in a formatted table."""
        table = Table(title="AI Model Performance Comparison (via Vercel AI Gateway)")

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Response Time", style="magenta")
        table.add_column("Input Tokens", style="green")
        table.add_column("Output Tokens", style="green")
        table.add_column("Cost ($)", style="yellow")
        table.add_column("Status", style="white")

        for result in results:
            if result.error:
                status = f"[red]Error: {result.error[:30]}...[/red]"
                table.add_row(
                    result.model_name, "N/A", "N/A", "N/A", "N/A", status
                )
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

    def display_responses(self, results: list[ModelResult]):
        """Display the actual responses from each model."""
        for result in results:
            if not result.error and result.response:
                self.console.print(
                    Panel(
                        result.response,
                        title=f"Response from {result.model_name}",
                        border_style="green",
                    )
                )
                self.console.print()

    async def run_tests(self):
        """Run tests on all models."""
        self.console.print("[bold blue]AI Gateway Model Performance Tester[/bold blue]")
        self.console.print(
            f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.console.print()

        self.display_prompts()

        models_to_test = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/gpt-5-mini",
            "openai/gpt-5",
            "openai/o3-mini",
            "openai/o4-mini",
        ]

        results = []

        with self.console.status("[bold green]Testing models...") as status:
            for model_name in models_to_test:
                status.update(f"[bold green]Testing {model_name}...")
                result = await self.test_model(model_name)
                results.append(result)
                self.console.print(f"✓ Completed {model_name}")

        self.console.print()
        self.display_results(results)
        self.display_responses(results)

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
    """Main function to run the AI Gateway tester."""
    tester = AIGatewayTester()

    # Check for API key
    if not os.getenv("AI_GATEWAY_API_KEY"):
        print("Error: AI_GATEWAY_API_KEY not found in environment variables")
        print("Please add it to your .env file")
        return

    await tester.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
