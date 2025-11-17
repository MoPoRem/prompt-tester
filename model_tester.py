#!/usr/bin/env python3
"""
AI Model Performance and Cost Tester
Tests different AI models with fixed prompts and tracks performance metrics.
"""

import os
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

import openai
import anthropic
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
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
    flex_cost: float
    standard_cost: float
    error: Optional[str] = None


class ModelTester:
    def __init__(self):
        self.console = Console()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.pricing_flex = {
            "gpt-5": {"input": 0.625, "output": 5.00},
            "gpt-5-mini": {"input": 0.125, "output": 1.00},
            "o3": {"input": 1.00, "output": 4.00},
            "o4-mini": {"input": 0.55, "output": 2.20},
        }

        self.pricing_standard = {
            "gpt-5": {"input": 1.25, "output": 10.00},
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "o3": {"input": 2.00, "output": 8.00},
            "o3-mini": {"input": 1.10, "output": 4.40},
            "o4-mini": {"input": 1.10, "output": 4.40},
        }

        # Fixed prompts for testing
        self.system_prompt = """You are a Lifestyle AI Coach. Your goal is to help the user design a system that will help them achieve their ideal daily routine and goals by asking thoughtful, structured follow-up questions. 
You will later progressively suggest atomic habits to the user. The habits you suggest later, are not always related to the first response. The user will be tracking their habits through the website.
The user might for example need to improve in a certain area (for example, discipline), but their first response is about a different area (for example, health).
- The user’s first response is provided to you.
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
        - Do you currently have any habits you’re proud of? No
        - What are the main obstacles that usually stop you from starting or sticking with a new habit? Lack of time, tiredness.
        - What’s one “non-negotiable” thing in your life you don’t want disrupted? Work and my fun time.
        """

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        tier: str = "standard",
    ) -> tuple[float, float]:
        """Calculate the cost based on token usage and model pricing for both tiers."""
        pricing_table = (
            self.pricing_standard if tier == "standard" else self.pricing_flex
        )

        if model_name not in pricing_table:
            return 0.0, 0.0

        flex_pricing = self.pricing_flex.get(model_name, {"input": 0, "output": 0})
        standard_pricing = self.pricing_standard.get(
            model_name, {"input": 0, "output": 0}
        )

        # Cost per 1M tokens, so divide by 1,000,000
        flex_cost = (input_tokens / 1_000_000) * flex_pricing["input"] + (
            output_tokens / 1_000_000
        ) * flex_pricing["output"]
        standard_cost = (input_tokens / 1_000_000) * standard_pricing["input"] + (
            output_tokens / 1_000_000
        ) * standard_pricing["output"]

        return flex_cost, standard_cost

    async def test_openai_model(self, model_name: str) -> ModelResult:
        """Test an OpenAI model."""
        try:
            start_time = time.time()

            response = self.openai_client.responses.create(
                model=model_name,
                instructions=self.system_prompt,
                input=self.user_prompt,
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Responses API returns different usage fields than Chat Completions
            input_tokens = getattr(
                response.usage,
                "input_tokens",
                getattr(response.usage, "prompt_tokens", 0),
            )
            output_tokens = getattr(
                response.usage,
                "output_tokens",
                getattr(response.usage, "completion_tokens", 0),
            )
            flex_cost, standard_cost = self.calculate_cost(
                model_name, input_tokens, output_tokens
            )

            # Extract text from Responses API structure
            response_text = ""
            try:
                if hasattr(response, "output_text") and response.output_text:
                    response_text = response.output_text
                elif hasattr(response, "output") and response.output:
                    first_output = response.output[0]
                    if hasattr(first_output, "content") and first_output.content:
                        first_content = first_output.content[0]
                        if hasattr(first_content, "text") and first_content.text:
                            response_text = first_content.text
                    elif hasattr(first_output, "text") and first_output.text:
                        response_text = first_output.text
            except Exception:
                response_text = ""

            return ModelResult(
                model_name=model_name,
                response=response_text,
                response_time=response_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                flex_cost=flex_cost,
                standard_cost=standard_cost,
            )

        except Exception as e:
            return ModelResult(
                model_name=model_name,
                response="",
                response_time=0,
                input_tokens=0,
                output_tokens=0,
                flex_cost=0,
                standard_cost=0,
                error=str(e),
            )

    async def test_anthropic_model(self, model_name: str) -> ModelResult:
        """Test an Anthropic model."""
        try:
            start_time = time.time()

            response = self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=500,
                system=self.system_prompt,
                messages=[{"role": "user", "content": self.user_prompt}],
            )

            end_time = time.time()
            response_time = end_time - start_time

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            flex_cost, standard_cost = self.calculate_cost(
                model_name, input_tokens, output_tokens
            )

            return ModelResult(
                model_name=model_name,
                response=response.content[0].text,
                response_time=response_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                flex_cost=flex_cost,
                standard_cost=standard_cost,
            )

        except Exception as e:
            return ModelResult(
                model_name=model_name,
                response="",
                response_time=0,
                input_tokens=0,
                output_tokens=0,
                flex_cost=0,
                standard_cost=0,
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
        table = Table(title="AI Model Performance Comparison")

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Response Time", style="magenta")
        table.add_column("Input Tokens", style="green")
        table.add_column("Output Tokens", style="green")
        table.add_column("Flex Cost ($)", style="yellow")
        table.add_column("Standard Cost ($)", style="red")
        table.add_column("Status", style="white")

        for result in results:
            if result.error:
                status = f"[red]Error: {result.error[:30]}...[/red]"
                table.add_row(
                    result.model_name, "N/A", "N/A", "N/A", "N/A", "N/A", status
                )
            else:
                table.add_row(
                    result.model_name,
                    f"{result.response_time:.2f}s",
                    str(result.input_tokens),
                    str(result.output_tokens),
                    f"${result.flex_cost:.6f}",
                    f"${result.standard_cost:.6f}",
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
        self.console.print("[bold blue]AI Model Performance Tester[/bold blue]")
        self.console.print(
            f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.console.print()

        self.display_prompts()

        models_to_test = [
            ("gpt-4o-mini", self.test_openai_model),
            ("gpt-4o", self.test_openai_model),
            ("gpt-5-mini", self.test_openai_model),
            ("o3-mini", self.test_openai_model),
            ("o4-mini", self.test_openai_model),
        ]

        results = []

        with self.console.status("[bold green]Testing models...") as status:
            for model_name, test_func in models_to_test:
                status.update(f"[bold green]Testing {model_name}...")
                result = await test_func(model_name)
                results.append(result)
                self.console.print(f"✓ Completed {model_name}")

        self.console.print()
        self.display_results(results)
        self.display_responses(results)

        # Summary
        successful_results = [r for r in results if not r.error]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x.response_time)
            cheapest_flex = min(successful_results, key=lambda x: x.flex_cost)
            cheapest_standard = min(successful_results, key=lambda x: x.standard_cost)

            self.console.print(
                Panel(
                    f"[green]Fastest Model:[/green] {fastest.model_name} ({fastest.response_time:.2f}s)\n"
                    f"[yellow]Cheapest (Flex):[/yellow] {cheapest_flex.model_name} (${cheapest_flex.flex_cost:.6f})\n"
                    f"[red]Cheapest (Standard):[/red] {cheapest_standard.model_name} (${cheapest_standard.standard_cost:.6f})",
                    title="Summary",
                    border_style="cyan",
                )
            )


async def main():
    """Main function to run the model tester."""
    tester = ModelTester()

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not found in environment variables")

    await tester.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
