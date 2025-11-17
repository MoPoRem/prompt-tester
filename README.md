# AI Model Performance Tester

A minimal CLI application to test and compare different AI models for cost and performance analysis.

## Features

- Tests multiple AI models: GPT-4o-mini, GPT-4o, GPT-4 (as GPT-5 placeholder), Claude-3.5-Sonnet
- Tracks response time, token usage, and cost
- Clean CLI interface with rich formatting
- Fixed prompts for consistent testing
- Detailed response comparison

## Setup

1. **Install dependencies:**
   ```bash
   # Windows
   venv\Scripts\python setup.py
   
   # Unix/Linux/Mac
   venv/bin/python setup.py
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Run the tester:**
   ```bash
   # Windows
   venv\Scripts\python model_tester.py
   
   # Unix/Linux/Mac
   venv/bin/python model_tester.py
   ```

## API Keys Required

- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models

## Output

The app displays:
- Test prompts being used
- Performance comparison table
- Full responses from each model
- Summary with fastest and cheapest models

## Models Tested

- **gpt-4o-mini** - Fast and cost-effective
- **gpt-4o** - Balanced performance
- **gpt-4** - High capability (placeholder for GPT-5)
- **claude-3-5-sonnet** - Anthropic's latest

## Cost Tracking

Costs are calculated based on current API pricing (per 1K tokens) and may need updates as pricing changes.