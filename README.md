# 🔍 Benchmark Function Calling

## Setup

1. Create a virtual environment:
   ```bash
    python -m venv env_function_calling
    source env_function_calling/bin/activate

2. Install the dependencies:
    ```bash
        pip install -r requirements.txt
3. Add your API keys in a .env file in the project root directory.

## Example .env file
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key

## Run the Benchmark
Run the main script to execute all function-calling benchmarks:
    ```bash
        python -m core.main  

Benchmark results will be saved in the results/ directory in .json format. This may take some times (around 45mn)

## Visualize Results
Explore and analyze the benchmark results using the notebook: