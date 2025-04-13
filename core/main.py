import json
import os
import pickle

import anthropic
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from tqdm import tqdm

from core.api_calls import claude_call, gemini_call, openai_call
from core.data_preprocessing import load_data
from core.evaluation import eval

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def start_clients():

    opena_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    return opena_ai_client, gemini_client, claude_client


def main():
    df_data = load_data(path="data/xlam_function_calling_60k.json")
    df_data_openai = load_data(path="data/xlam_function_calling_60k.json", backend="openai")
    with open("data/correct_i.pkl", "rb") as f:
        correct_i = pickle.load(f)
    df_data = df_data.iloc[correct_i]
    df_data_openai = df_data_openai.iloc[correct_i]

    all_calls_gemini = []
    calls_gemini = []
    params_gemini = []
    failed_return_function_gemini = 0

    all_calls_open_ai = []
    calls_open_ai = []
    params_open_ai = []
    failed_return_function_open_ai = 0

    all_calls_claude = []
    calls_claude = []
    params_claude = []
    failed_return_function_claude = 0

    opena_ai_client, gemini_client, claude_client = start_clients()

    dic_results = {}

    for _, row in tqdm(df_data.iterrows()):
        function_call = gemini_call(gemini_client, row.query, row.tools)
        all_calls_gemini.append(function_call)
        if function_call is not None and hasattr(function_call, "name"):
            correct_call, correct_params = eval(function_call, row.answers[0])
        else:
            correct_call, correct_params = 0, 0
            failed_return_function_gemini += 1
        calls_gemini.append(correct_call)
        params_gemini.append(correct_params)
    dic_results["Gemini"] = {
        "calls": calls_gemini,
        "params": params_gemini,
        "failed_calls": failed_return_function_gemini,
    }

    with open("results/dic_results.json", "w") as f:
        json.dump(dic_results, f, indent=4)

    for _, row in tqdm(df_data_openai.iterrows()):
        function_call = openai_call(opena_ai_client, row.query, row.tools_open_ai)
        all_calls_open_ai.append(function_call)
        if function_call is not None and hasattr(function_call, "name"):
            correct_call, correct_params = eval(function_call, row.answers[0], backend="openai")
        else:
            correct_call, correct_params = 0, 0
            failed_return_function_open_ai += 1
        calls_open_ai.append(correct_call)
        params_open_ai.append(correct_params)

    dic_results["OpenAI"] = {
        "calls": calls_open_ai,
        "params": params_open_ai,
        "failed_calls": failed_return_function_open_ai,
    }

    with open("results/dic_results.json", "w") as f:
        json.dump(dic_results, f, indent=4)

    for _, row in tqdm(df_data_openai.iterrows()):
        function_call = claude_call(
            claude_client, row.query, list({f["name"]: f for f in row.tools_claude}.values())
        )
        all_calls_claude.append(function_call)
        if function_call is not None and hasattr(function_call, "name"):
            correct_call, correct_params = eval(function_call, row.answers[0], backend="claude")
        else:
            correct_call, correct_params = 0, 0
            failed_return_function_claude += 1
        calls_claude.append(correct_call)
        params_claude.append(correct_params)

    dic_results["Claude"] = {
        "calls": calls_claude,
        "params": params_claude,
        "failed_calls": failed_return_function_claude,
    }

    with open("results/dic_results_final.json", "w") as f:
        json.dump(dic_results, f, indent=4)


if __name__ == "__main__":
    main()
