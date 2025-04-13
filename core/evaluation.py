import json
from typing import Any, Dict


def eval(function_call: Any, ground_true_answer: Dict[str, str], backend="gemini"):
    """
    Evaluates if a function call matches the expected ground truth answer.

    Args:
        function_call: A function call object from an AI model response.
            - For Gemini: function_call object with name and args attributes
            - For OpenAI: function call object with name and arguments (JSON string) attributes
            - For Claude: tool_use object with name and input attributes
        ground_true_answer: Dictionary containing expected function details with keys:
            - "name": Expected function name to be called
            - "arguments": Expected arguments that should be passed to the function
        backend: String indicating which AI backend was used.
                 Supported values: "gemini", "openai", "claude"

    Returns:
        tuple: (correct_call, correct_params)
            - correct_call (int): 1 if function name matches ground truth, 0 otherwise
            - correct_params (int): 1 if both name and parameters match ground truth, 0 otherwise

    Raises:
        NotImplementedError: If an unsupported backend is specified
    """
    correct_call = int(function_call.name == ground_true_answer["name"])

    if backend == "gemini":
        args = function_call.args
    elif backend == "openai":
        args = json.loads(function_call.arguments)
    elif backend == "claude":
        args = function_call.input
    else:
        raise NotImplementedError(f"Backend '{backend}' is not supported")

    correct_params = int(correct_call and args == ground_true_answer["arguments"])
    return correct_call, correct_params
