import json
import re

import pandas as pd


def convert_to_valid_json_gemini(function_description):
    """
    Converts a function description into a format compatible with Gemini's API requirements.

    Args:
        function_description (list): List of function descriptions with name, description and parameters.

    Returns:
        list: List of formatted function descriptions compatible with Gemini's API.

    Example:
        functions = [{"name": "search", "description": "Search for information",
                         "parameters": {"query": {"type": "str", "description": "Search term"}}}]
        gemini_functions = convert_to_valid_json_gemini(functions)
    """
    output_functions = []
    for i in range(len(function_description)):

        # get parameters
        function_parameters = function_description[i]["parameters"]
        # extract parameters names
        parameters_name = function_parameters.keys()
        # extract function description
        json_formatted_function_description = {
            "name": function_description[i]["name"],
            "description": function_description[i]["description"],
        }

        # create new format paramerters dict
        dict_of_parameters = {"type": "object"}
        properties_dict = {}
        required = []
        for parameter in parameters_name:
            properties_dict.update(
                {
                    parameter: {
                        **python_type_to_json_schema(
                            function_parameters[parameter]["type"].split(",")[0].strip()
                        ),  # get the type and remove if optional or not
                        **{"description": function_parameters[parameter]["description"]},
                    }
                }
            )
        dict_of_parameters["properties"] = properties_dict
        dict_of_parameters["required"] = required

        json_formatted_function_description["parameters"] = dict_of_parameters
        output_functions.append(json_formatted_function_description)
    return output_functions


def python_type_to_json_schema(type_str):
    """
    Converts Python type hints to JSON Schema type definitions.

    Args:
        type_str (str): A string representing a Python type annotation.

    Returns:
        dict: A JSON Schema type definition corresponding to the Python type.

    Example:
        python_type_to_json_schema("List[str]")
        {'type': 'array', 'items': {'type': 'string'}}
        python_type_to_json_schema("Union[int, str]")
        {'anyOf': [{'type': 'integer'}, {'type': 'string'}]}
    """
    type_str = type_str.strip()

    # Mapping of base Python types to JSON Schema types
    base_types = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "Any": {},
        "dict": "object",
        "Dict": "object",
        "set": "array",  # sets are converted to arrays too
    }

    # Match List[...] recursively
    list_match = re.match(r"List\[(.+)\]$", type_str)
    tuple_match = re.match(r"Tuple\[(.+)\]$", type_str)
    union_match = re.match(r"Union\[(.+)\]$", type_str)

    if list_match:
        inner_type = list_match.group(1).strip()
        return {"type": "array", "items": python_type_to_json_schema(inner_type)}

    elif tuple_match:
        # Gemini doesn't support fixed-length tuples, treat as general array
        inner_types = tuple_match.group(1).split(",")
        first_type = inner_types[0].strip() if inner_types else None
        return {
            "type": "array",
            "items": python_type_to_json_schema(first_type) if first_type else {},
        }

    elif union_match:
        options = [python_type_to_json_schema(t.strip()) for t in union_match.group(1).split(",")]
        return {"anyOf": options}

    # Explicitly handle 'list' or 'List' with unknown item types
    if type_str.lower() in ("list", "set"):
        return {"type": "array", "items": {"type": "object"}}

    if type_str in base_types:
        json_type = base_types[type_str]
        return {"type": json_type} if isinstance(json_type, str) else json_type

    # Fallback for unknown or complex types (e.g., Callable, CustomType, Optional[...])
    if type_str.startswith("Callable") or type_str.startswith("Optional"):
        return {"type": "string"}

    # Final fallback to generic object
    return {"type": "object"}


def convert_to_valid_json(function_description):
    """
    Converts a function description to a JSON format compatible with OpenAI and Claude APIs.
    This version includes additional features like handling optional parameters and defaults.

    Args:
        function_description (list): List of function descriptions with name, description and parameters.

    Returns:
        list: List of formatted function descriptions with proper JSON Schema structure.

    Example:
        functions = [{"name": "search", "description": "Search for information",
                         "parameters": {"query": {"type": "str, optional", "description": "Search term", "default": ""}}}]
        json_functions = convert_to_valid_json(functions)
    """
    output_functions = []

    for i in range(len(function_description)):
        try:
            func = function_description[i]
            function_parameters = func["parameters"]
            parameter_names = function_parameters.keys()

            json_formatted_function_description = {
                "name": func["name"],
                "description": func["description"],
            }

            dict_of_parameters = {"type": "object", "properties": {}, "required": []}

            for param in parameter_names:
                param_info = function_parameters[param]
                raw_type = param_info["type"].split(",")[0].strip()
                is_optional = (
                    len(param_info["type"].split(",")) > 1
                    and "optional" in param_info["type"].split(",")[1]
                )

                param_schema = python_type_to_json_schema(raw_type)
                param_schema["description"] = param_info["description"]

                # Safe default injection only for primitive types
                if is_optional:
                    default_val = param_info.get("default", None)

                    if default_val is not None:
                        if isinstance(default_val, (int, float, bool)):
                            param_schema["default"] = default_val
                        elif isinstance(default_val, str):
                            # Convert string default to enum if safe
                            param_schema["enum"] = [default_val]
                            # optionally: also include description of allowed value
                    # else: fallback enum or leave out
                else:
                    dict_of_parameters["required"].append(param)

                dict_of_parameters["properties"][param] = param_schema

            dict_of_parameters["additionalProperties"] = False
            json_formatted_function_description["parameters"] = dict_of_parameters
            output_functions.append(json_formatted_function_description)

        except Exception as e:
            print(f"Skipping function {func.get('name', 'unknown')} due to: {e}")
            continue

    return output_functions


def load_data(path="data/xlam_function_calling_60k.json", backend="gemini"):
    """
    Loads a dataset of function descriptions and converts them to formats compatible with
    different AI APIs (Gemini, OpenAI, Claude).

    Args:
        path (str): Path to the JSON file containing function descriptions.
        backend (str): Specifies the AI backend format ("gemini", "openai", or "claude").

    Returns:
        pandas.DataFrame: DataFrame containing the original data with added columns for
                         formatted function descriptions compatible with specified backends.

    Example:
        df = load_data("path/to/functions.json", backend="openai")
    """
    df_data = pd.read_json(path)
    # load json columns
    df_data["answers"] = df_data["answers"].apply(lambda x: json.loads(x))
    df_data["tools"] = df_data["tools"].apply(lambda x: json.loads(x))

    # convert json to be compatible with openAI / Gemini api
    ##df_data["tools"] = df_data["tools"].apply(lambda x : convert_to_valid_json(x))
    # speed up version
    if backend == "gemini":
        tools_cleaned = [convert_to_valid_json_gemini(x) for x in df_data["tools"]]
    else:
        tools_cleaned = [convert_to_valid_json(x) for x in df_data["tools"]]

    df_data["tools"] = tools_cleaned
    df_data.dropna()

    if backend == "gemini":
        return df_data

    tools_cleaned_open_ai = [
        [
            {
                "type": "function",
                "name": tool["name"],
                "strict": True,
                "description": tool["description"],
                "parameters": {**tool["parameters"], "additionalProperties": False},
            }
            for tool in gemini_tool
        ]
        for gemini_tool in tools_cleaned
    ]

    tools_cleaned_claude = [
        [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {**tool["parameters"], "additionalProperties": False},
            }
            for tool in gemini_tool
        ]
        for gemini_tool in tools_cleaned
    ]

    df_data["tools_open_ai"] = tools_cleaned_open_ai

    df_data["tools_claude"] = tools_cleaned_claude
    return df_data
