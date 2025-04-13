from google.genai import types


def gemini_call(client, query, tools):
    """
    Makes a function call to the Gemini API with the given query and tools.

    Args:
        client: The Gemini client instance used to access the API.
        query: The user query or prompt to send to the model.
        tools: A list of function declarations that the model can call.

    Returns:
        function_call: The function call object if the model called a function,
                      or None if no function was called.

    Example:
        tools = [{"name": "search", "description": "Search for information",
                     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]
        function_call = gemini_call(client, "What's the weather in New York?", tools)
    """
    # Configure the client and tools
    tools = types.Tool(function_declarations=tools)
    config = types.GenerateContentConfig(tools=[tools])

    # Send request with function declarations
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=query,
        config=config,
    )

    # Check for a function call
    try:
        function_call = response.candidates[0].content.parts[0].function_call
    except Exception as _:
        function_call = None

    return function_call


def openai_call(client, query, tools):
    """
    Makes a function call to the OpenAI API with the given query and tools.

    Args:
        client: The OpenAI client instance used to access the API.
        query: The user query or prompt to send to the model.
        tools: A list of tool definitions in OpenAI format.

    Returns:
        tool_call: The tool call object if the model called a tool,
                  or None if no tool was called.

    Example:
        tools = [{"type": "function", "function": {"name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]
        tool_call = openai_call(client, "What's the weather in New York?", tools)
    """
    response = client.responses.create(
        model="gpt-4o", input=[{"role": "user", "content": query}], tools=tools
    )

    try:
        tool_call = response.output[0]
    except Exception:
        tool_call = None

    return tool_call


def claude_call(client, query, tools):
    """
    Makes a function call to the Claude API with the given query and tools.

    Args:
        client: The Anthropic client instance used to access the API.
        query: The user query or prompt to send to the model.
        tools: A list of tool definitions in OpenAI-compatible format.

    Returns:
        function_call: The tool use object if the model used a tool,
                      or None if no tool was used.

    Example:
        tools = [{"type": "function", "function": {"name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]
        function_call = claude_call(client, "What's the weather in New York?", tools)
    """
    # Send request with function tool usage
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0,
        tools=tools,  # expects tools in OpenAI-compatible format
        messages=[{"role": "user", "content": query}],
    )

    function_call = None
    for content_block in response.content:
        if content_block.type == "tool_use":
            function_call = content_block
            break

    return function_call
