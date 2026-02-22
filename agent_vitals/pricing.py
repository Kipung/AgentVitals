# Pricing data per million tokens (input and output)
# All prices are in USD.
# Source: Keep this updated from official provider websites.

MODEL_PRICING = {
    # OpenAI
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    
    # Anthropic
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},

    # Google
    "gemini-1.5-pro": {"input": 7.00, "output": 21.00},

    # Default / Fallback
    "default": {"input": 1.00, "output": 1.00},
}

def get_price(model_name: str, token_type: str) -> float:
    """
    Gets the price for a given model and token type.
    
    Args:
        model_name: The name of the model.
        token_type: 'input' or 'output'.
        
    Returns:
        The price per million tokens.
    """
    model_key = "default"
    for key in MODEL_PRICING:
        if key in model_name:
            model_key = key
            break
            
    return MODEL_PRICING[model_key][token_type]
