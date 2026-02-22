import tiktoken

def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Counts the number of tokens in a text string using the specified encoding.

    Args:
        string: The text to count tokens in.
        encoding_name: The name of the encoding to use (e.g., "cl100k_base").

    Returns:
        The number of tokens in the string.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError:
        print(f"Warning: Encoding '{encoding_name}' not found. Using 'cl100k_base' as default.")
        encoding = tiktoken.get_encoding("cl100k_base")
        
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == '__main__':
    # Example Usage
    example_text = "Hello world! This is a test of the token counter."
    token_count = count_tokens(example_text)
    print(f"The text has {token_count} tokens.")

    example_code = """
import tiktoken

def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
"""
    token_count_code = count_tokens(example_code)
    print(f"The code snippet has {token_count_code} tokens.")
