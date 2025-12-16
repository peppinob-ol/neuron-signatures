"""ASCII-safe token string handling for Windows cp1252 console compatibility."""

import re
from typing import List


def sanitize_token(token: str) -> str:
    """
    Convert a single token string to ASCII-safe representation.
    
    - Replaces SentencePiece underscore marker with space.
    - Replaces non-ASCII characters with <?>.
    - Strips control characters.
    
    Args:
        token: Raw token string from tokenizer.
        
    Returns:
        ASCII-safe string suitable for console output and storage.
    """
    # Replace SentencePiece leading space marker
    result = token.replace("\u2581", " ")
    
    # Replace newlines/tabs with readable markers
    result = result.replace("\n", "<n>")
    result = result.replace("\r", "<r>")
    result = result.replace("\t", "<t>")
    
    # Strip other control characters (0x00-0x1F except already handled)
    result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", result)
    
    # Replace any remaining non-ASCII with <?>
    sanitized = []
    for ch in result:
        if ord(ch) < 128:
            sanitized.append(ch)
        else:
            sanitized.append("<?>")
    
    return "".join(sanitized)


def sanitize_tokens(tokens: List[str]) -> List[str]:
    """
    Sanitize a list of token strings for ASCII-safe output.
    
    Args:
        tokens: List of raw token strings from tokenizer.
        
    Returns:
        List of ASCII-safe token strings.
    """
    return [sanitize_token(t) for t in tokens]


def safe_print(msg: str) -> None:
    """
    Print a message, replacing any non-ASCII characters to avoid cp1252 errors.
    
    Args:
        msg: Message to print.
    """
    safe_msg = "".join(ch if ord(ch) < 128 else "?" for ch in msg)
    print(safe_msg)


