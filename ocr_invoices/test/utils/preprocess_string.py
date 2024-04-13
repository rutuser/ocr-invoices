def preprocess_string(s: str):
    """
    Preprocesses the input string by removing spaces, newline characters,
    and converting to lowercase.
    """
    return s.replace(" ", "").replace("\n", "").lower()
