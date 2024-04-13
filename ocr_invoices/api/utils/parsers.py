import re
from dateutil.parser import parse


# Globla utility parse class for parsing different classes
class Parsers:
    # For total parsing
    def parse_totals(text: str) -> float | None:
        return "".join(e for e in text if e.isnumeric() or e in [",", "."])

    # For date parsing
    def extract_date(date_str: str) -> str | None:
        # Adjusted regex pattern to ignore apostrophes in the day part of the date
        date_pattern = re.compile(
            r"""
            (\b                      # Word boundary to ensure we're starting at the beginning of a date component
            (?:"?\d{1,2}"?['’]?\s)?  # Optional day with optional quotes/apostrophe, followed by optional space
            (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*  # Month abbreviations, case-insensitive
            \s*,?\s*                 # Optional comma and whitespace
            \d{2,4}                  # Year
            \b)                      # Word boundary to ensure we're ending at the end of a date component
            |
            (\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b)  # Matches numerical date formats e.g., DD/MM/YYYY
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        if not isinstance(date_str, str):
            return None

        matches = date_pattern.findall(date_str)
        for date_tuple in matches:
            date_str = max(date_tuple, key=len)  # Extract the non-empty match
            # Normalize the date string by removing apostrophes
            date_str = re.sub(r"['’]", "", date_str)
            try:
                # Attempt to parse the date string
                return parse(date_str, dayfirst=True, fuzzy=True).date().isoformat()
            except ValueError:
                return None

    # For general text parsing
    def parse_general(text: str) -> str:
        # Use a list comprehension to filter out non-alphanumeric characters and
        # spaces, and remove line breaks
        cleaned_text = "".join(
            [char for char in text if char.isalnum() or char.isspace()]
        )

        # Replace line breaks with spaces to ensure words are not concatenated together
        cleaned_text = cleaned_text.replace("\n", " ")

        return cleaned_text
