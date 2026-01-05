import re

FACT_RULES = [
    (r"i am (\d+)\s*years? old", "fact", "age"),
    (r"my age is (\d+)", "fact", "age"),
    (r"my name is (\w+)", "fact", "name"),
    (r"my favorite color is (\w+)", "preference", "fav_color"),
]

def extract_memory(text):
    text = text.lower()

    for pattern, mem_type, key in FACT_RULES:
        match = re.search(pattern, text)
        if match:
            return {
                "type": mem_type,
                "key": key,
                "value": match.group(1),
                "confidence": 0.95
            }
    return None
