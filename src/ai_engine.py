from langchain_ollama import ChatOllama
from pathlib import Path
from memory_engine import store_memory, delete_by_key, retrieve_memory
from fact_extractor import extract_memory

PROFILE_PATH = Path("./profile/profile.md")

def load_profile():
    return PROFILE_PATH.read_text() if PROFILE_PATH.exists() else ""


def build_memory_context(memories):
    lines = []
    for m in memories:
        key = m.metadata.get("key")
        mem_type = m.metadata.get("type")

        if key:
            lines.append(f"{key}: {m.page_content}")
        elif mem_type != "convo":
            lines.append(m.page_content)

    return "\n".join(lines)


def get_ai(memory_context: str):
    profile = load_profile()

    system_prompt = f"""
You are a personal AI that thinks like Kirtan.

PERSONAL PROFILE:
{profile}

KNOWN FACTS (truth only):
{memory_context}

Rules:
- Never bluff
- If fact exists, answer ONLY from it
- If fact does not exist, say you don’t have that information
- Never invent preferences
- Be concise and factual
"""

    ai = ChatOllama(
        model="llama3",
        temperature=0.2,
    )
    return ai, system_prompt


def ask_ai(user_input: str):
    # 1️⃣ Retrieve memory FIRST
    memories = retrieve_memory(user_input)
    memory_context = build_memory_context(memories)

    ai, system_prompt = get_ai(memory_context)

    prompt = f"{system_prompt}\n\nUser: {user_input}"
    response = ai.invoke(prompt).content

    # 2️⃣ Extract structured memory
    extracted = extract_memory(user_input)

    if extracted:
        # Remove conflicting memory
        delete_by_key(extracted["key"])

        store_memory(
            text=f"{extracted['value']}",
            mem_type=extracted["type"],
            keyword=extracted["key"],
            confidence=extracted["confidence"]
        )
    else:
        # Store only as conversation (low confidence)
        store_memory(
            text=user_input,
            mem_type="convo",
            keyword="",
            confidence=0.3
        )

    return response
