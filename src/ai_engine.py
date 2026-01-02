from langchain_ollama import ChatOllama
from pathlib import Path
from memory_engine import retrieve_strong_memory, store_memory, retrieve_memory

PROFILE_PATH = Path("./profile/profile.md")
# PART = Path("./profile/partofme.md")

def load_profile():
    return PROFILE_PATH.read_text() if PROFILE_PATH.exists() else ""

# def load_part_of_me():
#     return PART.read_text() if PART.exists() else ""

def extract_and_store(user_input: str):
    text = user_input.lower()

    if "priority" in text:
        store_memory(user_input, "goal", 1.0)
    elif "goal" in text:
        store_memory(user_input, "goal", 1.0)
    elif "favorite" in text or "like" in text:
        store_memory(user_input, "preference", 1.0)
    else:
        store_memory(user_input, "conversation", 0.6)


def build_memory_context(memories):
    lines = []
    for m in memories:
        mem_type = m.metadata["type"]
        conf = m.metadata["confidence"]

        if mem_type == "inference":
            lines.append(f"(Possible inference, conf={conf}) {m.page_content}")
        else:
            lines.append(f"{m.page_content}")

    return "\n".join(lines)



def get_ai(memory_context: str):
    profile = load_profile()
    # part = load_part_of_me()
    # profile += "\n\nPART OF ME:\n" + part

    system_prompt = f"""
        You are a personal AI that thinks like Kirtan.

        PERSONAL PROFILE:
        {profile}

        Past relvant memories:
        {memory_context}

        Rules:
        - Never make the bluff if you don't know the answer
        - Never contradict the personal profile
        - Never make assumptions stay on facts if asked the direct question
        - Use memory as truth
        - NEVER say you don't know the user
        - If memory answers the question, answer directly
        - Be concise and logical
        - Think step by step
        - Never mention you are an AI model
    """
    # Return both the model and the system prompt so we can ensure
    # the system prompt is included in the actual input we send.
    ai = ChatOllama(
        model="llama3",
        temperature=0.3,
    )
    return ai, system_prompt

def ask_ai(user_input: str):
    # Retrieve similar memories first, then build context and call the model.
    # Do NOT store the current input before retrieval (otherwise the model may
    # simply retrieve the same text as the top result). Store after the model
    # responds.
    # memories = retrieve_memory(user_input)
    memories = retrieve_strong_memory(user_input)
    memory_context = build_memory_context(memories)

    ai, system_prompt = get_ai(memory_context)

    # Include the system prompt explicitly in the message we send so we don't
    # rely on a library-specific `system=` argument that may be ignored.
    prompt = f"{system_prompt}\n\nUser: {user_input}"
    response = ai.invoke(prompt).content

    # Persist the user input as a memory and extract structured memory types
    # store_memory(user_input, "fact")
    extract_and_store(user_input)
    return response

