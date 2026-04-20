"""
Chat helpers for the Streamlit dashboard.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def init_gemini_model() -> Optional[object]:
    load_environment()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    if genai is None:
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def build_species_context(species_name: str, species_info: dict) -> str:
    info = species_info.get(species_name, {})
    if not info:
        return ""

    lines = [
        f"Species: {species_name}",
        f"Scientific name: {info.get('scientific_name', 'Unknown')}",
        f"IUCN status: {info.get('iucn_status', 'Unknown')}",
        f"Habitat: {info.get('habitat', 'Unknown')}",
        f"Distribution: {info.get('distribution', 'Unknown')}",
        f"Diet: {info.get('diet', 'Unknown')}",
        f"Summary: {info.get('summary', 'Unknown')}",
        f"Fun fact: {info.get('fun_fact', 'Unknown')}",
    ]
    return "\n".join(lines)


def build_dataset_answer(question: str, species_name: Optional[str], species_info: dict) -> Optional[str]:
    if not species_name:
        return None

    info = species_info.get(species_name)
    if not info:
        return None

    question_lower = question.lower()
    if any(word in question_lower for word in {"diet", "eat", "food"}):
        return f"{species_name} diet: {info['diet']}"
    if any(word in question_lower for word in {"habitat", "live", "forest", "where", "distribution"}):
        return f"{species_name} habitat and distribution: {info['habitat']} Found in {info['distribution']}"
    if any(word in question_lower for word in {"iucn", "status", "endangered", "threat"}):
        return f"{species_name} IUCN status: {info['iucn_status']}"
    if any(word in question_lower for word in {"scientific", "name"}):
        return f"{species_name} scientific name: {info['scientific_name']}"
    if any(word in question_lower for word in {"fun", "fact"}):
        return info["fun_fact"]
    if any(word in question_lower for word in {"summary", "describe", "description", "about"}):
        return (
            f"{species_name}: {info['summary']} Habitat: {info['habitat']} "
            f"Diet: {info['diet']} IUCN status: {info['iucn_status']}"
        )

    return None


def local_chat_response(question: str, species_name: Optional[str], species_info: dict) -> str:
    if not species_name:
        return "Upload an image and get a valid mammal prediction first, then I can answer questions about that animal."

    dataset_answer = build_dataset_answer(question, species_name, species_info)
    if dataset_answer is not None:
        return dataset_answer

    info = species_info.get(species_name)
    if info is None:
        return f"I do not have species notes for {species_name} yet."

    return (
        f"{species_name}: {info['summary']} Habitat: {info['habitat']} "
        f"Diet: {info['diet']} IUCN status: {info['iucn_status']}"
    )


def get_chat_response(
    question: str,
    species_name: Optional[str],
    species_info: dict,
    gemini_model: Optional[object],
) -> dict:
    if not question.strip():
        return {
            "text": "Ask a question about the detected mammal.",
            "source": "system",
            "status": "empty_question",
        }

    dataset_answer = build_dataset_answer(question, species_name, species_info)
    if dataset_answer is not None:
        return {
            "text": dataset_answer,
            "source": "dataset",
            "status": "ok",
        }

    if gemini_model is None:
        return {
            "text": local_chat_response(question, species_name, species_info),
            "source": "local",
            "status": "gemini_unavailable",
        }

    context = build_species_context(species_name, species_info) if species_name else ""
    prompt = (
        "You are a wildlife guide for an Indian mammal classifier app.\n"
        "Use the provided species context first when it is relevant.\n"
        "If the answer is not present in that context, you may answer from general biological knowledge.\n"
        "When using general knowledge, say that it is a general estimate and may vary.\n"
        "Do not browse the internet and do not claim to use live sources.\n"
        "Answer briefly and clearly.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        cleaned = text.strip()
        if cleaned:
            return {
                "text": cleaned,
                "source": "gemini",
                "status": "ok",
            }
        return {
            "text": local_chat_response(question, species_name, species_info),
            "source": "local",
            "status": "empty_gemini_response",
        }
    except Exception as exc:
        return {
            "text": local_chat_response(question, species_name, species_info),
            "source": "local",
            "status": f"gemini_error: {type(exc).__name__}",
        }
