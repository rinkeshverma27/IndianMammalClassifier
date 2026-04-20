from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from chatbot import get_chat_response, init_gemini_model
from inference import (
    analyze_image,
    load_checkpoint,
    load_inference_config,
    load_mammal_detector,
    load_species_info,
)


st.set_page_config(page_title="Indian Mammal Classifier", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(209, 231, 221, 0.85), transparent 28%),
                radial-gradient(circle at bottom right, rgba(245, 205, 166, 0.75), transparent 24%),
                linear-gradient(135deg, #f4efe6 0%, #e7dcc8 100%);
        }
        .block-container {
            max-width: 1400px;
            padding-top: 1.4rem;
            padding-bottom: 1.2rem;
        }
        .app-title {
            border: 2px solid #3f3022;
            background: rgba(255, 252, 246, 0.82);
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 6px 6px 0 rgba(63, 48, 34, 0.12);
        }
        .app-title h1 {
            margin: 0;
            font-size: 2.2rem;
            color: #183a2e;
        }
        .app-title p {
            margin: 0.35rem 0 0;
            color: #5e4836;
            font-size: 1rem;
        }
        .panel-title {
            margin: 0 0 0.85rem;
            color: #7b341e;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            font-weight: 700;
            font-size: 0.95rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 251, 245, 0.9);
            border: 2px solid #3f3022;
            box-shadow: 8px 8px 0 rgba(63, 48, 34, 0.08);
        }
        .image-placeholder {
            border: 2px dashed #90a489;
            min-height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #62806c;
            background: linear-gradient(135deg, rgba(217, 232, 223, 0.35), rgba(255, 248, 239, 0.45));
            text-align: center;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            border: 1.5px solid #6d5845;
            background: rgba(246, 239, 227, 0.75);
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.75rem;
        }
        .metric-card strong {
            color: #183a2e;
        }
        .species-name {
            margin: 0 0 0.35rem;
            color: #183a2e;
            font-size: 1.65rem;
        }
        .species-summary {
            color: #5c493a;
            margin-bottom: 0.85rem;
        }
        .chat-shell {
            border: 1.5px solid #6d5845;
            background: rgba(249, 244, 236, 0.72);
            min-height: 420px;
            max-height: 420px;
            overflow-y: auto;
            padding: 0.9rem;
            margin-bottom: 0.8rem;
        }
        .chat-bubble-user {
            background: #f7d7d0;
            border: 1.5px solid #a8594c;
            color: #5f2c24;
            padding: 0.7rem 0.85rem;
            border-radius: 18px 18px 4px 18px;
            margin: 0 0 0.75rem auto;
            width: fit-content;
            max-width: 90%;
        }
        .chat-bubble-bot {
            background: #d7ece4;
            border: 1.5px solid #4b7e6b;
            color: #1f4e3f;
            padding: 0.7rem 0.85rem;
            border-radius: 18px 18px 18px 4px;
            margin: 0 0 0.75rem 0;
            width: fit-content;
            max-width: 90%;
        }
        .status-chip {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border: 1.5px solid #6d5845;
            background: #efe1cf;
            color: #5c493a;
            margin-bottom: 0.75rem;
            font-size: 0.84rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    model, classes, config = load_checkpoint()
    detector_model, detector_transform, mammal_indices = load_mammal_detector()
    inference_config = load_inference_config()
    species_info = load_species_info(Path("app/species_info.json"))
    gemini_model = init_gemini_model()
    return (
        model,
        classes,
        config,
        detector_model,
        detector_transform,
        mammal_indices,
        inference_config,
        species_info,
        gemini_model,
    )


def init_state() -> None:
    st.session_state.setdefault("chat_history", [])


def reset_chat_if_species_changed(active_species: str | None) -> None:
    previous_species = st.session_state.get("chat_species")
    if previous_species != active_species:
        st.session_state["chat_species"] = active_species
        st.session_state["chat_history"] = []


def render_header() -> None:
    st.markdown(
        """
        <div class="app-title">
            <h1>Indian Mammal Classifier</h1>
            <p>Upload an image, review the top prediction, explore species notes, and ask follow-up questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_bars(predictions: list[dict]) -> None:
    st.markdown('<div class="panel-title">Top 3 Predictions</div>', unsafe_allow_html=True)
    if not predictions:
        st.info("No class predictions available yet.")
        return

    for pred in predictions:
        st.markdown(
            f"<div class='metric-card'><strong>{pred['label']}</strong><br>{pred['confidence'] * 100:.2f}% confidence</div>",
            unsafe_allow_html=True,
        )
        st.progress(float(pred["confidence"]))


def render_info_panel(active_species: str | None, species_info: dict, analysis: dict | None) -> None:
    st.markdown('<div class="panel-title">Species Guide</div>', unsafe_allow_html=True)
    if analysis is None:
        st.markdown(
            "<div class='image-placeholder'>Upload an image to view mammal details here.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div class='status-chip'>Mammal filter score: {analysis['mammal_score'] * 100:.2f}%</div>",
        unsafe_allow_html=True,
    )

    if analysis["status"] == "rejected_non_mammal":
        st.error("Unknown / Not an Indian mammal")
        st.write(analysis["message"])
        return

    if active_species is None:
        st.warning("The image looks mammal-like, but the classifier is not confident enough to assign one of the 14 classes.")
        return

    info = species_info.get(active_species)
    if not info:
        st.warning("No species information available for this class.")
        return

    st.markdown(f"<h2 class='species-name'>{active_species}</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='species-summary'>{info['summary']}</div>", unsafe_allow_html=True)

    fields = [
        ("Scientific Name", info["scientific_name"]),
        ("IUCN Status", info["iucn_status"]),
        ("Habitat", info["habitat"]),
        ("Distribution", info["distribution"]),
        ("Diet", info["diet"]),
        ("Fun Fact", info["fun_fact"]),
    ]
    for label, value in fields:
        st.markdown(
            f"<div class='metric-card'><strong>{label}</strong><br>{value}</div>",
            unsafe_allow_html=True,
        )


def render_chat_history() -> None:
    st.markdown('<div class="panel-title">Ask Wildlife Guide</div>', unsafe_allow_html=True)
    chat_history = st.session_state.get("chat_history", [])
    if not chat_history:
        st.markdown(
            """
            <div class="chat-shell">
                <div class="chat-bubble-bot">Ask about habitat, diet, conservation status, or fun facts.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    html = ["<div class='chat-shell'>"]
    for item in chat_history:
        bubble_class = "chat-bubble-user" if item["role"] == "user" else "chat-bubble-bot"
        html.append(f"<div class='{bubble_class}'>{item['content']}</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_chat_panel(active_species: str | None, species_info: dict, gemini_model: object | None) -> None:
    render_chat_history()
    help_text = (
        f"Chat is focused on: {active_species}" if active_species else "Chat is idle until a species is confidently identified."
    )
    st.caption(help_text)

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Ask anything", placeholder="What does it eat? Where is it found?")
        send = st.form_submit_button("Ask")

    clear = st.button("Clear Chat", use_container_width=True)
    if clear:
        st.session_state["chat_history"] = []
        st.rerun()

    if send and question.strip():
        st.session_state["chat_history"].append({"role": "user", "content": question.strip()})
        response = get_chat_response(question, active_species, species_info, gemini_model)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.rerun()


def main() -> None:
    inject_styles()
    init_state()
    render_header()

    try:
        (
            model,
            classes,
            config,
            detector_model,
            detector_transform,
            mammal_indices,
            inference_config,
            species_info,
            gemini_model,
        ) = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    left_col, center_col, right_col = st.columns([1.05, 1.0, 1.05], gap="medium")

    uploaded_file = None
    analysis = None
    active_species = None

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Upload & Result</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Select file",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="visible",
            )

            if uploaded_file is None:
                st.markdown(
                    "<div class='image-placeholder'>Image of the mammal will appear here.</div>",
                    unsafe_allow_html=True,
                )
                render_prediction_bars([])
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded image", use_container_width=True)
                analysis = analyze_image(
                    image=image,
                    classifier_model=model,
                    classes=classes,
                    classifier_config=config,
                    detector_model=detector_model,
                    detector_transform=detector_transform,
                    mammal_indices=mammal_indices,
                    inference_config=inference_config,
                )
                if analysis["status"] == "accepted":
                    active_species = analysis["predictions"][0]["label"]
                render_prediction_bars(analysis["predictions"])

    reset_chat_if_species_changed(active_species)

    with center_col:
        with st.container(border=True):
            render_info_panel(active_species, species_info, analysis)

    with right_col:
        with st.container(border=True):
            render_chat_panel(active_species, species_info, gemini_model)


if __name__ == "__main__":
    main()
