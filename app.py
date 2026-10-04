"""Run with: python -m streamlit run app.py"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from rag import Settings, build_index, fetch_transcript, friendly_error, parse_video_id, timestamp

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

st.set_page_config(page_title="Video Notes · YouTube Chat", page_icon="▶️", layout="wide")


def setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        return default


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Transcript sources · {len(sources)} passages"):
        for index, source in enumerate(sources, 1):
            st.markdown(
                f"**[{index}] [{timestamp(source['start'])} – {timestamp(source['end'])}]({source['url']})**"
            )
            st.text(source["text"])


def main() -> None:
    st.session_state.setdefault("engine", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("loaded_languages", ())

    with st.sidebar:
        st.markdown("### ▶ Video Notes")
        st.caption("A conversation with what you watch.")
        st.divider()
        st.markdown("**API connections**")
        require_user_keys = setting("REQUIRE_USER_KEYS", "false").lower() in {"1", "true", "yes"}
        entered_openrouter_key = st.text_input(
            "OpenRouter API key", type="password",
            help="Used to embed the transcript and your questions.",
        ).strip()
        entered_google_key = st.text_input(
            "Google API key",
            type="password", help="Used by Gemini to answer questions.",
        ).strip()
        # Server credentials must never be sent to browser widgets, even masked ones.
        openrouter_key = entered_openrouter_key or ("" if require_user_keys else setting("OPENROUTER_API_KEY"))
        google_key = entered_google_key or ("" if require_user_keys else setting("GOOGLE_API_KEY", setting("GEMINI_API_KEY")))
        if require_user_keys:
            st.caption("This public app uses your own API keys for this session.")
        if openrouter_key and google_key:
            st.caption("● Keys configured · validated when you make a request")
        else:
            st.caption("Add both keys here." if require_user_keys else "Add both keys here or in your .env file.")
        with st.expander("Model & transcript settings"):
            embedding_model = st.text_input("Embedding model", value=setting("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")).strip()
            gemini_model = st.text_input("Gemini model", value=setting("GEMINI_MODEL", "gemini-3.5-flash")).strip()
            language_input = st.text_input("Preferred caption languages", value="en", help="Comma-separated language codes, e.g. en,hi. Falls back to another available caption track.")
        st.divider()
        st.markdown("**How it works**")
        st.caption("1. Paste a YouTube link.\n\n2. Load its captions.\n\n3. Ask a question and explore the sources.")
        st.caption("Answers use the transcript. Visuals and unspoken details are outside its scope.")

    settings = Settings(openrouter_key, google_key, embedding_model, gemini_model)
    languages = tuple(code.strip().lower() for code in language_input.split(",") if code.strip()) or ("en",)
    raw_directory = Path(setting("CHROMA_DIR", "data/chroma"))
    directory = raw_directory if raw_directory.is_absolute() else APP_DIR / raw_directory

    st.caption("WATCH LESS. UNDERSTAND MORE.")
    st.title("Your video. Your questions.")
    st.markdown("Turn a YouTube video into a conversation. Get clear answers with the moments that support them.")

    with st.form("load_video"):
        input_column, button_column = st.columns([5, 1], vertical_alignment="bottom")
        with input_column:
            video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…", help="Watch links, Shorts, youtu.be links, and video IDs are supported.")
        with button_column:
            load_clicked = st.form_submit_button("Load video", type="primary", use_container_width=True)

    if load_clicked:
        try:
            video_id = parse_video_id(video_url)
            settings.validate()
            with st.status("Preparing your video…", expanded=True) as status:
                st.write("Fetching available captions…")
                data = fetch_transcript(video_id, languages, setting("YOUTUBE_PROXY_URL"))
                st.write(f"Indexing the {data.language} transcript… Existing embeddings are reused when available.")
                engine = build_index(data, settings, directory)
                status.update(label="Ready to chat", state="complete", expanded=False)
            # Commit the new video only after indexing succeeds.
            st.session_state.engine = engine
            st.session_state.messages = []
            st.session_state.loaded_languages = languages
        except Exception as exc:
            st.error(friendly_error(exc))

    engine = st.session_state.engine
    if engine is None:
        st.markdown("---")
        left, middle, right = st.columns(3)
        with left:
            st.markdown("#### Find the answer")
            st.caption("Ask about a concept, example, or claim mentioned in the video.")
        with middle:
            st.markdown("#### Get the big picture")
            st.caption("Summarize the complete transcript and pull out the main takeaways.")
        with right:
            st.markdown("#### Go to the source")
            st.caption("Open timestamped passages to check an answer in context.")
        st.info("Paste a YouTube URL above and select Load video to begin. The video needs available captions.")
        st.chat_input("Ask about your video…", disabled=True)
        return

    config_changed = settings.signature != engine.settings_signature or languages != st.session_state.loaded_languages
    if config_changed:
        st.info("Your settings changed. Load the video again to apply them before asking a question.")

    video_column, chat_column = st.columns([1, 1.65], gap="large")
    with video_column:
        st.subheader("The video")
        st.video(engine.transcript.url)
        st.caption(f"Active video: {engine.transcript.video_id}")
        st.caption(f"{engine.transcript.language} captions · {'Auto-generated' if engine.transcript.is_generated else 'Manually supplied'}")
        coverage_column, passages_column = st.columns(2)
        coverage_column.metric("Caption coverage", timestamp(engine.transcript.caption_end))
        passages_column.metric("Indexed passages", len(engine.documents))
        with st.expander("Read the transcript"):
            st.text(engine.transcript.download_text())
        st.download_button(
            "Download transcript", data=engine.transcript.download_text(),
            file_name=f"{engine.transcript.video_id}-transcript.txt", mime="text/plain",
            use_container_width=True,
        )

    with chat_column:
        title_column, clear_column = st.columns([3, 1])
        title_column.subheader("Ask the video")
        if clear_column.button("Clear chat", disabled=not st.session_state.messages, use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        suggestion_columns = st.columns(2)
        summary_clicked = suggestion_columns[0].button("Summarize video", disabled=config_changed, use_container_width=True)
        takeaways_clicked = suggestion_columns[1].button("Key takeaways", disabled=config_changed, use_container_width=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                show_sources(message.get("sources", []))

        if not st.session_state.messages:
            st.caption("Try “What is the main idea?” or “Explain the example used in the video.” You can ask follow-up questions too.")

        question = st.chat_input("Ask a question about this video…", disabled=config_changed, max_chars=4000)
        if summary_clicked:
            question = "Can you summarize the entire video?"
        elif takeaways_clicked:
            question = "What are the key takeaways from the entire video?"
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Reading the transcript…" if summary_clicked or takeaways_clicked else "Finding your answer…"):
                        answer = engine.answer(question, st.session_state.messages, summarize=summary_clicked or takeaways_clicked)
                    sources = [
                        {"start": doc.metadata["start"], "end": doc.metadata["end"],
                         "url": doc.metadata["source"], "text": doc.page_content}
                        for doc in answer.sources
                    ]
                    st.markdown(answer.text)
                    show_sources(sources)
                st.session_state.messages.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer.text, "sources": sources},
                ])
                st.rerun()
            except Exception as exc:
                st.error(friendly_error(exc))
                st.caption("Your question wasn't saved. You can submit it again to retry.")

    st.caption("Answers can make mistakes. Use the transcript sources to verify details.")


if __name__ == "__main__":
    main()
