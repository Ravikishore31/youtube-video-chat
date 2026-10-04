# Video Notes — YouTube transcript chat

A Streamlit application that accepts a YouTube URL and answers questions using its captions. Built from the supplied LangChain RAG example, with OpenRouter embeddings, persistent Chroma storage, and Gemini answers.

## Run locally

Use Python 3.11 or 3.12. In this application's directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` to set both `OPENROUTER_API_KEY` and `GOOGLE_API_KEY`. You can also enter the keys in the app's sidebar. Server-provided keys are used internally and never populated into browser inputs. Keys need access to the configured models and sufficient provider quota/credits.

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501), paste a YouTube URL, and select **Load video**. Then type questions or select **Summarize video** or **Key takeaways**.

For macOS/Linux, activate with `source .venv/bin/activate` and copy the environment template with `cp .env.example .env`. If PowerShell blocks script activation, use `.\.venv\Scripts\python.exe` directly for the install and run commands.

## Features

- Watch, Shorts, live, mobile, embedded, and `youtu.be` URLs, plus video IDs.
- Manually supplied or generated captions, preferred languages, and fallback to another available caption language.
- Chat history scoped to the active browser session and active video; follow-up questions are rewritten before retrieval.
- Timestamped source passages and links to the corresponding YouTube moments.
- Full-transcript summaries that process every indexed passage in bounded groups.
- Read/download the timestamped transcript and clear the conversation.
- Actionable errors for invalid links, unavailable captions, blocked transcript requests, and common provider errors.

## Configuration

Copy `.env.example` to `.env`, use environment variables, or configure top-level Streamlit secrets in `.streamlit/secrets.toml`. Environment variables take precedence over `.env` and Streamlit secrets. API keys and model names can also be changed through the sidebar; reload the video to apply changes.

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
GOOGLE_API_KEY=your_google_ai_studio_key
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
GEMINI_MODEL=gemini-3.5-flash
CHROMA_DIR=data/chroma
```

`GEMINI_API_KEY` is accepted as a fallback for `GOOGLE_API_KEY`. The model defaults match the reference code and can be replaced with models your accounts support. Embedding requests use the OpenRouter API at `https://openrouter.ai/api/v1`.

The included `requirements-lock.txt` records the exact Python dependencies used for verification. To reproduce that environment, install it instead of `requirements.txt`.

## RAG behavior

1. Validate the URL and select a caption track.
2. Normalize caption text and retain its original start/duration timestamps.
3. Split it with `RecursiveCharacterTextSplitter` (1,000 characters, 200-character overlap) and map each passage back to its caption times.
4. Embed passages with `OpenAIEmbeddings` using OpenRouter and store them in Chroma.
5. Retrieve up to four passages for a question, with conversation context used to resolve follow-ups.
6. Ask Gemini to answer from these passages, cite source numbers, and say it does not know if they lack the answer.

Each Chroma collection is identified by the video, full caption content/timing, caption language, embedding model, and chunk configuration. Identical captions reuse existing embeddings; changing videos, transcript content, or embedding models selects another collection. Partially indexed collections can resume after a failed request. A failed video load preserves the previous working video and conversation.

Summary requests process the complete transcript rather than top-k retrieval. Very long summaries use intermediate notes and hierarchical reduction, so they take longer and incur additional Gemini requests. Follow-ups usually make an extra Gemini request to rewrite the question.

## Scope and hosting

This app answers from captions; it does not inspect video frames or transcribe audio. Without available captions, loading stops with an explanation. Source timestamps identify caption passages rather than exact word boundaries. Auto-generated captions and AI answers can be inaccurate; use the sources to check details.

YouTube can block transcript fetching, especially from cloud hosting IPs. The [transcript library documents this limitation](https://github.com/jdepoix/youtube-transcript-api#working-around-ip-bans-requestblocked-or-ipblocked-exception). If necessary, configure an existing HTTP/HTTPS proxy through `YOUTUBE_PROXY_URL` in `.env` or Streamlit secrets. A proxy does not guarantee access.

For Streamlit hosting, set the main file to `app.py` and configure the two API keys in the platform's secrets settings. Chroma needs a writable directory; persistence across server restarts depends on whether your host provides durable storage. This repository is runnable locally and has not been published to a hosting service.

For public sharing, see [DEPLOY.md](DEPLOY.md). Set `REQUIRE_USER_KEYS=true` to require every visitor to provide their own credentials. The included Render Blueprint and Dockerfile enable that mode by default.

API keys stay in server memory and are not written to Chroma or included in transcript/chat downloads. The caption text, metadata, and embeddings are persisted on the server; embeddings are sent to OpenRouter and selected passages/summary notes are sent to Google. Browser sessions have separate conversations, while the server reuses identical public-video transcript indexes. Treat this as a local/demo app: add authentication and appropriate per-user quotas before sharing a server configured with your own provider keys publicly.

## Verification

Run offline regression and Streamlit interaction tests without API keys:

```powershell
python -m unittest discover -s tests -v
```

Tests use a real persistent Chroma database with local deterministic embeddings and mock network/model responses. They check video isolation, repeated-load reuse, timestamp mapping, follow-up retrieval, summary coverage, caption fallback/errors, and the load/chat/clear UI flow. Live provider answers require valid keys and are not covered by the offline tests.

Library references: [Streamlit chat](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps), [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api), [LangChain Chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma), [LangChain Gemini](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai), [OpenRouter embeddings](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request).
