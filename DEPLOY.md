# Public hosting

This application can run on Streamlit Community Cloud, Render, or a host supporting Docker. Public configuration requires visitors to supply their own OpenRouter and Google API keys for their session. Server-configured keys are never placed in browser inputs.

## Permanent hosting on Render

1. Put the contents of this directory in a GitHub repository, with `app.py` and `render.yaml` at the root.
2. In Render, create a Blueprint from that repository. The included `render.yaml` specifies the Python version, dependency install, Streamlit start command, health endpoint, and visitor-key mode.
3. Deploy and share the HTTPS address shown by Render.

The Blueprint selects Render's free plan. Verify the current plan details in your account before deploying. Free hosting may sleep when unused, and Chroma storage is ephemeral unless your host provides a persistent volume. Re-indexing after a restart requires new embedding requests. YouTube may block cloud-host transcript requests; configure `YOUTUBE_PROXY_URL` as a server secret when needed.

For an always-on service, select an appropriate host plan yourself. No paid service has been provisioned by this project.

## Streamlit Community Cloud

1. Push this application's files to a GitHub repository.
2. Open https://share.streamlit.io and create an app from that repository with `app.py` as the main file and Python 3.12 selected.
3. Add this top-level setting under Advanced settings → Secrets:

```toml
REQUIRE_USER_KEYS = "true"
```

4. Deploy and use the assigned `*.streamlit.app` URL. Keep the app's sharing setting public.

The normal `requirements.txt` is used by Community Cloud. The separate lock file records the tested versions.

## Docker

```sh
docker build -t youtube-video-chat .
docker run --rm -p 8501:8501 youtube-video-chat
```

Deploy that image on a container host. Its listening port follows the host's `PORT` environment variable. Mount a writable volume at `/app/data` to retain Chroma embeddings between container restarts.

## Temporary public preview

A Cloudflare Quick Tunnel can expose a local Streamlit instance:

```sh
cloudflared tunnel --url http://127.0.0.1:8502
```

Run the local app with `REQUIRE_USER_KEYS=true` before sharing it. Share only the public URL actually printed by `cloudflared`. This preview ends when the computer sleeps, loses connectivity, or either the app or tunnel process stops. It is not permanent hosting or a registered custom domain.

References: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [Render Blueprint specification](https://render.com/docs/blueprint-spec), [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
