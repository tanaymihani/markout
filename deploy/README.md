# Deploying the hosted desk

The live demo on the site does not need any of this: it runs the desk inside the
visitor's browser (`docs/desk/`, Pyodide), so GitHub Pages serves it for free and
there is nothing to deploy. This page is for the server version
(`python -m markout.cup public`), which serves many visitors from one process: a single
container with no secrets and no outbound network calls.

## Hugging Face Spaces

Since July 2026, Docker Spaces need a paid Hugging Face plan (PRO for a personal account);
only static Spaces are free. With a PRO account, every push to `main` that touches the desk
redeploys it through `.github/workflows/desk-space.yml`, after this one-time setup:

1. Create a Hugging Face account and a new Space: **SDK: Docker**, template
   **Blank**, hardware **CPU basic (free)**, visibility **Public**. Name it, for
   example, `markout-desk`.
2. Create an access token at huggingface.co/settings/tokens: type
   **Fine-grained**, with **write** access to that one Space only.
3. In the GitHub repository, open Settings → Secrets and variables → Actions:
   - under **Secrets**, add `HF_TOKEN` with the token;
   - under **Variables**, add `HF_SPACE` with `<your-hf-username>/markout-desk`.
4. Run the **deploy desk** workflow once from the Actions tab (or push to `main`).

The Space builds the image from `Dockerfile` and serves it on port 7860. It is
then at `https://huggingface.co/spaces/<user>/markout-desk`, and full screen at
`https://<user>-markout-desk.hf.space`.

## Running the same image anywhere else

```bash
docker build -t markout-desk .
docker run --rm -p 7860:7860 markout-desk      # http://127.0.0.1:7860
```

Any host that runs a Dockerfile works. Render's free web service, for example, builds it
straight from the repository with no credit card, though a free instance sleeps after 15
idle minutes and takes about a minute to wake.

Settings come from the environment: `HOST` and `PORT` (the image uses `0.0.0.0`
and `7860`) and `TRUST_PROXY=1`, which makes the per-client limits read the client
address from `X-Forwarded-For`. Only set that behind a proxy that writes the
header, as Hugging Face's does.

## What to expect

- Sandboxes live in memory and in `/tmp`, so a restart or a redeploy clears them.
  Visitors get a fresh sandbox on the next request.
- `GET /healthz` reports the number of live sessions. Each request is one JSON
  line in the Space's logs.
