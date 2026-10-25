# Deploying the hosted desk

The hosted desk (`python -m markout.cup public`) is a single container with no
secrets and no outbound network calls, so it fits the free tier of Hugging Face
Spaces (2 vCPU, 16 GB RAM, no credit card). Every push to `main` that touches the
desk redeploys it through `.github/workflows/desk-space.yml`.

## One-time setup

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

Settings come from the environment: `HOST` and `PORT` (the image uses `0.0.0.0`
and `7860`) and `TRUST_PROXY=1`, which makes the per-client limits read the client
address from `X-Forwarded-For`. Only set that behind a proxy that writes the
header, as Hugging Face's does.

## What to expect

- A free Space sleeps after two days without visitors; the next visit wakes it in
  about a minute.
- Sandboxes live in memory and in `/tmp`, so a restart or a redeploy clears them.
  Visitors get a fresh sandbox on the next request.
- `GET /healthz` reports the number of live sessions. Each request is one JSON
  line in the Space's logs.
