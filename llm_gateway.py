"""
Finds whatever LLM gateway the user already has configured.

PaperTrail only needs one thing from an LLM: turn a free-text description of a
finding into predictions.yaml. That is a single chat completion call, and just
about every gateway in use today speaks the OpenAI-compatible dialect, so there
is no reason to hardcode one vendor.

The problem is naming. One person exports LITELLM_API_KEY + LITELLM_BASE_URL,
the next has OPENAI_API_KEY, their colleague is on a company proxy called
ACME_GATEWAY_TOKEN, and somebody is running Ollama with nothing set at all.
Rather than demand a specific variable, this module reads the environment and
works out what is there.

Two rules hold everywhere in this file:

  * Nothing raises. Discovery returns None when it finds nothing, and chat()
    returns an (answer, error) pair. A variable spelled in a way we did not
    anticipate is a miss, never a crash.
  * The user is never left guessing. setup_help() prints what to export.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# ── How environment variables get read ───────────────────────────────────────
# Variables are split into a prefix and a role suffix, so ACME_GATEWAY_API_KEY
# becomes prefix "ACME_GATEWAY", role "key". Longest suffix wins, otherwise
# "..._API_KEY" would match the short "..._KEY" rule and leave a stray "_API"
# glued to the prefix.
KEY_SUFFIXES = ("API_KEY", "APIKEY", "ACCESS_TOKEN", "AUTH_TOKEN",
                "SECRET_KEY", "TOKEN", "KEY", "API")
URL_SUFFIXES = ("BASE_URL", "API_BASE", "API_URL", "ENDPOINT_URL",
                "ENDPOINT", "BASE", "HOST", "URL")
MODEL_SUFFIXES = ("MODEL_NAME", "MODEL_ID", "DEPLOYMENT_NAME",
                  "DEPLOYMENT", "MODEL")

# Providers that need no base URL because theirs is public and stable. A user
# with only FOO_API_KEY set still gets a working gateway if FOO is in here.
KNOWN_ENDPOINTS = {
    "GROQ":        "https://api.groq.com/openai/v1",
    "OPENAI":      "https://api.openai.com/v1",
    "GEMINI":      "https://generativelanguage.googleapis.com/v1beta/openai",
    "GOOGLE":      "https://generativelanguage.googleapis.com/v1beta/openai",
    "MISTRAL":     "https://api.mistral.ai/v1",
    "DEEPSEEK":    "https://api.deepseek.com/v1",
    "OPENROUTER":  "https://openrouter.ai/api/v1",
    "TOGETHER":    "https://api.together.xyz/v1",
    "FIREWORKS":   "https://api.fireworks.ai/inference/v1",
    "PERPLEXITY":  "https://api.perplexity.ai",
    "XAI":         "https://api.x.ai/v1",
    "CEREBRAS":    "https://api.cerebras.ai/v1",
    "SAMBANOVA":   "https://api.sambanova.ai/v1",
    "NVIDIA":      "https://integrate.api.nvidia.com/v1",
    "HF":          "https://router.huggingface.co/v1",
    "HUGGINGFACE": "https://router.huggingface.co/v1",
    "DASHSCOPE":   "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "MOONSHOT":    "https://api.moonshot.cn/v1",
}

# Only consulted when the user has not named a model. Ordered best-effort; if
# the first is retired the call falls through to the next.
KNOWN_MODELS = {
    "GROQ":        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "OPENAI":      ["gpt-4o-mini"],
    "GEMINI":      ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "GOOGLE":      ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "MISTRAL":     ["mistral-small-latest"],
    "DEEPSEEK":    ["deepseek-chat"],
    "OPENROUTER":  ["meta-llama/llama-3.3-70b-instruct"],
    "TOGETHER":    ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
    "CEREBRAS":    ["llama-3.3-70b"],
    "XAI":         ["grok-2-latest"],
}

# Words that mark a prefix as LLM-ish. A prefix that has both a key and a URL
# is accepted regardless — that pairing is signal enough for a custom gateway —
# but a key on its own needs one of these to avoid grabbing, say, a database
# password.
LLM_HINTS = (
    "LLM", "GPT", "OPENAI", "GENAI", "GATEWAY", "PROXY", "LITELLM", "INFERENCE",
    "CHAT", "COMPLETION", "GROQ", "GEMINI", "MISTRAL", "COHERE", "TOGETHER",
    "FIREWORKS", "DEEPSEEK", "PERPLEXITY", "OPENROUTER", "VERTEX", "BEDROCK",
    "HUGGINGFACE", "OLLAMA", "VLLM", "XAI", "GROK", "QWEN", "MOONSHOT",
    "NVIDIA", "SAMBANOVA", "CEREBRAS", "DASHSCOPE", "ZHIPU", "AI",
)

# Prefixes that carry credentials for something else entirely. NCBI matters
# most here: the pipeline genuinely uses NCBI_API_KEY for PubMed, and without
# this guard the scanner would offer it up as an LLM key.
NOT_LLM = {
    "NCBI", "ENTREZ", "PUBMED", "CROSSREF", "ORCID", "ELSEVIER", "SPRINGER",
    "AWS", "GITHUB", "GH", "GIT", "GITLAB", "DOCKER", "KUBE", "POSTGRES", "PG",
    "MYSQL", "REDIS", "MONGO", "S3", "GCS", "SLACK", "JIRA", "STRIPE", "TWILIO",
    "SENDGRID", "NPM", "PYPI", "CONDA", "SSH", "GPG", "KAGGLE", "WANDB",
    "NEPTUNE", "COMET", "SENTRY", "DATADOG", "GRAFANA",
}

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# PAPERTRAIL_LLM_* always wins, so there is one documented way to override
# whatever else happens to be in the environment.
CANONICAL_PREFIX = "PAPERTRAIL_LLM"


class Gateway:
    """One resolved endpoint, plus a note of where each value came from."""

    def __init__(self, name, base_url, api_key="", model="", origin=None, local=False):
        self.name = name
        self.base_url = _tidy_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.local = local
        # Which env vars produced this, so the UI can show the user what was
        # picked up rather than making them guess.
        self.origin = origin or {}

    def __repr__(self):
        return f"<Gateway {self.name} {self.base_url} model={self.model or 'auto'}>"

    def label(self):
        who = self.name.replace("_", " ").title() if self.name else "LLM"
        return f"{who}: {self.model}" if self.model else who


def _tidy_base_url(url):
    """Strip the bits people commonly paste in but that we add back ourselves."""
    url = (url or "").strip().rstrip("/")
    return re.sub(r"/(?:v1|chat/completions|v1/chat/completions)$", "", url).rstrip("/")


def _split_var(name):
    """Return (prefix, role) for an env var name, or None if it is not one of ours."""
    upper = name.upper()
    for role, suffixes in (("key", KEY_SUFFIXES),
                           ("url", URL_SUFFIXES),
                           ("model", MODEL_SUFFIXES)):
        for suffix in sorted(suffixes, key=len, reverse=True):
            if upper == suffix:
                return "", role
            if upper.endswith("_" + suffix):
                return upper[: -len(suffix) - 1], role
    return None


def _looks_like_llm(prefix):
    if not prefix:
        return True          # a bare API_KEY / BASE_URL pair is generic by nature
    head = prefix.split("_")[0]
    if head in NOT_LLM or prefix in NOT_LLM:
        return False
    return any(hint in prefix for hint in LLM_HINTS)


def _scan_environment(env):
    """Group the environment into {prefix: {"key": .., "url": .., "model": ..}}."""
    found = {}
    for name, value in env.items():
        if not value or not value.strip():
            continue
        parsed = _split_var(name)
        if not parsed:
            continue
        prefix, role = parsed
        slot = found.setdefault(prefix, {})
        # First writer wins, so a later FOO_URL cannot clobber an explicit
        # FOO_BASE_URL that was already matched by the longer suffix.
        slot.setdefault(role, value.strip())
        slot.setdefault("origin", {})[role] = name
    return found


def _score(prefix, slot):
    """Higher is better. Returns None for prefixes we should not use at all."""
    has_key, has_url = bool(slot.get("key")), bool(slot.get("url"))
    known = prefix.split("_")[0] in KNOWN_ENDPOINTS or prefix in KNOWN_ENDPOINTS

    if prefix == CANONICAL_PREFIX:
        return 100
    if not _looks_like_llm(prefix):
        return None
    if has_key and has_url:
        return 80            # a self-declared gateway: the clearest signal there is
    if prefix in ("LLM", "OPENAI_COMPATIBLE", ""):
        return 70
    if has_key and known:
        return 60
    if has_url:
        return 40            # keyless endpoint, e.g. a local vLLM server
    return None


def _endpoint_for(prefix, slot):
    if slot.get("url"):
        return slot["url"]
    for candidate in (prefix, prefix.split("_")[0]):
        if candidate in KNOWN_ENDPOINTS:
            return KNOWN_ENDPOINTS[candidate]
    return ""


def _default_model_for(prefix, slot):
    if slot.get("model"):
        return slot["model"]
    for candidate in (prefix, prefix.split("_")[0]):
        if candidate in KNOWN_MODELS:
            return KNOWN_MODELS[candidate][0]
    return ""               # resolved later by asking the gateway itself


def _ollama_gateway(env):
    """Ollama needs no key, so it is only offered once we see it answering."""
    host = (env.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    models = ollama_models(host)
    if not models:
        return None
    preferred = ("llama3.2", "qwen2.5", "phi3.5", "llama3.1", "mistral", "gemma2")
    pick = next((m for m in models if m.split(":")[0] in preferred), models[0])
    return Gateway("ollama", host, model=pick, local=True,
                   origin={"url": "OLLAMA_HOST" if env.get("OLLAMA_HOST") else "(default)"})


def ollama_models(host=None):
    """Model tags from a local Ollama daemon, or [] if it is not running."""
    host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as resp:
            return [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except Exception:
        return []


def resolve(env=None):
    """
    Best available gateway, or None when the environment has nothing usable.

    Candidates are ranked rather than tried in a fixed order, so an explicitly
    configured gateway always beats one we merely guessed at.
    """
    env = os.environ if env is None else env

    ranked = []
    for prefix, slot in _scan_environment(env).items():
        score = _score(prefix, slot)
        if score is None:
            continue
        endpoint = _endpoint_for(prefix, slot)
        if not endpoint:
            continue         # a key with nowhere to send it is not a gateway
        ranked.append((score, prefix, slot, endpoint))

    for _score_, prefix, slot, endpoint in sorted(ranked, key=lambda r: -r[0]):
        return Gateway(
            name=prefix.lower() or "llm",
            base_url=endpoint,
            api_key=slot.get("key", ""),
            model=_default_model_for(prefix, slot),
            origin=slot.get("origin", {}),
        )

    # Nothing declared in the environment — a running Ollama is still a gateway.
    return _ollama_gateway(env)


def _completion_urls(gateway):
    """
    Candidate chat-completion URLs, most likely first.

    Gateways disagree about whether /v1 belongs in the base URL, and users paste
    it either way. Instead of insisting on one convention we try both and let a
    404 move us along.
    """
    base = gateway.base_url
    if gateway.local:
        return [f"{base}/v1/chat/completions", f"{base}/api/chat"]
    return [f"{base}/v1/chat/completions", f"{base}/chat/completions"]


def _model_candidates(gateway):
    """Models to attempt, in order. Named models are tried before guesses."""
    if gateway.model:
        return [gateway.model]
    head = gateway.name.upper().split("_")[0]
    guesses = KNOWN_MODELS.get(head, [])
    discovered = list_models(gateway)
    # Whatever the gateway itself reports is more trustworthy than our table.
    return (discovered[:3] + [m for m in guesses if m not in discovered]) or ["gpt-4o-mini"]


def list_models(gateway):
    """Model ids the gateway advertises, or [] if it will not say."""
    if gateway.local:
        return ollama_models(gateway.base_url)
    headers = {"Content-Type": "application/json"}
    if gateway.api_key:
        headers["Authorization"] = f"Bearer {gateway.api_key}"
    for url in (f"{gateway.base_url}/v1/models", f"{gateway.base_url}/models"):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                payload = json.loads(resp.read())
            entries = payload.get("data") if isinstance(payload, dict) else payload
            ids = [e.get("id") for e in (entries or []) if isinstance(e, dict) and e.get("id")]
            if ids:
                return ids
        except Exception:
            continue
    return []


def _error_detail(exc):
    """Pull the useful part out of an HTTP error body, if there is one."""
    body = ""
    if hasattr(exc, "read"):
        try:
            body = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            body = ""
    return body


def chat(messages, *, gateway=None, model=None, max_tokens=800,
         temperature=0.05, timeout=30):
    """
    Send a chat completion.

    Returns (text, error). Exactly one is ever non-empty, and no exception
    escapes — callers all have a rule-based fallback to fall back to.
    """
    gateway = gateway or resolve()
    if gateway is None:
        return None, "No LLM gateway configured"

    headers = {"Content-Type": "application/json"}
    if gateway.api_key:
        headers["Authorization"] = f"Bearer {gateway.api_key}"

    models = [model] if model else _model_candidates(gateway)
    urls = _completion_urls(gateway)
    last_error = ""

    for candidate in models:
        payload = json.dumps({
            "model": candidate,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode()

        for url in urls:
            try:
                req = urllib.request.Request(url, data=payload,
                                             headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
                text = _extract_text(data)
                if text:
                    # Remember what worked so the next call skips the probing.
                    gateway.model = candidate
                    return text, None
                last_error = f"{candidate}: response had no message content"
            except urllib.error.HTTPError as exc:
                detail = _error_detail(exc)
                last_error = f"{candidate} @ {url}: HTTP {exc.code} {detail}".strip()
                if exc.code in (401, 403):
                    # Credentials are wrong; trying other models will not help.
                    return None, (f"{gateway.label()} rejected the API key "
                                  f"(HTTP {exc.code}). Check the key is current "
                                  f"and has access to this gateway.")
                if exc.code not in (404, 400):
                    break     # a real server-side problem, not a URL/model mismatch
            except Exception as exc:
                last_error = f"{candidate} @ {url}: {exc}"

    return None, f"{gateway.label()} call failed. Last error: {last_error}"


def _extract_text(data):
    """Read the reply out of either an OpenAI-shaped or Ollama-shaped response."""
    try:
        choice = data["choices"][0]
        return (choice.get("message", {}).get("content") or choice.get("text", "")).strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    try:
        return data["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError):
        return ""


def describe(gateway=None):
    """A small dict for the status panel. Safe to call with nothing configured."""
    gateway = gateway or resolve()
    if gateway is None:
        return {
            "configured": False,
            "name": "",
            "model": "",
            "base_url": "",
            "label": "No LLM gateway found",
            "detected_from": {},
            "help": setup_help(),
        }
    return {
        "configured": True,
        "name": gateway.name,
        "model": gateway.model,
        "base_url": gateway.base_url,
        "label": gateway.label(),
        "detected_from": gateway.origin,
        "help": "",
    }


def setup_help(short=False):
    """What to tell a user who has no gateway configured."""
    if short:
        return ("Set PAPERTRAIL_LLM_BASE_URL and PAPERTRAIL_LLM_API_KEY to enable "
                "LLM extraction. Without them PaperTrail uses its rule-based parser.")
    return (
        "No LLM gateway detected. PaperTrail will use its rule-based parser, which "
        "works but is less flexible with unusual phrasing.\n"
        "\n"
        "To enable LLM extraction, export the endpoint you already use:\n"
        "\n"
        "  export PAPERTRAIL_LLM_BASE_URL='https://your-gateway.example.com'\n"
        "  export PAPERTRAIL_LLM_API_KEY='your-key'\n"
        "  export PAPERTRAIL_LLM_MODEL='your-model'      # optional\n"
        "\n"
        "Any OpenAI-compatible gateway works. If you already have variables set "
        "under another name — OPENAI_API_KEY, LITELLM_BASE_URL, GROQ_API_KEY, a "
        "company proxy, anything ending in _API_KEY next to a _BASE_URL — "
        "PaperTrail picks them up on its own and you need not export anything.\n"
        "\n"
        "A local Ollama daemon is detected automatically; no key required."
    )


if __name__ == "__main__":
    # `python llm_gateway.py` answers "what will PaperTrail actually use?"
    found = resolve()
    if found is None:
        print(setup_help())
    else:
        print(f"Gateway   : {found.name}")
        print(f"Endpoint  : {found.base_url}")
        print(f"Model     : {found.model or '(ask the gateway)'}")
        print(f"API key   : {'set' if found.api_key else 'not needed'}")
        for role, var in sorted(found.origin.items()):
            print(f"  {role:<6}<- {var}")
