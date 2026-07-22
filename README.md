# marvin-integration-openai-images

A **capability** integration for [Marvin](https://claude.ai/code): generate images from a text
prompt via the OpenAI image API. Advertises the `image.generate` capability, so Marvin's
media-enrichment resolver can discover and invoke it uniformly.

## Install

```bash
uv pip install marvin-integration-openai-images   # on a Marvin host, then restart
```

Then **Settings → Integrations → OpenAI Images → Configure**, paste an OpenAI API key (`sk-…`),
optionally set `model` / `size`. Once configured, any enrichment step that needs `image.generate`
resolves to it automatically.

## Capability

| | |
|---|---|
| **Capability** | `image.generate` |
| **Credential** | `api_key` — an OpenAI API key |
| **Config** | `model` (default `gpt-image-1`), `size` (default `1024x1024`) |
| **Invoke** | in `{prompt, count?}` → out `{images: [{image_b64 \| url}]}` |
| **Cost / approval** | `cost_hint="paid"`, `requires_approval=True` |

Generation is slow, so the provider sets a generous 120s HTTP timeout; the resolver caps
wall-clock on its side.

## Develop

```bash
uv run --extra dev pytest
```

Depends only on `marvin-integration-sdk` — tests run without Marvin or network.
