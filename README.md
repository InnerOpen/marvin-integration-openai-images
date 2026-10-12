# marvin-integration-openai-images

A **capability** integration for [Marvin](https://claude.ai/code): generate images from a text
prompt, or edit a picture (remove its background, set it in a new scene, make a variant), via the
OpenAI image API. Advertises the `image.generate` and `image.edit` capabilities, so Marvin's
workflows and media resolver can discover and invoke it uniformly.

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
| **Capabilities** | `image.generate` (`generate`), `image.edit` (`edit`) |
| **Credential** | `api_key` — an OpenAI API key |
| **Config** | `model` (default `gpt-image-1`), `size` (default `1024x1024`) |
| **generate** | in `{prompt, count?}` → out `{images: [{image_b64 \| url}]}` |
| **edit** | in `{prompt, image_b64, image_mime?, background? auto\|transparent\|opaque, fidelity? high\|low}` → out `{image_b64}`; the picture goes to `/v1/images/edits` as a file, with `input_fidelity` (default high) so its details stay; a transparent background comes back as PNG. In a Marvin workflow, the integration step's `source` fills `image_b64`, and the result is kept as a version of that picture. |
| **Cost / approval** | both `cost_hint="paid"`, `requires_approval=True` |

Generation is slow, so the provider sets a generous 120s HTTP timeout; the resolver caps
wall-clock on its side.

## Logo

`logo.svg` is the black OpenAI Blossom, unmodified, from OpenAI's logo pack at [openai.com/brand](https://openai.com/brand/). OpenAI's terms: show it only where it relates to OpenAI services, exactly as provided (no added colours, so on a light background), never more prominent than Marvin's own branding and never implying endorsement.

## Develop

```bash
uv run --extra dev pytest
```

Depends only on `marvin-integration-sdk` — tests run without Marvin or network.
