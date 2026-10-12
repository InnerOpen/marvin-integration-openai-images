"""OpenAI images — a capability provider advertising `image.generate` and `image.edit`.

* ``generate`` — in ``{prompt, count?}`` → out ``{images: [{image_b64}]}``.
* ``edit`` — in ``{prompt, image_b64, image_mime?, background?, fidelity?}`` → out ``{image_b64}``: the picture changed
  as the prompt says (its background removed, set in a new scene, a seasonal variant), keeping its details
  (``input_fidelity: high``). Marvin fills ``image_b64`` from the workflow step's ``source`` and keeps the result as a
  version of that picture.

Pure with respect to Marvin — it gets the API key + config + http, and returns image bytes; Marvin persists them.
"""

import base64
import uuid
from typing import ClassVar

from marvin_integration_sdk import (
    CATEGORY_CAPABILITY,
    CredentialField,
    IntegrationContext,
    IntegrationProvider,
    ProviderAction,
    register_provider,
)

_ENDPOINT = "https://api.openai.com/v1/images/generations"
_EDIT_ENDPOINT = "https://api.openai.com/v1/images/edits"
_MAX_EDIT_BYTES = 50 * 1024 * 1024  # the edits API's limit per image
# Image generation is slow — set a generous timeout (the resolver caps wall-clock on its side).
_TIMEOUT = 120


@register_provider
class OpenAIImagesProvider(IntegrationProvider):
    slug = "openai_images"
    name = "OpenAI Images"
    description = "Generate images from a text prompt, or edit a picture (remove its background, new scenes, variants), via the OpenAI image API."
    category = CATEGORY_CAPABILITY
    icon = "🎨"
    logo = "logo.svg"

    credentials = (CredentialField(key="api_key", label="OpenAI API Key", help="An OpenAI API key (sk-…) with image access."),)
    config_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "model": {"type": "string", "title": "Model", "default": "gpt-image-1"},
            "size": {"type": "string", "title": "Size", "default": "1024x1024"},
        },
        "additionalProperties": False,
    }
    actions = (
        ProviderAction(
            key="generate",
            label="Generate image",
            description="Generate an image from a text prompt.",
            capability="image.generate",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "title": "Prompt"},
                    "count": {"type": "integer", "title": "Count", "default": 1},
                    "reference_image_b64": {"type": "string", "title": "Reference image (ignored for generation)"},
                },
                "required": ["prompt"],
            },
            output_schema={
                "type": "object",
                "properties": {"images": {"type": "array", "items": {"type": "object", "properties": {"image_b64": {"type": "string"}}}}},
            },
            cost_hint="paid",
            requires_approval=True,
        ),
        ProviderAction(
            key="edit",
            label="Edit an image",
            description="Change a picture as the prompt says — remove its background, set it in a new scene, make a seasonal "
            "variant — keeping its details. The workflow step's source is the picture.",
            capability="image.edit",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "title": "What to change"},
                    "image_b64": {"type": "string", "title": "The picture (filled in from the step's source)"},
                    "image_mime": {"type": "string", "title": "Its type", "default": "image/png"},
                    "background": {"type": "string", "title": "Background", "enum": ["auto", "transparent", "opaque"], "default": "auto"},
                    "fidelity": {"type": "string", "title": "Keep the details", "enum": ["high", "low"], "default": "high"},
                },
                "required": ["prompt", "image_b64"],
            },
            output_schema={"type": "object", "properties": {"image_b64": {"type": "string"}}},
            cost_hint="paid",
            requires_approval=True,
        ),
    )

    def check(self, ctx: IntegrationContext) -> tuple[str, str | None]:
        if not ctx.secret:
            return ("unconfigured", "Missing OpenAI API key.")
        if not ctx.secret.startswith("sk-"):
            return ("error", "That doesn't look like an OpenAI API key (should start with sk-).")
        return ("ok", None)

    def run_action(self, key: str, args: dict, ctx: IntegrationContext) -> dict:
        if key not in ("generate", "edit"):
            raise NotImplementedError(f"openai_images has no action '{key}'")
        if not ctx.secret:
            raise ValueError("No OpenAI API key configured.")
        if key == "edit":
            return self._edit(args or {}, ctx)
        prompt = (args or {}).get("prompt")
        if not prompt:
            raise ValueError("A 'prompt' is required.")

        cfg = ctx.config or {}
        model = cfg.get("model") or "gpt-image-1"
        # No response_format: newer image endpoints reject it. gpt-image-1 returns b64_json;
        # dall-e-* return a URL. run_action normalizes both below.
        body = {"model": model, "prompt": prompt, "n": int((args or {}).get("count") or 1), "size": cfg.get("size") or "1024x1024"}

        try:
            resp = ctx.http.post(_ENDPOINT, json=body, headers={"Authorization": f"Bearer {ctx.secret}"}, timeout=_TIMEOUT)
        except Exception as e:  # surface transport/guard failures cleanly
            ctx.logger.warning(f"[openai_images] request failed: {e}")
            raise ValueError(f"OpenAI images request failed: {e}") from e

        if not resp.ok:
            raise ValueError(f"OpenAI images returned HTTP {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        images: list[dict] = []
        for item in payload.get("data", []):
            if item.get("b64_json"):
                images.append({"image_b64": item["b64_json"]})
            elif item.get("url"):
                images.append({"url": item["url"]})

        # Surface usage so the core can log cost against the monthly AI spend (data only — the
        # provider stays DB-free). The images API returns token usage for gpt-image-* models.
        result: dict = {"images": images}
        u = payload.get("usage") or {}
        result["usage"] = {
            "provider_type": "openai",
            "model": model,
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens", (u.get("input_tokens", 0) + u.get("output_tokens", 0))),
        }
        return result

    def _edit(self, args: dict, ctx: IntegrationContext) -> dict:
        prompt, image_b64 = args.get("prompt"), args.get("image_b64")
        if not prompt:
            raise ValueError("A 'prompt' is required.")
        if not image_b64:
            raise ValueError("No picture to edit: set the workflow step's source.")
        try:
            image = base64.b64decode(image_b64.split(",", 1)[-1] if image_b64.startswith("data:") else image_b64)
        except ValueError as e:
            raise ValueError("The picture isn't valid base64.") from e
        if len(image) > _MAX_EDIT_BYTES:
            raise ValueError("The picture is over 50 MB, the most the edits API takes.")
        cfg = ctx.config or {}
        model = cfg.get("model") or "gpt-image-1"
        background = args.get("background") or "auto"
        fields = {
            "model": model,
            "prompt": prompt,
            "size": cfg.get("size") or "1024x1024",
            "background": background,
            "input_fidelity": args.get("fidelity") or "high",
            # A transparent background needs a format with alpha.
            "output_format": "png" if background == "transparent" else cfg.get("output_format") or "png",
        }
        mime = args.get("image_mime") or "image/png"
        body, content_type = _multipart(fields, ("image", f"image.{mime.split('/')[-1]}", mime, image))
        try:
            resp = ctx.http.post(
                _EDIT_ENDPOINT, data=body, headers={"Authorization": f"Bearer {ctx.secret}", "Content-Type": content_type}, timeout=_TIMEOUT
            )
        except Exception as e:  # surface transport/guard failures cleanly
            ctx.logger.warning(f"[openai_images] edit request failed: {e}")
            raise ValueError(f"OpenAI images edit request failed: {e}") from e
        if not resp.ok:
            raise ValueError(f"OpenAI images returned HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        first = next(iter(payload.get("data") or []), {})
        if not first.get("b64_json") and not first.get("url"):
            raise ValueError("OpenAI images returned no picture.")
        u = payload.get("usage") or {}
        return {
            **({"image_b64": first["b64_json"]} if first.get("b64_json") else {"url": first["url"]}),
            "usage": {
                "provider_type": "openai",
                "model": model,
                "input_tokens": u.get("input_tokens", 0),
                "output_tokens": u.get("output_tokens", 0),
                "total_tokens": u.get("total_tokens", (u.get("input_tokens", 0) + u.get("output_tokens", 0))),
            },
        }


def _multipart(fields: dict, file: tuple[str, str, str, bytes]) -> tuple[bytes, str]:
    """A multipart/form-data body (the edits endpoint takes the picture as a file) and its Content-Type."""
    boundary = f"marvin-{uuid.uuid4().hex}"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    name, filename, mime, data = file
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
        + data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
