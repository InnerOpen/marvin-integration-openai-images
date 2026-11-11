"""OpenAI images — a capability provider advertising `image.generate` and `image.edit`.

* ``generate`` — in ``{prompt, count?}`` → out ``{images: [{image_b64}]}``.
* ``edit`` — in ``{prompt, image_b64?, image_mime?, reference_images?, background?, fidelity?}`` → out ``{image_b64}``: the
  picture changed as the prompt says (its background removed, set in a new scene, a seasonal variant), keeping its details
  (``input_fidelity: high``, on the models that take it). Marvin fills ``image_b64`` from the workflow step's ``source`` and keeps
  the result as a version of that picture; ``reference_images`` (the step's ``references``: up to 16 pictures in all, e.g.
  a character's master and a style sheet) make a new picture that goes by them all.

Pure with respect to Marvin — it gets the API key + config + http, and returns image bytes; Marvin persists them.
"""

import base64
import re
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
_MODELS_ENDPOINT = "https://api.openai.com/v1/models"
_DEFAULT_MODEL = "gpt-image-2"  # gpt-image-1 was shut down on 2026-10-23 (OpenAI SDK docs)
_QUALITIES = ("auto", "low", "medium", "high")
_SIZE = re.compile(r"^\d{3,4}x\d{3,4}$")


def _size(args: dict, cfg: dict) -> str:
    """The picture's size: the step's own `size` (1536x1024 for a landscape sheet), else the connection's, else square. `auto` lets the model choose."""
    size = str((args or {}).get("size") or cfg.get("size") or "1024x1024").strip().lower()
    if size != "auto" and not _SIZE.match(size):
        raise ValueError("size must be like 1536x1024 (width x height in pixels) or auto")
    return size


_FORMATS = ("png", "jpeg", "webp")
_MAX_EDIT_BYTES = 50 * 1024 * 1024  # the edits API's limit per image
_MAX_EDIT_IMAGES = 16  # …and how many pictures one edit can go by
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
    # What Marvin learns about this provider (SDK 0.9.0): models, cost, formats and how a workflow uses it.
    guide = "guide.md"

    credentials = (CredentialField(key="api_key", label="OpenAI API Key", help="An OpenAI API key (sk-…) with image access."),)
    config_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "title": "Model",
                "default": _DEFAULT_MODEL,
                "description": "gpt-image-2 for finals, gpt-image-1-mini for drafts. (gpt-image-1 is retired.)",
            },
            "size": {
                "type": "string",
                "title": "Size (pixels)",
                "default": "1024x1024",
                "description": "The generated picture's dimensions: 1024x1024 (square), 1536x1024 (landscape), 1024x1536 (portrait) or auto.",
            },
            "quality": {
                "type": "string",
                "title": "Quality",
                "enum": list(_QUALITIES),
                "default": "auto",
                "description": "The default for every call; a step may override it.",
            },
            "output_format": {
                "type": "string",
                "title": "Picture format",
                "enum": list(_FORMATS),
                "default": "png",
                "description": "jpeg is smaller and what Instagram's feed takes.",
            },
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
                    "size": {
                        "type": "string",
                        "title": "Size (pixels)",
                        "description": "This step's own size, e.g. 1536x1024 (landscape), 1024x1536 (portrait), 1024x1024 or auto; default from the connection.",
                    },
                    "quality": {
                        "type": "string",
                        "title": "Quality",
                        "enum": list(_QUALITIES),
                        "description": "low is quick and cheap, high is slow and detailed; default from the connection.",
                    },
                    "output_format": {
                        "type": "string",
                        "title": "Picture format",
                        "enum": list(_FORMATS),
                        "description": "Default from the connection.",
                    },
                    "background": {
                        "type": "string",
                        "title": "Background",
                        "enum": ["auto", "transparent", "opaque"],
                        "description": "transparent needs png or webp.",
                    },
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
            "variant — keeping its details. The workflow step's source is the picture. With `references` (up to 16 more "
            "pictures: a character's master, a style sheet) it makes a new picture that goes by them all, with or without a source.",
            capability="image.edit",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "title": "What to change"},
                    "size": {
                        "type": "string",
                        "title": "Size (pixels)",
                        "description": "This step's own size, e.g. 1536x1024 (landscape), 1024x1536 (portrait), 1024x1024 or auto; default from the connection.",
                    },
                    "image_b64": {"type": "string", "title": "The picture (filled in from the step's source)"},
                    "image_mime": {"type": "string", "title": "Its type", "default": "image/png"},
                    "reference_images": {
                        "type": "array",
                        "title": "More pictures to go by (filled in from the step's references)",
                        "items": {
                            "type": "object",
                            "properties": {"image_b64": {"type": "string"}, "image_mime": {"type": "string"}, "name": {"type": "string"}},
                        },
                    },
                    "background": {"type": "string", "title": "Background", "enum": ["auto", "transparent", "opaque"], "default": "auto"},
                    "fidelity": {"type": "string", "title": "Keep the details", "enum": ["high", "low"], "default": "high"},
                },
                "required": ["prompt"],
            },
            output_schema={"type": "object", "properties": {"image_b64": {"type": "string"}}},
            cost_hint="paid",
            requires_approval=True,
        ),
        ProviderAction(
            key="list_models",
            label="List image models",
            description="The image models this key can use, and whether the connection's model is one of them — a free way to test the connection.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={
                "type": "object",
                "properties": {
                    "models": {"type": "array", "items": {"type": "string"}},
                    "configured": {"type": "string"},
                    "available": {"type": "boolean"},
                },
            },
            cost_hint="free",
        ),
    )

    def check(self, ctx: IntegrationContext) -> tuple[str, str | None]:
        """A real check (free): the key must be accepted and the connection's model must be one it can use."""
        if not ctx.secret:
            return ("unconfigured", "Missing OpenAI API key.")
        if not ctx.secret.startswith("sk-"):
            return ("error", "That doesn't look like an OpenAI API key (should start with sk-).")
        model = self._model(ctx)
        try:
            resp = ctx.http.get(f"{_MODELS_ENDPOINT}/{model}", headers={"Authorization": f"Bearer {ctx.secret}"}, timeout=20)
        except Exception as e:  # noqa: BLE001 — the message is the diagnosis
            return ("error", f"OpenAI didn't answer: {e}")
        if resp.status_code == 401:
            return ("error", "OpenAI rejected the API key.")
        if resp.status_code == 404:
            return ("error", f"This key can't use the model “{model}” — pick one from List image models.")
        if not resp.ok:
            return ("error", f"OpenAI returned HTTP {resp.status_code}: {resp.text[:200]}")
        return ("ok", None)

    @staticmethod
    def _model(ctx: IntegrationContext) -> str:
        return (ctx.config or {}).get("model") or _DEFAULT_MODEL

    def run_action(self, key: str, args: dict, ctx: IntegrationContext) -> dict:
        if key not in ("generate", "edit", "list_models"):
            raise NotImplementedError(f"openai_images has no action '{key}'")
        if not ctx.secret:
            raise ValueError("No OpenAI API key configured.")
        if key == "list_models":
            return self._list_models(ctx)
        if key == "edit":
            return self._edit(args or {}, ctx)
        prompt = (args or {}).get("prompt")
        if not prompt:
            raise ValueError("A 'prompt' is required.")

        cfg = ctx.config or {}
        model = self._model(ctx)
        # No response_format: newer image endpoints reject it. gpt-image-1 returns b64_json;
        # dall-e-* return a URL. run_action normalizes both below.
        body = {"model": model, "prompt": prompt, "n": int((args or {}).get("count") or 1), "size": _size(args, cfg)}
        if model.startswith("gpt-image"):  # the gpt-image models take these; dall-e-* don't
            quality = (args or {}).get("quality") or cfg.get("quality") or "auto"
            fmt = (args or {}).get("output_format") or cfg.get("output_format") or "png"
            background = (args or {}).get("background") or "auto"
            if quality not in _QUALITIES or fmt not in _FORMATS:
                raise ValueError(f"quality must be one of {', '.join(_QUALITIES)} and output_format one of {', '.join(_FORMATS)}.")
            if background == "transparent" and fmt == "jpeg":
                raise ValueError("A transparent background needs png or webp, not jpeg.")
            body.update({"quality": quality, "output_format": fmt, **({"background": background} if background != "auto" else {})})

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

    def _list_models(self, ctx: IntegrationContext) -> dict:
        """The image models this key can use — free, and the plain way to see that the key works."""
        resp = ctx.http.get(_MODELS_ENDPOINT, headers={"Authorization": f"Bearer {ctx.secret}"}, timeout=20)
        if not resp.ok:
            raise ValueError(f"OpenAI returned HTTP {resp.status_code}: {resp.text[:200]}")
        ids = sorted(m.get("id") for m in (resp.json().get("data") or []) if isinstance(m, dict) and m.get("id"))
        models = [i for i in ids if "image" in i or i.startswith("dall-e")]
        configured = self._model(ctx)
        return {"models": models, "configured": configured, "available": configured in models}

    def _edit(self, args: dict, ctx: IntegrationContext) -> dict:
        prompt = args.get("prompt")
        if not prompt:
            raise ValueError("A 'prompt' is required.")
        pictures = _pictures(args)
        if not pictures:
            raise ValueError("No picture to go by: set the workflow step's source or references.")
        cfg = ctx.config or {}
        model = self._model(ctx)
        background = args.get("background") or "auto"
        fields = {
            "model": model,
            "prompt": prompt,
            "size": _size(args, cfg),
            "background": background,
            # A transparent background needs a format with alpha.
            "output_format": "png" if background == "transparent" else cfg.get("output_format") or "png",
        }
        if model.startswith("gpt-image-1.5"):  # only this one takes it high; gpt-image-2 refuses the field, 1-mini is always low
            fields["input_fidelity"] = args.get("fidelity") or "high"
        name = "image[]" if len(pictures) > 1 else "image"
        body, content_type = _multipart(
            fields, [(name, f"image-{i}.{mime.split('/')[-1]}", mime, data) for i, (data, mime) in enumerate(pictures, 1)]
        )
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


def _pictures(args: dict) -> list[tuple[bytes, str]]:
    """The pictures an edit goes by, as (bytes, mime): the step's source first, then its references. Each must be valid
    base64 and under the API's size limit; at most `_MAX_EDIT_IMAGES` in all."""
    given = []
    if args.get("image_b64"):
        given.append((args["image_b64"], args.get("image_mime")))
    given += [(r.get("image_b64"), r.get("image_mime")) for r in args.get("reference_images") or [] if isinstance(r, dict) and r.get("image_b64")]
    if len(given) > _MAX_EDIT_IMAGES:
        raise ValueError(f"An edit goes by up to {_MAX_EDIT_IMAGES} pictures; this has {len(given)}.")
    out = []
    for raw, mime in given:
        try:
            data = base64.b64decode(raw.split(",", 1)[-1] if raw.startswith("data:") else raw, validate=True)
        except ValueError as e:
            raise ValueError("A picture isn't valid base64.") from e
        if not data:
            raise ValueError("A picture is empty.")
        if len(data) > _MAX_EDIT_BYTES:
            raise ValueError("A picture is over 50 MB, the most the edits API takes.")
        out.append((data, mime or "image/png"))
    return out


def _multipart(fields: dict, files: list[tuple[str, str, str, bytes]]) -> tuple[bytes, str]:
    """A multipart/form-data body (the edits endpoint takes each picture as a file part) and its Content-Type."""
    boundary = f"marvin-{uuid.uuid4().hex}"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    for name, filename, mime, data in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
