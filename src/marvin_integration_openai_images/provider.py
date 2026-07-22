"""OpenAI image generation — a capability provider advertising `image.generate`.

The media-enrichment resolver discovers this via ``integrations_providing("image.generate", …)``
and invokes it with the canonical shape: in ``{prompt, count?}`` → out ``{images: [{image_b64}]}``.
Pure with respect to Marvin — it gets the API key + config + http, and returns image bytes; the
resolver persists them.
"""

from marvin_integration_sdk import (
    CATEGORY_CAPABILITY,
    CredentialField,
    IntegrationContext,
    IntegrationProvider,
    ProviderAction,
    register_provider,
)

_ENDPOINT = "https://api.openai.com/v1/images/generations"
# Image generation is slow — set a generous timeout (the resolver caps wall-clock on its side).
_TIMEOUT = 120


@register_provider
class OpenAIImagesProvider(IntegrationProvider):
    slug = "openai_images"
    name = "OpenAI Images"
    description = "Generate images from a text prompt via the OpenAI image API."
    category = CATEGORY_CAPABILITY

    credentials = (
        CredentialField(key="api_key", label="OpenAI API Key", help="An OpenAI API key (sk-…) with image access."),
    )
    config_schema = {
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
                "properties": {
                    "images": {"type": "array", "items": {"type": "object", "properties": {"image_b64": {"type": "string"}}}}
                },
            },
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
        if key != "generate":
            raise NotImplementedError(f"openai_images has no action '{key}'")
        if not ctx.secret:
            raise ValueError("No OpenAI API key configured.")
        prompt = (args or {}).get("prompt")
        if not prompt:
            raise ValueError("A 'prompt' is required.")

        cfg = ctx.config or {}
        model = cfg.get("model") or "gpt-image-1"
        body = {"model": model, "prompt": prompt, "n": int((args or {}).get("count") or 1), "size": cfg.get("size") or "1024x1024"}
        # dall-e-* return a URL unless asked for b64; gpt-image-1 returns b64 by default.
        if model.startswith("dall-e"):
            body["response_format"] = "b64_json"

        try:
            resp = ctx.http.post(_ENDPOINT, json=body, headers={"Authorization": f"Bearer {ctx.secret}"}, timeout=_TIMEOUT)
        except Exception as e:  # noqa: BLE001 — surface transport/guard failures cleanly
            ctx.logger.warning(f"[openai_images] request failed: {e}")
            raise ValueError(f"OpenAI images request failed: {e}") from e

        if not resp.ok:
            raise ValueError(f"OpenAI images returned HTTP {resp.status_code}: {resp.text[:200]}")

        images: list[dict] = []
        for item in resp.json().get("data", []):
            if item.get("b64_json"):
                images.append({"image_b64": item["b64_json"]})
            elif item.get("url"):
                images.append({"url": item["url"]})
        return {"images": images}
