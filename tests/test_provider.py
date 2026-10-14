"""Tests for the OpenAI Images capability provider — no network."""

import base64
import logging

import pytest
from marvin_integration_sdk import IntegrationContext, Response

from marvin_integration_openai_images import OpenAIImagesProvider

_LOG = logging.getLogger("test")
_FAKE_B64 = base64.b64encode(b"fake-png").decode()


class _StubHttp:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload if payload is not None else {"data": [{"b64_json": _FAKE_B64}]}
        self.last = None

    def get(self, url, *, headers=None, timeout=15):
        import json as _json

        self.last_get = {"url": url, "headers": headers}
        if url.endswith("/v1/models"):
            return Response(
                status_code=200,
                content=_json.dumps(
                    {"data": [{"id": m} for m in ("gpt-4.1", "gpt-image-1", "gpt-image-1-mini", "gpt-image-2", "dall-e-3")]}
                ).encode(),
            )
        model = url.rsplit("/", 1)[1]
        if self.status == 401:
            return Response(status_code=401, content=b'{"error":"bad key"}')
        return Response(status_code=200 if model in ("gpt-image-1", "gpt-image-1-mini", "gpt-image-2", "dall-e-3") else 404, content=b"{}")

    def post(self, url, *, json=None, data=None, headers=None, timeout=15):
        import json as _json

        self.last = {"url": url, "json": json, "data": data, "headers": headers, "timeout": timeout}
        return Response(status_code=self.status, content=_json.dumps(self.payload).encode())


def _ctx(secret="sk-test", config=None, http=None):
    return IntegrationContext(config=config or {}, secret=secret, logger=_LOG, http=http or _StubHttp())


def test_advertises_image_generate_capability():
    action = OpenAIImagesProvider().actions[0]
    assert action.capability == "image.generate"
    assert action.requires_approval is True
    assert action.cost_hint == "paid"


def test_check_tries_the_key_and_the_models_availability():
    p = OpenAIImagesProvider()
    assert p.check(_ctx(secret=None))[0] == "unconfigured"
    assert p.check(_ctx(secret="nope"))[0] == "error"
    assert p.check(_ctx(config={"model": "gpt-image-2"})) == ("ok", None)
    status, why = p.check(_ctx(config={"model": "gpt-image-9"}))
    assert status == "error" and "gpt-image-9" in why and "List image models" in why
    status, why = p.check(_ctx(http=_StubHttp(status=401)))
    assert status == "error" and "rejected" in why


def test_list_models_is_free_and_says_whether_the_configured_model_is_usable():
    action = next(a for a in OpenAIImagesProvider.actions if a.key == "list_models")
    assert action.cost_hint == "free" and not action.requires_approval
    out = OpenAIImagesProvider().run_action("list_models", {}, _ctx(config={"model": "gpt-image-2"}))
    assert out == {"models": ["dall-e-3", "gpt-image-1", "gpt-image-1-mini", "gpt-image-2"], "configured": "gpt-image-2", "available": True}
    assert OpenAIImagesProvider().run_action("list_models", {}, _ctx(config={"model": "gpt-image-9"}))["available"] is False


def test_generate_sends_quality_format_and_background_for_gpt_image_models():
    http = _StubHttp()
    OpenAIImagesProvider().run_action(
        "generate",
        {"prompt": "a lantern", "quality": "high", "output_format": "jpeg"},
        _ctx(config={"model": "gpt-image-2", "quality": "low"}, http=http),
    )
    assert http.last["json"]["quality"] == "high" and http.last["json"]["output_format"] == "jpeg" and "background" not in http.last["json"]
    http = _StubHttp()
    OpenAIImagesProvider().run_action(
        "generate",
        {"prompt": "a lantern", "background": "transparent"},
        _ctx(config={"model": "gpt-image-1-mini", "output_format": "webp"}, http=http),
    )
    assert (http.last["json"]["quality"], http.last["json"]["output_format"], http.last["json"]["background"]) == ("auto", "webp", "transparent")
    http = _StubHttp()
    OpenAIImagesProvider().run_action("generate", {"prompt": "a lantern", "quality": "high"}, _ctx(config={"model": "dall-e-3"}, http=http))
    assert "quality" not in http.last["json"] and "output_format" not in http.last["json"]  # dall-e takes none of these
    with pytest.raises(ValueError, match="transparent background needs png or webp"):
        OpenAIImagesProvider().run_action(
            "generate", {"prompt": "x", "background": "transparent", "output_format": "jpeg"}, _ctx(config={"model": "gpt-image-2"})
        )
    with pytest.raises(ValueError, match="quality must be one of"):
        OpenAIImagesProvider().run_action("generate", {"prompt": "x", "quality": "ultra"}, _ctx(config={"model": "gpt-image-2"}))


def test_the_guide_ships_and_names_the_models():
    from marvin_integration_sdk import load_guide

    guide = load_guide(OpenAIImagesProvider)
    assert guide.startswith("# OpenAI Images") and "gpt-image-2" in guide and "gpt-image-1-mini" in guide and "List image models" in guide


def test_generate_returns_canonical_output():
    http = _StubHttp(payload={"data": [{"b64_json": _FAKE_B64}], "usage": {"input_tokens": 10, "output_tokens": 100}})
    out = OpenAIImagesProvider().run_action("generate", {"prompt": "a rustic workshop", "count": 1}, _ctx(http=http))
    assert out["images"] == [{"image_b64": _FAKE_B64}]
    # usage surfaced for core cost logging
    assert out["usage"]["provider_type"] == "openai"
    assert out["usage"]["input_tokens"] == 10 and out["usage"]["output_tokens"] == 100
    # canonical inputs → OpenAI request shape; generous timeout for slow generation
    assert http.last["json"]["prompt"] == "a rustic workshop"
    assert http.last["json"]["n"] == 1
    assert http.last["headers"]["Authorization"].startswith("Bearer ")
    assert http.last["timeout"] == 120


def test_generate_passes_through_url_form():
    http = _StubHttp(payload={"data": [{"url": "https://img.example/1.png"}]})
    out = OpenAIImagesProvider().run_action("generate", {"prompt": "x"}, _ctx(http=http))
    assert out["images"] == [{"url": "https://img.example/1.png"}]


def test_generate_requires_prompt():
    with pytest.raises(ValueError):
        OpenAIImagesProvider().run_action("generate", {}, _ctx())


def test_http_error_surfaces():
    http = _StubHttp(status=401, payload={"error": {"message": "bad key"}})
    with pytest.raises(ValueError):
        OpenAIImagesProvider().run_action("generate", {"prompt": "x"}, _ctx(http=http))


# ── edit ──────────────────────────────────────────────────────────────────────
_PICTURE = b"\x89PNG-the-product-photo"


def _edit(http, **args):
    base = {"prompt": "Remove the background", "image_b64": base64.b64encode(_PICTURE).decode(), "image_mime": "image/png"}
    return OpenAIImagesProvider().run_action("edit", {**base, **args}, _ctx(http=http))


def test_edit_is_an_image_edit_capability_that_needs_approval():
    action = next(a for a in OpenAIImagesProvider().actions if a.key == "edit")
    assert action.capability == "image.edit" and action.requires_approval is True and action.cost_hint == "paid"
    assert action.input_schema["required"] == ["prompt", "image_b64"]


def test_edit_sends_the_picture_as_a_file_and_returns_one_image():
    http = _StubHttp(payload={"data": [{"b64_json": _FAKE_B64}], "usage": {"input_tokens": 5, "output_tokens": 50}})

    out = _edit(http, background="transparent")

    assert out["image_b64"] == _FAKE_B64 and out["usage"]["output_tokens"] == 50
    assert http.last["url"].endswith("/v1/images/edits") and http.last["json"] is None
    content_type = http.last["headers"]["Content-Type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    body = http.last["data"]
    assert _PICTURE in body and b'name="image"; filename="image.png"' in body
    for field, value in (
        ("prompt", b"Remove the background"),
        ("background", b"transparent"),
        ("output_format", b"png"),
        ("input_fidelity", b"high"),
    ):
        assert f'name="{field}"\r\n\r\n'.encode() + value in body, field


def test_edit_without_a_picture_or_with_an_error_says_so():
    with pytest.raises(ValueError, match="set the workflow step's source"):
        _edit(_StubHttp(), image_b64="")
    with pytest.raises(ValueError, match="HTTP 400"):
        _edit(_StubHttp(status=400, payload={"error": {"message": "bad"}}))
