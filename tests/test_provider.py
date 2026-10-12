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

    def get(self, url, *, headers=None, timeout=15):  # pragma: no cover
        raise AssertionError("no GET expected")

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


def test_check_logic():
    p = OpenAIImagesProvider()
    assert p.check(_ctx(secret="sk-abc")) == ("ok", None)
    assert p.check(_ctx(secret=None))[0] == "unconfigured"
    assert p.check(_ctx(secret="nope"))[0] == "error"


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
