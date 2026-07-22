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

        self.last = {"url": url, "json": json, "headers": headers, "timeout": timeout}
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
    http = _StubHttp()
    out = OpenAIImagesProvider().run_action("generate", {"prompt": "a rustic workshop", "count": 1}, _ctx(http=http))
    assert out == {"images": [{"image_b64": _FAKE_B64}]}
    # canonical inputs → OpenAI request shape; generous timeout for slow generation
    assert http.last["json"]["prompt"] == "a rustic workshop"
    assert http.last["json"]["n"] == 1
    assert http.last["headers"]["Authorization"].startswith("Bearer ")
    assert http.last["timeout"] == 120


def test_generate_passes_through_url_form():
    http = _StubHttp(payload={"data": [{"url": "https://img.example/1.png"}]})
    out = OpenAIImagesProvider().run_action("generate", {"prompt": "x"}, _ctx(http=http))
    assert out == {"images": [{"url": "https://img.example/1.png"}]}


def test_generate_requires_prompt():
    with pytest.raises(ValueError):
        OpenAIImagesProvider().run_action("generate", {}, _ctx())


def test_http_error_surfaces():
    http = _StubHttp(status=401, payload={"error": {"message": "bad key"}})
    with pytest.raises(ValueError):
        OpenAIImagesProvider().run_action("generate", {"prompt": "x"}, _ctx(http=http))
