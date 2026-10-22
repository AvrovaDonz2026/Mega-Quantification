import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r6_native_response", HERE / "native_response.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class FakeTokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [
            {"role": "user", "content": "local deterministic prompt"}
        ] or messages == [{"role": "user", "content": "hi"}]
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert type(kwargs["enable_thinking"]) is bool
        self.template_calls.append(kwargs["enable_thinking"])
        return [11, 12]


def setup_evidence(tmp_path):
    exported = tmp_path / "export"
    exported.mkdir()
    (exported / "config.json").write_text("{}")
    tree = native._runner()._tree_hash(exported)
    evidence_path = tmp_path / "arm.json"
    evidence_path.write_text(
        json.dumps(
            {
                "arm": "static-max",
                "execution": "complete",
                "export": tree,
                "export_index_verification": {"keys_validated": True},
            }
        )
    )
    return evidence_path, exported


def fake_request(exported, response=None, model_path=None):
    calls = []

    def request(url, *, timeout, payload=None):
        calls.append((url, timeout, payload))
        if url.endswith("/get_model_info"):
            return {"model_path": model_path or str(exported)}
        return response or {"prompt_token_ids": [11, 12], "output_ids": [21, 22]}

    return request, calls


def test_collect_writes_only_sanitized_evidence_atomically(tmp_path):
    evidence_path, exported = setup_evidence(tmp_path)
    request, calls = fake_request(exported)
    output = tmp_path / "attached.json"
    tokenizer = FakeTokenizer()

    result = native.collect(
        evidence_path=evidence_path,
        base_url="http://127.0.0.1:30000",
        prompt="local deterministic prompt",
        timeout=9,
        max_new_tokens=64,
        output_path=output,
        tokenizer=tokenizer,
        request_json=request,
    )

    saved = json.loads(result.read_text())
    assert result == output
    assert [call[0].rsplit("/", 1)[-1] for call in calls] == ["get_model_info", "generate"]
    assert calls[1][2]["sampling_params"] == {"temperature": 0.0, "max_new_tokens": 64}
    assert calls[1][2]["input_ids"] == [11, 12]
    assert "text" not in calls[1][2]
    assert saved["response_observation"]["response_count"] == 1
    assert saved["trajectory_metrics"]["output_tokens"] == 2
    assert saved["text_only_protocol"]["text_only"] is True
    assert saved["response_thinking_enabled"] is True
    assert tokenizer.template_calls == [True]
    assert saved["response_loaded_weights_verified"] is False
    assert saved["rendered_prompt_ids_sha256"] == native._runner()._ids_sha256([11, 12])
    assert len(saved["response_server_model_path_sha256"]) == 64
    assert "response_server_model_path" not in saved
    assert '"output_ids":' not in output.read_text()
    assert '"prompt_token_ids":' not in output.read_text()
    assert evidence_path.read_text() != output.read_text()


def test_thinking_defaults_on_and_can_be_disabled(tmp_path):
    evidence_path, exported = setup_evidence(tmp_path)
    tokenizer = FakeTokenizer()
    request, _ = fake_request(exported)
    native.collect(
        evidence_path=evidence_path,
        base_url="http://127.0.0.1:30000",
        prompt="local deterministic prompt",
        tokenizer=tokenizer,
        request_json=request,
        output_path=tmp_path / "thinking.json",
    )
    native.collect(
        evidence_path=evidence_path,
        base_url="http://127.0.0.1:30000",
        prompt="local deterministic prompt",
        enable_thinking=False,
        tokenizer=tokenizer,
        request_json=request,
        output_path=tmp_path / "no-thinking.json",
    )
    assert tokenizer.template_calls == [True, False]
    enabled = json.loads((tmp_path / "thinking.json").read_text())
    disabled = json.loads((tmp_path / "no-thinking.json").read_text())
    assert enabled["response_thinking_enabled"] is True
    assert disabled["response_thinking_enabled"] is False
    assert enabled["response_protocol_sha256"] != disabled["response_protocol_sha256"]


def test_collect_rejects_non_loopback_before_requests(tmp_path):
    evidence_path, exported = setup_evidence(tmp_path)
    request, calls = fake_request(exported)
    with pytest.raises(ValueError, match="loopback"):
        native.collect(
            evidence_path=evidence_path,
            base_url="http://example.com:30000",
            prompt="hi",
            tokenizer=FakeTokenizer(),
            request_json=request,
        )
    assert calls == []


def test_collect_rejects_server_model_path_mismatch(tmp_path):
    evidence_path, exported = setup_evidence(tmp_path)
    request, calls = fake_request(exported, model_path=str(tmp_path / "other"))
    with pytest.raises(ValueError, match="model_path"):
        native.collect(
            evidence_path=evidence_path,
            base_url="http://localhost:30000",
            prompt="hi",
            tokenizer=FakeTokenizer(),
            request_json=request,
        )
    assert len(calls) == 1


def test_collect_rejects_changed_export_identity(tmp_path):
    evidence_path, exported = setup_evidence(tmp_path)
    (exported / "config.json").write_text('{"changed": true}')
    request, calls = fake_request(exported)
    with pytest.raises(ValueError, match="fingerprint"):
        native.collect(
            evidence_path=evidence_path,
            base_url="http://127.0.0.1:30000",
            prompt="hi",
            tokenizer=FakeTokenizer(),
            request_json=request,
        )
    assert calls == []


def test_collect_requires_native_prompt_and_output_ids(tmp_path):
    evidence_path, exported = setup_evidence(tmp_path)
    request, _ = fake_request(exported, response={"text": "generated"})
    with pytest.raises(ValueError, match="prompt token IDs"):
        native.collect(
            evidence_path=evidence_path,
            base_url="http://127.0.0.1:30000",
            prompt="hi",
            tokenizer=FakeTokenizer(),
            request_json=request,
        )

    request, _ = fake_request(
        exported, response={"prompt_token_ids": [11, 12], "text": "generated"}
    )
    with pytest.raises(ValueError, match="output token IDs"):
        native.collect(
            evidence_path=evidence_path,
            base_url="http://127.0.0.1:30000",
            prompt="hi",
            tokenizer=FakeTokenizer(),
            request_json=request,
        )


def test_collect_rejects_native_prompt_id_mismatch(tmp_path):
    evidence_path, exported = setup_evidence(tmp_path)
    request, _ = fake_request(exported, response={"prompt_token_ids": [99], "output_ids": [21]})
    with pytest.raises(ValueError, match="do not match"):
        native.collect(
            evidence_path=evidence_path,
            base_url="http://127.0.0.1:30000",
            prompt="hi",
            tokenizer=FakeTokenizer(),
            request_json=request,
        )


def test_redirect_handler_refuses_redirects():
    handler = native._NoRedirectHandler()
    request = native.urllib.request.Request("http://127.0.0.1:30000/get_model_info")
    assert (
        handler.redirect_request(
            request, None, 302, "Found", {"Location": "http://example.com/"}, "http://example.com/"
        )
        is None
    )
