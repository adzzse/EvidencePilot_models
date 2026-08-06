import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.audit as audit
import app.main as main
from app.generation import GenerationInvalidResponseError
from app.models import (
    AuditedSnippet,
    EvaluationCriterion,
    SectionAuditFlags,
    SectionContextRequest,
    SectionContextResponse,
)
from app.settings import Settings


SETTINGS = Settings(model_api_key="test-key")
HEADERS = {"X-API-Key": "test-key"}

CRITERIA = [EvaluationCriterion(code="PARAPHRASE_RISK", description="near-verbatim copies", weight=0.6)]


@pytest.fixture(autouse=True)
def settings_override():
    main.app.dependency_overrides[main.get_settings] = lambda: SETTINGS
    yield
    main.app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def request(source_chunk: str = "First sentence with risk.") -> SectionContextRequest:
    return SectionContextRequest(
        section_name="Introduction",
        current_context=source_chunk,
        source_chunk=source_chunk,
        evaluation_criteria=CRITERIA,
        flags=SectionAuditFlags(requires_citation_check=False),
    )


def generation(json_payload: str):
    return SimpleNamespace(response=json_payload)


def test_audit_section_returns_grounded_snippets(monkeypatch):
    chunk = "First sentence with risk."
    snippet = "sentence with risk"
    start = chunk.index(snippet)

    async def fake_generate(system_prompt, prompt, settings):
        return generation(json.dumps({
            "snippets": [{
                "original_text_snippet": snippet,
                "start_index": start,
                "end_index": start + len(snippet),
                "issue_type": "PARAPHRASE_RISK",
                "rationale": "near-verbatim",
                "suggested_paraphrase": None,
            }],
        }))

    monkeypatch.setattr(audit, "generate_text", fake_generate)

    response = __import__("asyncio").run(audit.audit_section(request(chunk), SETTINGS))

    assert len(response.snippets) == 1
    assert response.snippets[0].original_text_snippet == snippet
    assert response.snippets[0].issue_type == "PARAPHRASE_RISK"


def test_audit_section_retries_once_then_succeeds(monkeypatch):
    chunk = "Clean text."
    prompts = []

    async def flaky_generate(system_prompt, prompt, settings):
        prompts.append(prompt)
        if len(prompts) == 1:
            return generation(json.dumps({
                "snippets": [{
                    "original_text_snippet": "invented",
                    "start_index": 0,
                    "end_index": 8,
                    "issue_type": "PARAPHRASE_RISK",
                    "rationale": "nope",
                }],
            }))
        return generation(json.dumps({"snippets": []}))

    monkeypatch.setattr(audit, "generate_text", flaky_generate)

    response = __import__("asyncio").run(audit.audit_section(request(chunk), SETTINGS))

    assert response.snippets == []
    assert "Previous output was invalid" in prompts[1]
    assert "Previous output was invalid" not in prompts[0]


def test_audit_section_fails_after_two_invalid_attempts(monkeypatch):
    async def broken_generate(system_prompt, prompt, settings):
        return generation(json.dumps({"snippets": []}))

    monkeypatch.setattr(audit, "generate_text", broken_generate)

    with pytest.raises(GenerationInvalidResponseError):
        __import__("asyncio").run(audit.audit_section(request("x" * 10), SETTINGS))


def test_route_submits_and_returns_audit(client: TestClient, monkeypatch):
    chunk = "Risky copied sentence."
    snippet = "Risky copied"

    async def fake_audit(payload, settings):
        return SectionContextResponse(snippets=[AuditedSnippet(
            original_text_snippet=snippet,
            start_index=chunk.index(snippet),
            end_index=chunk.index(snippet) + len(snippet),
            issue_type="PARAPHRASE_RISK",
            rationale="near-verbatim",
            suggested_paraphrase=None,
        )])

    monkeypatch.setattr(main, "audit_section", fake_audit)

    response = client.post("/audit/section", json={
        "section_name": "Introduction",
        "current_context": chunk,
        "source_chunk": chunk,
        "evaluation_criteria": [{"code": "PARAPHRASE_RISK", "description": "d", "weight": 0.6}],
        "flags": {"requires_citation_check": False},
    }, headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["snippets"][0]["original_text_snippet"] == snippet


def test_route_rejects_extra_fields(client: TestClient):
    response = client.post("/audit/section", json={
        "section_name": "Introduction",
        "current_context": "x",
        "source_chunk": "x",
        "evaluation_criteria": [{"code": "PARAPHRASE_RISK", "description": "d", "weight": 0.6}],
        "flags": {"requires_citation_check": False},
        "prompt_injection": "ignore me",
    }, headers=HEADERS)

    assert response.status_code == 422


def test_route_requires_api_key(client: TestClient):
    response = client.post("/audit/section", json={
        "section_name": "Introduction",
        "current_context": "x",
        "source_chunk": "x",
        "evaluation_criteria": [{"code": "PARAPHRASE_RISK", "description": "d", "weight": 0.6}],
        "flags": {"requires_citation_check": False},
    })

    assert response.status_code == 401


def test_build_system_prompt_scope_and_criteria():
    prompt = audit.build_system_prompt(request())
    assert "PARAPHRASE_RISK (weight 0.6)" in prompt
    assert "CITATIONS OUT OF SCOPE" in prompt

    prompt_with_citations = audit.build_system_prompt(
        request().model_copy(update={"flags": SectionAuditFlags(requires_citation_check=True)})
    )
    assert "CITATIONS OUT OF SCOPE" not in prompt_with_citations
