HANDBOOK = """Vacation policy

Employees get 30 days of paid vacation per year. Requests go to your team lead.

Expenses

Travel expenses are reimbursed within 14 days. Questions: finance@acme.example or +49 69 1234 5678.

IT security

Laptops must use full-disk encryption. Report lost devices to the IT helpdesk immediately."""


def test_health(client, llm):
    assert client.get("/health").json()["status"] == "ok"
    llm.up = False
    assert client.get("/health").json()["status"] == "degraded"


def test_ask_before_ingest_returns_409(client):
    assert client.post("/ask", json={"question": "How many vacation days?"}).status_code == 409


def test_ingest_redacts_personal_data(client):
    r = client.post("/documents", json={"source": "handbook.md", "text": HANDBOOK})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] >= 2
    assert body["redacted"] == {"EMAIL": 1, "PHONE": 1}


def test_ask_retrieves_relevant_chunk_and_hides_pii(client, llm):
    client.post("/documents", json={"source": "handbook.md", "text": HANDBOOK})
    r = client.post(
        "/ask",
        json={"question": "How many vacation days? My email is jane@acme.example"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "vacation" in body["sources"][0]["excerpt"].lower()
    assert body["redacted"] == {"EMAIL": 1}
    # Nothing personal ever reaches the model.
    prompt = llm.prompts[-1]
    assert "jane@acme.example" not in prompt
    assert "finance@acme.example" not in prompt
    assert "1234 5678" not in prompt


def test_reingest_replaces_and_delete_removes(client):
    client.post("/documents", json={"source": "a.md", "text": "first version"})
    client.post("/documents", json={"source": "a.md", "text": "second version"})
    docs = client.get("/documents").json()
    assert docs == {"sources": ["a.md"], "chunks": 1}
    assert client.delete("/documents/a.md").status_code == 200
    assert client.delete("/documents/a.md").status_code == 404


def test_metrics_endpoint(client):
    client.get("/health")
    text = client.get("/metrics").text
    assert "rag_requests_total" in text
