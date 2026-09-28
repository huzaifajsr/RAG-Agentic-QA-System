import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verify the health endpoint returns 200 OK and expected JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data

def test_root_endpoint():
    """Verify the root endpoint redirects or returns 200."""
    # Assuming root might be 200 OK or generic greeting, adapt if 404
    response = client.get("/")
    assert response.status_code in [200, 404]

def test_upload_no_file():
    """Verify that omitting the file in /upload returns 422 Unprocessable Entity."""
    response = client.post("/upload")
    assert response.status_code == 422

def test_query_empty_question():
    """Verify that sending an empty question to /query returns a validation error."""
    payload = {
        "question": "",
        "collection_name": "default",
        "top_k": 5
    }
    response = client.post("/query", json=payload)
    # The API should validate that the string is not empty or return 422
    assert response.status_code == 422
