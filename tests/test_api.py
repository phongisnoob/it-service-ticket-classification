import pytest

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():

    with TestClient(app) as client:
        yield client


def test_root(client):

    response = client.get("/")

    assert response.status_code == 200


def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    assert "model_backend" in data


def test_predict(client):

    response = client.post(
        "/predict",
        json={
            "text":
                "I cannot access the shared network folder"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "category" in data
    assert "confidence" in data
    assert "threshold" in data
    assert "needs_manual_review" in data
    assert "top_3" in data

    assert isinstance(
        data["category"],
        str,
    )

    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["threshold"] <= 1

    assert isinstance(
        data["needs_manual_review"],
        bool,
    )

    assert len(
        data["top_3"]
    ) <= 3


def test_routing_logic(client):

    response = client.post(
        "/predict",
        json={
            "text":
                "I need administrator access "
                "to install an application"
        },
    )

    data = response.json()

    expected_review = (
        data["confidence"]
        <
        data["threshold"]
    )

    assert (
        data["needs_manual_review"]
        ==
        expected_review
    )


def test_top3_sorted(client):

    response = client.post(
        "/predict",
        json={
            "text":
                "My laptop keyboard "
                "is not working"
        },
    )

    data = response.json()

    probabilities = [
        item["probability"]
        for item in data["top_3"]
    ]

    assert probabilities == sorted(
        probabilities,
        reverse=True,
    )


def test_empty_ticket(client):

    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422