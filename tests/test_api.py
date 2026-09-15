from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "personalization-ranking-engine",
    }


def test_known_user_recommendations():
    response = client.post(
        "/recommend",
        json={"visitorid": 1, "k": 10},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["visitorid"] == 1
    assert len(body["recommendations"]) == 10

    for recommendation in body["recommendations"]:
        assert "itemid" in recommendation
        assert "ltr_score" in recommendation
        assert "mmr_score" in recommendation


def test_unknown_user_cold_start():
    response = client.post(
        "/recommend",
        json={"visitorid": 999999999, "k": 10},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["visitorid"] == 999999999
    assert len(body["recommendations"]) == 10

    for recommendation in body["recommendations"]:
        assert recommendation["ltr_score"] is None
        assert recommendation["mmr_score"] is None


def test_invalid_k_below_minimum():
    response = client.post(
        "/recommend",
        json={"visitorid": 1, "k": 0},
    )

    assert response.status_code == 422


def test_invalid_k_above_maximum():
    response = client.post(
        "/recommend",
        json={"visitorid": 1, "k": 101},
    )

    assert response.status_code == 422


def test_maximum_k():
    response = client.post(
        "/recommend",
        json={"visitorid": 1, "k": 100},
    )

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 100
