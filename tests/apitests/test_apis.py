import os
from fastapi.testclient import TestClient
from eurosat_classifier.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_forest():
    """
    Test prediction using the local Forest_1.jpg image.
    """
    # Ensure the file exists before trying to open it
    file_path = "tests/apitests/images/Forest_1.jpg"
    assert os.path.exists(file_path), f"File not found: {file_path}"

    with open(file_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("Forest_1.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    
    assert data["filename"] == "Forest_1.jpg"
    assert "prediction" in data
    # Optional: Check if the model actually predicted Forest
    # assert data["prediction"] == "Forest" 
    assert data["meta_max_length"] == 10  # Default value

def test_predict_river_custom_param():
    """
    Test prediction using River_1.jpg with a custom query parameter.
    """
    file_path = "tests/apitests/images/River_1.jpg"
    assert os.path.exists(file_path), f"File not found: {file_path}"

    with open(file_path, "rb") as f:
        response = client.post(
            "/predict",
            params={"max_length": 42},  # Testing the query parameter
            files={"file": ("River_1.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    
    assert data["filename"] == "River_1.jpg"
    assert "prediction" in data
    assert data["meta_max_length"] == 42