from app.services.ml_client.schemas import MLPredictResponse


def test_ml_predict_response_accepts_numeric_gradcam_regions() -> None:
    payload = {
        "prediction": {"combined_grade": 3},
        "confidence_score": 0.91,
        "model_name": "efficientnet_b4",
        "model_version": "v1.0.0",
        "embedding": [0.1, 0.2, 0.3],
        "regions_left": [
            {
                "name": "macula_center",
                "intensity": 0.82,
                "area": 123,
                "center_x": 55,
                "center_y": 64,
                "saliency_score": 0.21,
            }
        ],
        "regions_right": [],
        "top_hotspots_left": [
            {"region": "macula_center", "intensity": 0.82, "rank": 1}
        ],
        "top_hotspots_right": [],
    }

    model = MLPredictResponse.model_validate(payload)

    assert model.regions_left is not None
    assert model.regions_left[0].name == "macula_center"
    assert model.top_hotspots_left is not None
    assert model.top_hotspots_left[0].rank == 1
    assert model.embedding == [0.1, 0.2, 0.3]
