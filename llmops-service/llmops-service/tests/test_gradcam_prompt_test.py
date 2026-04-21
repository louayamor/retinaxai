from app.pipeline.xai_pipeline import XAIPipeline


def test_build_gradcam_prompt_formats_numeric_regions() -> None:
    pipeline = XAIPipeline.__new__(XAIPipeline)

    prompt = pipeline._build_gradcam_prompt(  # type: ignore[attr-defined]
        [
            {
                "name": "macula_center",
                "intensity": 0.82,
                "area": 123,
                "center_x": 55,
                "center_y": 64,
                "saliency_score": 0.21,
            }
        ],
        ["temporal_arcade"],
    )

    assert "macula_center" in prompt
    assert "intensity: 0.82" in prompt
    assert "temporal_arcade" in prompt
