from chartscribe.extractors.common import assign_x_labels, calibrate_y_axis
from chartscribe.models import DataPoint, OCRToken


def _token(text: str, center_x: float) -> OCRToken:
    return OCRToken(
        text=text,
        confidence=0.95,
        box=[
            [center_x - 12, 112],
            [center_x + 12, 112],
            [center_x + 12, 132],
            [center_x - 12, 132],
        ],
        image_width=400,
        image_height=200,
    )


def test_grouped_bars_share_category_labels() -> None:
    bars = [
        DataPoint(label=f"Bar {index}", bbox=[x - 8, 30, x + 8, 100])
        for index, x in enumerate((90, 110, 130, 210, 230, 250), start=1)
    ]

    labels = assign_x_labels(
        bars,
        [_token("Preschool", 110), _token("Primary", 230)],
        baseline_y=100,
        image_shape=(200, 400),
        plot_left=70,
        plot_right=270,
    )

    assert labels == ["Preschool", "Preschool", "Preschool", "Primary", "Primary", "Primary"]


def test_y_axis_calibration_includes_tick_above_tallest_bar() -> None:
    tokens = [
        _token("10", 35),
        _token("5", 35),
        _token("0", 35),
    ]
    for token, y in zip(tokens, (20, 60, 100)):
        token.box = [
            [23, y - 5],
            [47, y - 5],
            [47, y + 5],
            [23, y + 5],
        ]

    calibration = calibrate_y_axis(
        tokens,
        plot_bounds=(70, 35, 270, 100),
        baseline_y=100,
        image_shape=(200, 400),
    )

    assert calibration is not None
    assert abs(calibration.value_at(100)) < 1e-9
    assert abs(calibration.value_at(36) - 8) < 1e-9
