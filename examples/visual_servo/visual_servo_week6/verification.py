"""Reproducible software checks, with synthetic results separated from real calibration."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .calibration import save_yaml
from .geometry import rotation_error_rad
from .hand_eye_workflow import append_pair, create_session, solve_session
from .synthetic import generate_board_views
from .task3_simulation import run_task3_simulation


def verify_software(root, intrinsics, output):
    root, output = Path(root), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(root / "pytest.ini"),
            str(root / "tests"),
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={output / 'tests.xml'}",
        ],
        cwd=root.parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    (output / "tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    if test.returncode:
        raise RuntimeError(f"软件测试失败，请检查 {output / 'tests.txt'}")
    images = output / "task2-synthetic/source-images"
    records, matrix, expected, board = generate_board_views(images, inner_corners=(8, 6), count=16)
    calibration = output / "task2-synthetic/intrinsics.yaml"
    save_yaml(
        calibration,
        {
            "K": matrix.tolist(),
            "distortion": [0.0] * 5,
            "image_size": [1280, 720],
            "board_inner_corners": [8, 6],
            "square_size_m": 0.025,
            "data_source": "synthetic",
        },
    )
    directory = output / "task2-synthetic/pairs"
    create_session(directory, calibration)
    for index, (image, record) in enumerate(records.items()):
        append_pair(
            directory,
            image,
            {
                "T_base_from_end": record["T_base_from_end"].tolist(),
                "translation_unit": "metre",
                "end_frame": "synthetic_end",
                "base_frame": "synthetic_base",
                "source": "synthetic_ground_truth",
            },
            validation=index >= 12,
        )
    result = solve_session(directory, output / "task2-synthetic/result", board[:3, 3])
    estimated = np.asarray(result["T_end_from_camera"])
    error_m = float(np.linalg.norm(estimated[:3, 3] - expected[:3, 3]))
    error_deg = float(np.rad2deg(rotation_error_rad(estimated[:3, :3], expected[:3, :3])))
    task3 = run_task3_simulation(intrinsics, output / "task3-simulation")
    report = {
        "physical_hardware_used": False,
        "physical_task2_completed": False,
        "physical_task3_completed": False,
        "unit_tests_passed": True,
        "task2_synthetic": {
            "training_pairs": 12,
            "validation_pairs": 4,
            "translation_error_m": error_m,
            "rotation_error_deg": error_deg,
            "validated_for_physical_servo": result["validated_for_servo"],
            "passed": error_m < 0.01 and error_deg < 2,
        },
        "task3_simulation": task3,
    }
    report["passed"] = report["task2_synthetic"]["passed"] and task3["passed"]
    (output / "software_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError("离线验证未通过")
    return report
