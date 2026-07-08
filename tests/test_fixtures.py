import json


def test_mini_run_manifest_is_deterministic(fixtures_dir):
    manifest = json.loads((fixtures_dir / "mini_run_manifest.json").read_text())
    assert manifest["run_id"] == "demo-run-0001"
    assert [t["topic"] for t in manifest["topics"]] == [
        "/camera/front/image",
        "/imu/data",
        "/tf",
    ]
    assert sum(t["message_count"] for t in manifest["topics"]) == 42
