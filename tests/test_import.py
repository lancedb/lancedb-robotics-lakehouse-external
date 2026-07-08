def test_import_package():
    import lancedb_robotics

    assert lancedb_robotics.__version__


def test_version_is_pep440_like():
    import lancedb_robotics

    parts = lancedb_robotics.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])
