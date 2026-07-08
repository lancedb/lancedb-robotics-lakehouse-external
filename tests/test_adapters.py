"""Adapter registry capability-discovery tests (backlog 0003)."""

import pytest

from lancedb_robotics.adapters import (
    AdapterError,
    AdapterInfo,
    get_adapter,
    list_adapters,
)


def test_registry_discovers_mcap_adapter():
    assert "mcap" in [info.name for info in list_adapters()]


def test_registry_discovers_rosbag_adapter():
    assert "rosbag" in [info.name for info in list_adapters()]


def test_mcap_adapter_declares_identity_and_capabilities():
    adapter = get_adapter("mcap")
    assert isinstance(adapter.info, AdapterInfo)
    assert adapter.info.name == "mcap"
    assert adapter.info.format == "mcap"
    assert "inspect" in adapter.info.capabilities


def test_rosbag_adapter_declares_identity_and_capabilities():
    adapter = get_adapter("rosbag")
    assert isinstance(adapter.info, AdapterInfo)
    assert adapter.info.name == "rosbag"
    assert adapter.info.format == "rosbag"
    assert {"inspect", "ingest"} <= set(adapter.info.capabilities)


def test_mcap_adapter_implements_declared_inspect():
    adapter = get_adapter("mcap")
    assert callable(adapter.inspect)


def test_rosbag_adapter_implements_declared_capabilities():
    adapter = get_adapter("rosbag")
    assert callable(adapter.inspect)
    assert callable(adapter.ingest)


def test_unknown_adapter_raises():
    with pytest.raises(AdapterError):
        get_adapter("asam-md4")


def test_list_adapters_is_sorted_by_name():
    names = [info.name for info in list_adapters()]
    assert names == sorted(names)
