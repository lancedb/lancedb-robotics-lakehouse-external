"""Adapter registry: identity and capability declarations for log-format adapters.

An adapter owns one robot-log format (MCAP first) and declares which
capabilities it supports. The registry is how the CLI and ingest discover
what the package can do with a given format, instead of hard-coding format
knowledge at each call site.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Known capability vocabulary. Adapters may implement a subset; declaring a
# capability outside this set is a registration error.
CAPABILITIES: tuple[str, ...] = ("inspect", "ingest", "export")


class AdapterError(Exception):
    """Raised for unknown adapters, bad registrations, or failed adapter operations."""


class CodecUnavailableError(AdapterError):
    """A chunk uses a compression codec whose decoder is not installed (backlog 0017).

    Distinct from corruption: the bytes are fine, the environment is missing the
    codec. Carries the codec name so the message can name the package to install.
    The fix is to install the codec, so this is a hard error, never quarantined.
    """

    def __init__(self, message: str, *, codec: str) -> None:
        super().__init__(message)
        self.codec = codec


class CorruptMcapError(AdapterError):
    """The MCAP byte stream is damaged: a CRC mismatch or a truncation (backlog 0017).

    Unlike a codec gap, corruption is data the ingest path can *partly* keep: the
    error is raised only after the readable prefix has been yielded, and it
    carries ``status`` (``crc-mismatch`` | ``truncated``), a human ``reason``, and
    ``recovered`` (how many messages were yielded before the damage) so the run is
    quarantined rather than lost.
    """

    def __init__(
        self, message: str, *, status: str, reason: str | None = None, recovered: int = 0
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.recovered = recovered


@dataclass(frozen=True)
class AdapterInfo:
    """Adapter identity: registry name, file format handled, declared capabilities."""

    name: str
    format: str
    capabilities: tuple[str, ...]


@runtime_checkable
class Adapter(Protocol):
    info: AdapterInfo


class AdapterRegistry:
    """Holds adapters by name and validates capability declarations on register."""

    def __init__(self) -> None:
        self._adapters: dict[str, Adapter] = {}

    def register(self, adapter: Adapter) -> None:
        info = adapter.info
        if info.name in self._adapters:
            raise AdapterError(f"adapter {info.name!r} is already registered")
        unknown = [c for c in info.capabilities if c not in CAPABILITIES]
        if unknown:
            raise AdapterError(f"adapter {info.name!r} declares unknown capabilities: {unknown}")
        missing = [c for c in info.capabilities if not callable(getattr(adapter, c, None))]
        if missing:
            raise AdapterError(
                f"adapter {info.name!r} declares capabilities without methods: {missing}"
            )
        self._adapters[info.name] = adapter

    def get(self, name: str) -> Adapter:
        try:
            return self._adapters[name]
        except KeyError:
            known = ", ".join(sorted(self._adapters)) or "<none>"
            raise AdapterError(f"unknown adapter {name!r}; registered adapters: {known}") from None

    def list(self) -> list[AdapterInfo]:
        return [self._adapters[name].info for name in sorted(self._adapters)]


registry = AdapterRegistry()


def get_adapter(name: str) -> Adapter:
    """Return the registered adapter for ``name`` or raise :class:`AdapterError`."""
    return registry.get(name)


def list_adapters() -> list[AdapterInfo]:
    """Return registered adapter identities, sorted by name."""
    return registry.list()


def _register_builtins() -> None:
    from lancedb_robotics.adapters.lerobot_adapter import LeRobotAdapter
    from lancedb_robotics.adapters.mcap_adapter import McapAdapter
    from lancedb_robotics.adapters.rosbag_adapter import RosBagAdapter

    registry.register(LeRobotAdapter())
    registry.register(McapAdapter())
    registry.register(RosBagAdapter())


_register_builtins()
