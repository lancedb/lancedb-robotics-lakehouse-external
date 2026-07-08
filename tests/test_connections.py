"""Enterprise and namespace connection resolver tests (backlog 0036)."""

from types import SimpleNamespace

import pytest

from lancedb_robotics.connections import (
    ManagedVersioningMismatch,
    NamespaceConfigError,
    PylanceNamespaceAccess,
    UnsupportedLakeSchemeError,
    namespace_auth_context,
    namespace_properties_from_options,
    namespace_worker_spec,
    plan_namespace_write,
    resolve_lake_connection,
)


def test_resolver_classifies_db_uri_and_resolves_enterprise_kwargs(monkeypatch):
    monkeypatch.setenv(
        "LANCEDB_ROBOTICS_AUTH_ENTERPRISE_PROD_REMOTE_OPTIONS_JSON",
        '{"api_key": "ldb-secret", "region": "us-west-2", '
        '"host_override": "https://phalanx.acme.internal", '
        '"client_config": {"retry_config": {"retries": 3}}}',
    )

    spec = resolve_lake_connection(
        "db://robotics",
        remote_auth_ref="enterprise-prod",
    )

    assert spec.kind == "lancedb_remote_db"
    assert spec.lancedb_connect_kwargs == {
        "api_key": "ldb-secret",
        "region": "us-west-2",
        "host_override": "https://phalanx.acme.internal",
        "client_config": {"retry_config": {"retries": 3}},
    }
    assert "storage_options" not in spec.lancedb_connect_kwargs
    assert spec.capabilities.server_side_query
    assert not spec.capabilities.direct_object_io


@pytest.mark.parametrize("uri", ["lancedb://robotics", "phalanx://robotics"])
def test_resolver_rejects_invented_lancedb_schemes(uri):
    with pytest.raises(UnsupportedLakeSchemeError):
        resolve_lake_connection(uri)


def test_namespace_connection_builds_rest_properties_and_auth_provider():
    properties = namespace_properties_from_options(
        uri="https://phalanx.acme.internal",
        database="acme",
        database_prefix="robotics",
        delimiter="$",
    )

    spec = resolve_lake_connection(
        namespace_client_impl="rest",
        namespace_client_properties=properties,
        namespace_auth_ref="phalanx-prod",
        namespace_client_pushdown_operations=["QueryTable", "CreateTable", "QueryTable"],
    )

    assert spec.kind == "rest_namespace_lancedb"
    assert spec.uri is None
    assert spec.lancedb_connect_kwargs["namespace_client_impl"] == "rest"
    assert spec.lancedb_connect_kwargs["namespace_client_properties"] == {
        "uri": "https://phalanx.acme.internal",
        "header.x-lancedb-database": "acme",
        "header.x-lancedb-database-prefix": "robotics",
        "delimiter": "$",
        "dynamic_context_provider.impl": (
            "lancedb_robotics.connections.RuntimeNamespaceAuthProvider"
        ),
        "dynamic_context_provider.auth_ref": "phalanx-prod",
    }
    assert spec.lancedb_connect_kwargs["namespace_client_pushdown_operations"] == [
        "QueryTable",
        "CreateTable",
    ]
    assert spec.capabilities.server_side_query
    assert spec.capabilities.direct_object_io
    assert spec.capabilities.geneva_worker_specs


def test_namespace_pushdown_rejects_unknown_operation():
    with pytest.raises(NamespaceConfigError, match="unsupported namespace pushdown"):
        resolve_lake_connection(
            namespace_client_impl="rest",
            namespace_client_properties={"uri": "https://phalanx"},
            namespace_client_pushdown_operations=["DeleteTable"],
        )


def test_namespace_runtime_auth_context_uses_ref_without_persisting_secret(monkeypatch):
    monkeypatch.setenv(
        "LANCEDB_ROBOTICS_AUTH_PHALANX_PROD_NAMESPACE_HEADERS_JSON",
        '{"x-lancedb-database": "acme"}',
    )
    monkeypatch.setenv(
        "LANCEDB_ROBOTICS_AUTH_PHALANX_PROD_NAMESPACE_BEARER_TOKEN",
        "token-secret",
    )

    assert namespace_auth_context("phalanx-prod") == {
        "headers.x-lancedb-database": "acme",
        "headers.Authorization": "Bearer token-secret",
    }


class _FakeNamespace:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def describe_table(self, request):
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[index]


def _request_attr(request, name):
    if isinstance(request, dict):
        return request[name]
    return getattr(request, name)


def test_direct_pylance_access_describes_refreshes_and_opens_dataset():
    namespace = _FakeNamespace(
        [
            {
                "location": "s3://robotics/features.lance",
                "storage_options": {"session": "old", "expires_at_millis": "1"},
                "managed_versioning": True,
            },
            {
                "location": "s3://robotics/features.lance",
                "storage_options": {"session": "fresh", "expires_at_millis": "9999999999999"},
                "managed_versioning": True,
            },
        ]
    )
    access = PylanceNamespaceAccess(namespace, ("features", "embeddings"))
    opened = {}

    def fake_dataset(**kwargs):
        opened.update(kwargs)
        return "dataset"

    assert (
        access.open_dataset(
            dataset_factory=fake_dataset,
            expected_managed_versioning=True,
        )
        == "dataset"
    )

    assert len(namespace.requests) == 2
    assert _request_attr(namespace.requests[0], "id") == ["features", "embeddings"]
    assert _request_attr(namespace.requests[0], "vend_credentials") is True
    assert opened == {
        "namespace_client": namespace,
        "table_id": ["features", "embeddings"],
    }


def test_managed_versioning_write_plan_refuses_unsupported_path():
    namespace = _FakeNamespace(
        [
            SimpleNamespace(
                location="s3://robotics/features.lance",
                storage_options={},
                managed_versioning=True,
            )
        ]
    )
    access = PylanceNamespaceAccess(namespace, ("features",))

    with pytest.raises(ManagedVersioningMismatch, match="managed_versioning=true"):
        plan_namespace_write(access, supports_managed_versioning=False)


def test_namespace_worker_spec_is_secret_free_and_applies_worker_overrides():
    spec = namespace_worker_spec(
        namespace_client_impl="rest",
        namespace_client_properties={
            "uri": "https://phalanx.acme.internal",
            "header.x-lancedb-database": "acme",
            "header.Authorization": "Bearer hidden",
            "credential_vendor.aws_role_arn": "arn:aws:iam::123:role/secret",
            "_lancedb_worker_.uri": "https://phalanx-worker.acme.internal",
        },
        table_ids=[["features", "embeddings"]],
        namespace_auth_ref="phalanx-prod",
        storage_auth_ref="lake-storage",
        source_auth_ref="raw-source",
        logical_inputs={"job": "embed"},
    )

    assert spec["namespace"]["properties"] == {
        "uri": "https://phalanx-worker.acme.internal",
        "header.x-lancedb-database": "acme",
    }
    assert "hidden" not in str(spec)
    assert "arn:aws" not in str(spec)
    assert spec["auth_refs"] == {
        "namespace": "phalanx-prod",
        "storage": "lake-storage",
        "source": "raw-source",
    }
