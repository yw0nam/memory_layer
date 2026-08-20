"""Wiring checks for docker-compose.yml: state that must outlive a container."""

from __future__ import annotations

import pathlib
import re

import yaml

COMPOSE = yaml.safe_load(
    (pathlib.Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
)
API = COMPOSE["services"]["api"]
MCP = COMPOSE["services"]["mcp"]
STATEFUL_SERVICES = ("db", "api")
DATA_ROOT = re.compile(r"^\$\{DATA_ROOT:\?[^}]+\}/")


def _mounts(service: str) -> list[str]:
    return COMPOSE["services"][service]["volumes"]


def _state_mounts(service: str) -> list[str]:
    """Writable mounts only; a read-only mount carries config into the container, not state."""
    return [mount for mount in _mounts(service) if not mount.endswith(":ro")]


def _split(mount: str) -> tuple[str, str]:
    """Source and target of a short-syntax mount; the source may hold colons itself."""
    variable, brace, rest = mount.rpartition("}")
    path, _, target = (rest if brace else mount).partition(":")
    return variable + brace + path, target


def _mount_targets() -> set[str]:
    return {_split(mount)[1] for mount in API["volumes"]}


def test_api_persists_cocoindex_state_on_a_volume():
    state = API["environment"]["COCOINDEX_DB"]
    assert not state.startswith("${"), "state path must not be inherited from the host"
    assert state in _mount_targets()


def test_api_persists_repo_cache_on_a_volume():
    assert API["environment"]["REPO_CACHE"] in _mount_targets()


def test_api_persists_ingest_spool_on_a_volume():
    assert API["environment"]["INGEST_SPOOL"] == "/data/ingest-spool"
    assert API["environment"]["INGEST_SPOOL"] in _mount_targets()


def test_api_reads_git_credentials_read_only():
    """Private clones authenticate from this file; the container must not be able to write it."""
    assert "./git-credentials:/run/git-credentials:ro" in _mounts("api")


def test_every_stateful_mount_sits_under_the_data_root():
    for service in STATEFUL_SERVICES:
        for mount in _state_mounts(service):
            assert DATA_ROOT.match(_split(mount)[0]), f"{service}: {mount}"


def test_an_unset_data_root_fails_instead_of_mounting_the_host_root():
    """A bare ${DATA_ROOT} would expand to nothing and bind / into the container."""
    for service in STATEFUL_SERVICES:
        for mount in _state_mounts(service):
            source = _split(mount)[0]
            assert ":?" in source, f"{service}: {mount} must fail when DATA_ROOT is unset"


def test_no_state_hides_in_a_docker_managed_volume():
    """One host directory holds everything, so nothing survives outside it."""
    assert "volumes" not in COMPOSE


def test_every_stateful_service_persists_something():
    for service in STATEFUL_SERVICES:
        assert _mounts(service)


def test_api_receives_every_backend_endpoint_and_model_name():
    """Anything the container reads from the environment must be forwarded to it."""
    for name in ("LLM_URL", "EMB_URL", "RERANK_URL", "LLM_MODEL", "EMB_MODEL", "RERANK_MODEL"):
        assert API["environment"][name] == f"${{{name}}}"


def test_query_password_is_forwarded_to_api_and_mcp():
    expected = "${TABLES_QUERY_PASSWORD:?set TABLES_QUERY_PASSWORD in .env}"
    assert API["environment"]["TABLES_QUERY_PASSWORD"] == expected
    assert MCP["environment"]["TABLES_QUERY_PASSWORD"] == expected


def test_api_depends_only_on_postgres():
    assert API["depends_on"] == ["db"]


def test_api_healthcheck_probes_liveness_not_the_vllm_endpoints():
    """The container healthcheck must not go red because a backend model server is down."""
    probe = " ".join(API["healthcheck"]["test"][1:])
    assert "/health" in probe
    assert "/health/services" not in probe


def test_mcp_waits_for_a_healthy_api():
    assert COMPOSE["services"]["mcp"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_every_service_restarts_unless_stopped():
    """A host reboot must bring the whole stack back without manual intervention."""
    for name, service in COMPOSE["services"].items():
        assert service.get("restart") == "unless-stopped", name
