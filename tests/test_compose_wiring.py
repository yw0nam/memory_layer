"""Wiring checks for docker-compose.yml: state that must outlive a container."""

from __future__ import annotations

import pathlib
import re

import yaml

COMPOSE = yaml.safe_load(
    (pathlib.Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
)
API = COMPOSE["services"]["api"]
STATEFUL_SERVICES = ("db",)
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
    for name in (
        "EMB_URL",
        "EMB_MODEL",
        "RERANK_URL",
        "RERANK_MODEL",
    ):
        assert API["environment"][name] == f"${{{name}}}"


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
