"""Wiring checks for docker-compose.yml: state that must outlive a container."""

from __future__ import annotations

import pathlib
import re

import yaml

COMPOSE = yaml.safe_load(
    (pathlib.Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
)
API = COMPOSE["services"]["api"]
STATEFUL_SERVICES = ("db", "redis", "api")
DATA_ROOT = re.compile(r"^\$\{DATA_ROOT:\?[^}]+\}/")


def _mounts(service: str) -> list[str]:
    return COMPOSE["services"][service]["volumes"]


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


def test_every_stateful_mount_sits_under_the_data_root():
    for service in STATEFUL_SERVICES:
        for mount in _mounts(service):
            assert DATA_ROOT.match(_split(mount)[0]), f"{service}: {mount}"


def test_an_unset_data_root_fails_instead_of_mounting_the_host_root():
    """A bare ${DATA_ROOT} would expand to nothing and bind / into the container."""
    for service in STATEFUL_SERVICES:
        for mount in _mounts(service):
            source = _split(mount)[0]
            assert ":?" in source, f"{service}: {mount} must fail when DATA_ROOT is unset"


def test_no_state_hides_in_a_docker_managed_volume():
    """One host directory holds everything, so nothing survives outside it."""
    assert "volumes" not in COMPOSE


def test_every_stateful_service_persists_something():
    for service in STATEFUL_SERVICES:
        assert _mounts(service)


def test_api_reaches_redis_by_service_name():
    url = API["environment"]["REDIS_URL"]
    assert "redis" in url, "must address the compose service, not a host endpoint"
    assert "localhost" not in url and "127.0.0.1" not in url


def test_redis_survives_its_own_restart():
    """Job state is what Redis holds; losing it on restart restores the 404 bug."""
    redis = COMPOSE["services"]["redis"]
    assert "appendonly yes" in " ".join(
        redis["command"] if isinstance(redis["command"], list) else [redis["command"]]
    )
    assert any(_split(mount)[1] == "/data" for mount in _mounts("redis"))


def test_api_starts_after_redis():
    assert "redis" in API["depends_on"]
