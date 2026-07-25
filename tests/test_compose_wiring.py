"""Wiring checks for docker-compose.yml: state that must outlive a container."""

from __future__ import annotations

import pathlib

import yaml

COMPOSE = yaml.safe_load(
    (pathlib.Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
)
API = COMPOSE["services"]["api"]


def _mount_targets() -> set[str]:
    return {mount.split(":")[1] for mount in API["volumes"]}


def test_api_persists_cocoindex_state_on_a_volume():
    state = API["environment"]["COCOINDEX_DB"]
    assert not state.startswith("${"), "state path must not be inherited from the host"
    assert state in _mount_targets()


def test_api_persists_repo_cache_on_a_volume():
    assert API["environment"]["REPO_CACHE"] in _mount_targets()


def test_declared_volumes_cover_every_named_api_mount():
    named = {mount.split(":")[0] for mount in API["volumes"]}
    assert named <= set(COMPOSE["volumes"])
