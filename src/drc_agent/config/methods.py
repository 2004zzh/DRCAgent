from __future__ import annotations

import re
from typing import Literal

from drc_agent.config.loader import AppConfig, FeaturesConfig
from drc_agent.schemas.common import StrictModel


class MethodPreset(StrictModel):
    name: str
    runtime: Literal["noop", "research"]
    llm_reasoning: bool
    dynamic_graph: bool
    graph_rewire: bool
    joint_planning: bool
    coordinator: Literal["none", "greedy", "cp_sat"]
    message_passing: bool
    experience_graph: bool
    geometry_edges: bool
    shared_net_edges: bool
    resource_edges: bool
    klayout_transaction: bool
    connectivity_gate: bool


_PRESETS = {
    "NO_OP": MethodPreset(
        name="NO_OP", runtime="noop", llm_reasoning=False,
        dynamic_graph=False, graph_rewire=False, joint_planning=False,
        coordinator="none", message_passing=False, experience_graph=False,
        geometry_edges=False, shared_net_edges=False, resource_edges=False,
        klayout_transaction=False, connectivity_gate=False,
    ),
    "E0": MethodPreset(
        name="E0", runtime="noop", llm_reasoning=False,
        dynamic_graph=False, graph_rewire=False, joint_planning=False,
        coordinator="none", message_passing=False, experience_graph=False,
        geometry_edges=False, shared_net_edges=False, resource_edges=False,
        klayout_transaction=False, connectivity_gate=False,
    ),
    "B2": MethodPreset(
        name="B2", runtime="research", llm_reasoning=True,
        dynamic_graph=False, graph_rewire=False, joint_planning=False,
        coordinator="greedy", message_passing=False, experience_graph=False,
        geometry_edges=False, shared_net_edges=False, resource_edges=False,
        klayout_transaction=True, connectivity_gate=True,
    ),
    "B3": MethodPreset(
        name="B3", runtime="research", llm_reasoning=True,
        dynamic_graph=False, graph_rewire=False, joint_planning=False,
        coordinator="greedy", message_passing=True, experience_graph=False,
        geometry_edges=True, shared_net_edges=False, resource_edges=False,
        klayout_transaction=True, connectivity_gate=True,
    ),
    "B4": MethodPreset(
        name="B4", runtime="research", llm_reasoning=True,
        dynamic_graph=True, graph_rewire=True, joint_planning=False,
        coordinator="greedy", message_passing=True, experience_graph=False,
        geometry_edges=True, shared_net_edges=True, resource_edges=True,
        klayout_transaction=True, connectivity_gate=True,
    ),
    "B5": MethodPreset(
        name="B5", runtime="research", llm_reasoning=True,
        dynamic_graph=True, graph_rewire=True, joint_planning=True,
        coordinator="cp_sat", message_passing=True, experience_graph=False,
        geometry_edges=True, shared_net_edges=True, resource_edges=True,
        klayout_transaction=True, connectivity_gate=True,
    ),
    "B6": MethodPreset(
        name="B6", runtime="research", llm_reasoning=True,
        dynamic_graph=True, graph_rewire=True, joint_planning=True,
        coordinator="cp_sat", message_passing=True, experience_graph=True,
        geometry_edges=True, shared_net_edges=True, resource_edges=True,
        klayout_transaction=True, connectivity_gate=True,
    ),
    "NO_EDGE": MethodPreset(
        name="NO_EDGE", runtime="research", llm_reasoning=True,
        dynamic_graph=True, graph_rewire=True, joint_planning=True,
        coordinator="cp_sat", message_passing=False, experience_graph=True,
        geometry_edges=False, shared_net_edges=False, resource_edges=False,
        klayout_transaction=True, connectivity_gate=True,
    ),
}


_METHOD_TOKEN = re.compile(
    r"(?<![A-Z0-9])B[0-9]+(?![A-Z0-9])"
)
_RUN_CASE_ABBREVIATION = re.compile(
    r"(?<=-)B[0-9]+(?=-S[0-9]+(?:-|$))"
)


def method_tokens(value: str) -> set[str]:
    """Return explicit B-method tokens without confusing Block1 for B1."""
    return set(_METHOD_TOKEN.findall(value.upper()))


def validate_method_identity(
    name: str, config: AppConfig, *, run_id: str | None = None,
    experiment_id: str | None = None, profile_name: str | None = None,
) -> None:
    """Fail closed when independently supplied run identity fields disagree."""
    normalized = name.upper()
    if normalized not in _PRESETS:
        raise ValueError(
            f"unsupported method: {name}; expected one of {sorted(_PRESETS)}"
        )
    expected_experience = _PRESETS[normalized].experience_graph
    if config.features.experience_graph is not expected_experience:
        raise ValueError(
            "METHOD_IDENTITY_MISMATCH: "
            f"method={normalized} requires experience_graph="
            f"{str(expected_experience).lower()}, resolved config has "
            f"{str(config.features.experience_graph).lower()}"
        )
    for field, value in (
        ("run_id", run_id),
        ("experiment_id", experiment_id),
        ("profile_name", profile_name),
    ):
        if not value:
            continue
        tokens = method_tokens(value)
        if field == "run_id":
            # In the short formal run convention, ``-b4-s1101`` denotes
            # case Block4, not method B4.  Remove that unambiguous case slot
            # only when another explicit method token remains; a legacy run
            # such as ``experiment-b4-s1101`` still encodes method B4.
            without_case = _RUN_CASE_ABBREVIATION.sub(
                "CASE", value.upper(), count=1,
            )
            remaining = method_tokens(without_case)
            if remaining:
                tokens = remaining
        if tokens and tokens != {normalized}:
            raise ValueError(
                "METHOD_IDENTITY_MISMATCH: "
                f"method={normalized} conflicts with {field}={value!r} "
                f"(encoded methods={sorted(tokens)})"
            )


def resolve_method(name: str, config: AppConfig, *, test_mode: bool = False) -> tuple[MethodPreset, AppConfig]:
    normalized = name.upper()
    if normalized not in _PRESETS:
        raise ValueError(f"unsupported method: {name}; expected one of {sorted(_PRESETS)}")
    preset = _PRESETS[normalized]
    if preset.llm_reasoning:
        if not config.llm.enabled:
            raise ValueError(f"{preset.name} requires llm.enabled=true; use NO_OP/E0 for smoke runs")
        if config.llm.provider == "fake" and not test_mode:
            raise ValueError("fake LLM is allowed only with explicit test_mode")
    if preset.connectivity_gate and not config.acceptance.require_connectivity:
        raise ValueError(f"{preset.name} requires acceptance.require_connectivity=true")
    features = FeaturesConfig(
        experience_graph=preset.experience_graph,
        dynamic_graph=preset.dynamic_graph,
        graph_rewire=preset.graph_rewire,
        joint_planning=preset.joint_planning,
        message_passing=preset.message_passing,
        geometry_edges=preset.geometry_edges,
        shared_net_edges=preset.shared_net_edges,
        resource_edges=preset.resource_edges,
        timing_edges=False,
        timing_verification=False,
    )
    resolved = config.model_copy(update={"features": features})
    validate_method_identity(preset.name, resolved)
    return preset, resolved
