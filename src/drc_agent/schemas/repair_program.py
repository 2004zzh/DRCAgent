from __future__ import annotations

from typing import Annotated, Any, Literal, Union, get_args

from pydantic import Field, create_model, model_validator

from .common import Point, StrictModel


EdgeSelector = Literal["LEFT", "RIGHT", "BOTTOM", "TOP"]
ProgramPostcondition = Literal[
    "TARGET_RULE_PREDICATE_SATISFIED",
    "PRESERVE_CONNECTIVITY",
    "NO_NEW_DRC",
    "PRESERVE_ROUTE_ENDPOINTS",
]


class MoveObjectOp(StrictModel):
    op: Literal["MOVE_OBJECT"]
    target_object_id: str
    dx_dbu: int
    dy_dbu: int


class ResizeEdgeOp(StrictModel):
    op: Literal["RESIZE_EDGE"]
    target_object_id: str
    edge: EdgeSelector
    target_coordinate_dbu: int


class AddBoundedPolygonOp(StrictModel):
    op: Literal["ADD_BOUNDED_POLYGON"]
    source_owner_object_id: str
    layer: str
    polygon_dbu: list[Point] = Field(min_length=4, max_length=12)


class AddBoundedJogOp(StrictModel):
    op: Literal["ADD_BOUNDED_JOG"]
    target_fragment_id: str
    source_owner_object_id: str
    layer: str
    polygon_dbu: list[Point] = Field(min_length=4, max_length=12)


class LocalRouteReplacementOp(StrictModel):
    op: Literal["LOCAL_ROUTE_REPLACEMENT"]
    target_segment_id: str
    layer: str
    replacement_polygon_dbu: list[Point] = Field(min_length=4, max_length=16)


class MoveViaStackOp(StrictModel):
    op: Literal["MOVE_COMPLETE_VIA_STACK"]
    target_via_id: str
    dx_dbu: int
    dy_dbu: int


class InstanceLayerReplacement(StrictModel):
    instance_anchor_id: str
    parent_cell: str
    source_cell: str
    layer: str
    rotation: int = Field(ge=0, le=3)
    mirror: bool
    dx_dbu: int
    dy_dbu: int
    original_polygons_local_dbu: list[list[Point]] = Field(min_length=1)
    replacement_polygons_local_dbu: list[list[Point]] = Field(min_length=1)
    # True only for an exact-lineage copy-on-write edit of one local route
    # fragment.  The shared source definition remains untouched; every other
    # layer and polygon in the specialized occurrence is preserved verbatim.
    occurrence_local_fragment: bool = False


class SpecializeInstanceLayerOp(StrictModel):
    """Replace complete layer geometry for named target cell occurrences only."""

    op: Literal["SPECIALIZE_INSTANCE_LAYER"]
    source_owner_object_id: str
    instance_layer_replacements: list[InstanceLayerReplacement] = Field(
        min_length=1, max_length=4,
    )


class SpecializeInstanceBoundaryOp(StrictModel):
    """Atomically resize a direct source and target-only shared occurrences."""

    op: Literal["SPECIALIZE_INSTANCE_BOUNDARY"]
    source_owner_object_id: str
    edge: EdgeSelector
    target_coordinate_dbu: int
    instance_layer_replacements: list[InstanceLayerReplacement] = Field(
        min_length=1, max_length=4,
    )


RepairOperation = Annotated[
    Union[
        MoveObjectOp, ResizeEdgeOp, AddBoundedPolygonOp, AddBoundedJogOp,
        LocalRouteReplacementOp, MoveViaStackOp,
        SpecializeInstanceBoundaryOp, SpecializeInstanceLayerOp,
    ],
    Field(discriminator="op"),
]


class RepairProgram(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    program_id: str
    region_id: str
    target_violation_ids: list[str] = Field(min_length=1, max_length=4)
    witness_ids: list[str] = Field(min_length=1, max_length=4)
    operations: list[RepairOperation] = Field(min_length=1, max_length=4)
    semantic_postconditions: list[ProgramPostcondition] = Field(
        default_factory=lambda: [
            "TARGET_RULE_PREDICATE_SATISFIED",
            "PRESERVE_CONNECTIVITY",
            "NO_NEW_DRC",
        ],
        min_length=3,
        max_length=4,
    )
    strategy_code: str = Field(max_length=80)
    revision: int = Field(default=0, ge=0, le=2)
    prior_program_fingerprint: str | None = None

    @model_validator(mode="after")
    def atomic_and_bounded(self) -> "RepairProgram":
        targets = []
        for operation in self.operations:
            for field in (
                "target_object_id", "source_owner_object_id",
                "target_fragment_id", "target_segment_id", "target_via_id",
            ):
                value = getattr(operation, field, None)
                if value:
                    targets.append(value)
        if len(set(targets)) > 8:
            raise ValueError("repair program exceeds bounded local target set")
        return self


class RepairProgramBatch(StrictModel):
    programs: list[RepairProgram] = Field(min_length=1, max_length=2)


RepairAgentStepKind = Literal[
    "QUERY_WITNESS", "QUERY_GEOMETRY", "QUERY_CONNECTIVITY",
    "QUERY_LOCAL_OBSTACLES", "PROPOSE_PROGRAM", "STOP",
]


class RepairAgentStep(StrictModel):
    step: RepairAgentStepKind
    violation_ids: list[str] = Field(default_factory=list, max_length=4)
    object_ids: list[str] = Field(default_factory=list, max_length=8)
    program: RepairProgram | None = None
    stop_reason: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def valid_payload(self) -> "RepairAgentStep":
        if self.step == "PROPOSE_PROGRAM" and self.program is None:
            raise ValueError("PROPOSE_PROGRAM requires program")
        if self.step != "PROPOSE_PROGRAM" and self.program is not None:
            raise ValueError("query/stop steps cannot carry a program")
        if self.step == "QUERY_WITNESS" and not self.violation_ids:
            raise ValueError("QUERY_WITNESS requires violation_ids")
        if self.step == "QUERY_GEOMETRY" and not self.object_ids:
            raise ValueError("QUERY_GEOMETRY requires object_ids")
        return self


def constrained_repair_program_model(
    *,
    region_id: str,
    object_ids: set[str],
    segment_ids: set[str],
    fragment_ids: set[str],
    via_ids: set[str],
    layers: set[str],
    violation_ids: set[str],
    witness_ids: set[str],
    allowed_operations: set[str],
) -> type[RepairProgramBatch]:
    """Build an action/target-conditioned provider JSON schema."""
    if not object_ids or not violation_ids or not witness_ids:
        raise ValueError("repair program requires object, violation, and witness enums")
    object_type = Literal.__getitem__(tuple(sorted(object_ids)))
    violation_type = Literal.__getitem__(tuple(sorted(violation_ids)))
    witness_type = Literal.__getitem__(tuple(sorted(witness_ids)))
    layer_type = Literal.__getitem__(tuple(sorted(layers))) if layers else str
    operation_models = []

    if "MOVE_OBJECT" in allowed_operations:
        operation_models.append(create_model(
            "ConstrainedMoveObjectOp", __base__=MoveObjectOp,
            target_object_id=(object_type, ...),
        ))
    if "RESIZE_EDGE" in allowed_operations:
        operation_models.append(create_model(
            "ConstrainedResizeEdgeOp", __base__=ResizeEdgeOp,
            target_object_id=(object_type, ...),
        ))
    if "ADD_BOUNDED_POLYGON" in allowed_operations and layers:
        operation_models.append(create_model(
            "ConstrainedAddBoundedPolygonOp", __base__=AddBoundedPolygonOp,
            source_owner_object_id=(object_type, ...), layer=(layer_type, ...),
        ))
    if (
        "ADD_BOUNDED_JOG" in allowed_operations and fragment_ids and layers
    ):
        fragment_type = Literal.__getitem__(tuple(sorted(fragment_ids)))
        operation_models.append(create_model(
            "ConstrainedAddBoundedJogOp", __base__=AddBoundedJogOp,
            target_fragment_id=(fragment_type, ...),
            source_owner_object_id=(object_type, ...), layer=(layer_type, ...),
        ))
    if (
        "LOCAL_ROUTE_REPLACEMENT" in allowed_operations
        and segment_ids and layers
    ):
        segment_type = Literal.__getitem__(tuple(sorted(segment_ids)))
        operation_models.append(create_model(
            "ConstrainedLocalRouteReplacementOp",
            __base__=LocalRouteReplacementOp,
            target_segment_id=(segment_type, ...), layer=(layer_type, ...),
        ))
    if "MOVE_COMPLETE_VIA_STACK" in allowed_operations and via_ids:
        via_type = Literal.__getitem__(tuple(sorted(via_ids)))
        operation_models.append(create_model(
            "ConstrainedMoveViaStackOp", __base__=MoveViaStackOp,
            target_via_id=(via_type, ...),
        ))
    if not operation_models:
        raise ValueError("capabilities leave no legal RepairProgram operation")

    operation_union = (
        operation_models[0] if len(operation_models) == 1
        else Union.__getitem__(tuple(operation_models))
    )
    constrained_program = create_model(
        "ConstrainedRepairProgram", __base__=RepairProgram,
        region_id=(Literal.__getitem__((region_id,)), ...),
        target_violation_ids=(
            list[violation_type], Field(min_length=1, max_length=4),
        ),
        witness_ids=(list[witness_type], Field(min_length=1, max_length=4)),
        operations=(
            list[operation_union], Field(min_length=1, max_length=4),
        ),
    )
    return create_model(
        "ConstrainedRepairProgramBatch", __base__=RepairProgramBatch,
        programs=(list[constrained_program], Field(min_length=1, max_length=2)),
    )


def constrained_repair_agent_step_model(
    *, repair_program_batch_model: type[RepairProgramBatch],
    object_ids: set[str], violation_ids: set[str],
) -> type[StrictModel]:
    """Provider schema for one host-mediated, capability-conditioned step."""
    object_type = Literal.__getitem__(tuple(sorted(object_ids)))
    violation_type = Literal.__getitem__(tuple(sorted(violation_ids)))
    program_list = repair_program_batch_model.model_fields["programs"].annotation
    program_type = get_args(program_list)[0]
    query_witness = create_model(
        "ConstrainedQueryWitnessStep", __base__=StrictModel,
        step=(Literal["QUERY_WITNESS"], ...),
        violation_ids=(list[violation_type], Field(min_length=1, max_length=4)),
    )
    query_geometry = create_model(
        "ConstrainedQueryGeometryStep", __base__=StrictModel,
        step=(Literal["QUERY_GEOMETRY"], ...),
        object_ids=(list[object_type], Field(min_length=1, max_length=8)),
    )
    query_connectivity = create_model(
        "ConstrainedQueryConnectivityStep", __base__=StrictModel,
        step=(Literal["QUERY_CONNECTIVITY"], ...),
    )
    query_obstacles = create_model(
        "ConstrainedQueryLocalObstaclesStep", __base__=StrictModel,
        step=(Literal["QUERY_LOCAL_OBSTACLES"], ...),
    )
    propose = create_model(
        "ConstrainedProposeProgramStep", __base__=StrictModel,
        step=(Literal["PROPOSE_PROGRAM"], ...),
        program=(program_type, ...),
    )
    stop = create_model(
        "ConstrainedStopStep", __base__=StrictModel,
        step=(Literal["STOP"], ...),
        stop_reason=(str, Field(max_length=120)),
    )
    action_union = Annotated[
        Union.__getitem__((
            query_witness, query_geometry, query_connectivity,
            query_obstacles, propose, stop,
        )),
        Field(discriminator="step"),
    ]
    return create_model(
        "ConstrainedRepairAgentStep", __base__=StrictModel,
        action=(action_union, ...),
    )


class IntroducedViolation(StrictModel):
    violation_id: str
    rule_id: str | None = None
    bbox_dbu: dict | None = None


class EditedObjectImpact(StrictModel):
    object_id: str
    operation_types: list[str]
    before_bbox_dbu: dict | None = None
    after_bbox_dbu: dict | None = None


class PhysicalEffectFingerprint(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    sha256: str = Field(min_length=64, max_length=64)
    target_source_object_ids: list[str]
    canonical_geometry_before: list[dict[str, Any]]
    canonical_geometry_after: list[dict[str, Any]]
    normalized_operation_sequence: list[dict[str, Any]]
    target_violation_ids: list[str]


class MeasurementDelta(StrictModel):
    name: str
    before: int | bool | str | None = None
    after: int | bool | str | None = None
    required: int | bool | str | None = None
    comparator: str
    satisfied_after: bool | None = None


class RulePostconditionProbeResult(StrictModel):
    status: Literal[
        "PREDICATE_SATISFIED", "PREDICATE_STILL_VIOLATED", "UNAVAILABLE"
    ]
    violation_id: str
    rule_id: str
    before_rule_witness: dict[str, Any] | None = None
    after_rule_witness: dict[str, Any] | None = None
    changed_measurements: list[MeasurementDelta] = Field(default_factory=list)
    failure_code: str | None = None
    failure_message: str | None = None


class PreviewFeedback(StrictModel):
    status: Literal[
        "CLEAN", "NO_PROGRESS", "REGRESSION", "CONNECTIVITY_FAIL",
        "COMPILE_FAIL", "CHECK_FAIL", "PREDICATE_STILL_VIOLATED",
        "DUPLICATE_PHYSICAL_EFFECT", "EXECUTION_FAIL",
    ]
    target_violation_disappeared: bool
    removed_original_ids: list[str] = Field(default_factory=list)
    introduced_violations: list[IntroducedViolation] = Field(default_factory=list)
    connectivity_verdict: Literal["PRESERVED", "FAILED", "UNAVAILABLE"]
    edited_object_impact: list[EditedObjectImpact] = Field(default_factory=list)
    violated_semantic_postconditions: list[str] = Field(default_factory=list)
    candidate_fingerprint: str | None = None
    physical_effect_fingerprint: str | None = None
    before_rule_witness: dict[str, Any] | None = None
    after_rule_witness: dict[str, Any] | None = None
    changed_measurements: list[MeasurementDelta] = Field(default_factory=list)
    verification_artifact_path: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None


class RepairProgrammingAttempt(StrictModel):
    revision: int
    program: RepairProgram | None = None
    program_fingerprint: str | None = None
    physical_effect_fingerprint: str | None = None
    candidate_id: str | None = None
    feedback: PreviewFeedback


class RepairProgrammingTrace(StrictModel):
    region_id: str
    target_violation_ids: list[str]
    attempts: list[RepairProgrammingAttempt] = Field(default_factory=list)
    clean_candidate_id: str | None = None
    exhausted: bool = False
    llm_calls: int = 0
    query_steps: int = 0
    duplicate_physical_effect_count: int = 0
