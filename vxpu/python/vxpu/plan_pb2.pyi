# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class KernelTable(_message.Message):
    __slots__ = ("gpu", "knobs", "sites", "knobs_by_rows")
    GPU_FIELD_NUMBER: _ClassVar[int]
    KNOBS_FIELD_NUMBER: _ClassVar[int]
    SITES_FIELD_NUMBER: _ClassVar[int]
    KNOBS_BY_ROWS_FIELD_NUMBER: _ClassVar[int]
    gpu: str
    knobs: Knobs
    sites: _containers.RepeatedCompositeFieldContainer[SiteEntry]
    knobs_by_rows: _containers.RepeatedCompositeFieldContainer[KnobsByRows]
    def __init__(self, gpu: _Optional[str] = ..., knobs: _Optional[_Union[Knobs, _Mapping]] = ..., sites: _Optional[_Iterable[_Union[SiteEntry, _Mapping]]] = ..., knobs_by_rows: _Optional[_Iterable[_Union[KnobsByRows, _Mapping]]] = ...) -> None: ...

class KnobsByRows(_message.Message):
    __slots__ = ("rows_max", "knobs")
    ROWS_MAX_FIELD_NUMBER: _ClassVar[int]
    KNOBS_FIELD_NUMBER: _ClassVar[int]
    rows_max: int
    knobs: Knobs
    def __init__(self, rows_max: _Optional[int] = ..., knobs: _Optional[_Union[Knobs, _Mapping]] = ...) -> None: ...

class SiteEntry(_message.Message):
    __slots__ = ("op", "site", "shapes", "impl")
    OP_FIELD_NUMBER: _ClassVar[int]
    SITE_FIELD_NUMBER: _ClassVar[int]
    SHAPES_FIELD_NUMBER: _ClassVar[int]
    IMPL_FIELD_NUMBER: _ClassVar[int]
    op: str
    site: str
    shapes: str
    impl: str
    def __init__(self, op: _Optional[str] = ..., site: _Optional[str] = ..., shapes: _Optional[str] = ..., impl: _Optional[str] = ...) -> None: ...

class Knobs(_message.Message):
    __slots__ = ("GEMV_ROWS_MAX", "FP4_DIRECT_MAX", "FFN_FRONT_EB", "FFN_FRONT_BD", "FRONT_SPLIT_BD", "FRONT_SPLIT_EB", "FRONT_SPLIT_RB", "FP4_KSPLIT", "FP4_NATIVE", "FP4_DIRECT", "MXFP8_RECODE", "FP4_GU_BN", "FP4_GU_BK", "FP4_GU_WARPS", "FP4_GU_STAGES", "FP4_DN_BN", "FP4_DN_BK", "FP4_DN_WARPS", "FP4_DN_STAGES", "FOLD_POST", "FOLD_POST_RB", "FP4_ONE_LAUNCH", "FP4_ONE_PER_SM", "FP4_ONE_WARPS", "FP4_ONE_STAGES", "FP4_PDL", "PDL_ALL", "FP4_DIRECT_COMPACT", "FP4_DIRECT_GROUP", "FP4_DIRECT_FIRST", "SHARED_FFN_GEMM_MIN_ROWS", "FP4_GU_KS", "FP4_DN_KS", "FP4_GEMM_MIN_PAIRS", "FP4_GG_BM", "FP4_GG_BN", "FP4_GG_BK", "FP4_GG_WARPS", "FP4_GG_STAGES", "FP4_GD_BM", "FP4_GD_BN", "FP4_GD_BK", "FP4_GD_WARPS", "FP4_GD_STAGES", "FP4_FUSE_EPI", "ALLREDUCE_SCATTER", "ALLREDUCE_SCATTER_MIN_ROWS", "ALLREDUCE_PUSH_BF16", "ALLREDUCE_PULL", "DRAFT_ALLREDUCE_PULL", "ALLREDUCE_BLOCK", "TP_PARTIALS_BF16", "GEMV_PUSH_BN")
    GEMV_ROWS_MAX_FIELD_NUMBER: _ClassVar[int]
    FP4_DIRECT_MAX_FIELD_NUMBER: _ClassVar[int]
    FFN_FRONT_EB_FIELD_NUMBER: _ClassVar[int]
    FFN_FRONT_BD_FIELD_NUMBER: _ClassVar[int]
    FRONT_SPLIT_BD_FIELD_NUMBER: _ClassVar[int]
    FRONT_SPLIT_EB_FIELD_NUMBER: _ClassVar[int]
    FRONT_SPLIT_RB_FIELD_NUMBER: _ClassVar[int]
    FP4_KSPLIT_FIELD_NUMBER: _ClassVar[int]
    FP4_NATIVE_FIELD_NUMBER: _ClassVar[int]
    FP4_DIRECT_FIELD_NUMBER: _ClassVar[int]
    MXFP8_RECODE_FIELD_NUMBER: _ClassVar[int]
    FP4_GU_BN_FIELD_NUMBER: _ClassVar[int]
    FP4_GU_BK_FIELD_NUMBER: _ClassVar[int]
    FP4_GU_WARPS_FIELD_NUMBER: _ClassVar[int]
    FP4_GU_STAGES_FIELD_NUMBER: _ClassVar[int]
    FP4_DN_BN_FIELD_NUMBER: _ClassVar[int]
    FP4_DN_BK_FIELD_NUMBER: _ClassVar[int]
    FP4_DN_WARPS_FIELD_NUMBER: _ClassVar[int]
    FP4_DN_STAGES_FIELD_NUMBER: _ClassVar[int]
    FOLD_POST_FIELD_NUMBER: _ClassVar[int]
    FOLD_POST_RB_FIELD_NUMBER: _ClassVar[int]
    FP4_ONE_LAUNCH_FIELD_NUMBER: _ClassVar[int]
    FP4_ONE_PER_SM_FIELD_NUMBER: _ClassVar[int]
    FP4_ONE_WARPS_FIELD_NUMBER: _ClassVar[int]
    FP4_ONE_STAGES_FIELD_NUMBER: _ClassVar[int]
    FP4_PDL_FIELD_NUMBER: _ClassVar[int]
    PDL_ALL_FIELD_NUMBER: _ClassVar[int]
    FP4_DIRECT_COMPACT_FIELD_NUMBER: _ClassVar[int]
    FP4_DIRECT_GROUP_FIELD_NUMBER: _ClassVar[int]
    FP4_DIRECT_FIRST_FIELD_NUMBER: _ClassVar[int]
    SHARED_FFN_GEMM_MIN_ROWS_FIELD_NUMBER: _ClassVar[int]
    FP4_GU_KS_FIELD_NUMBER: _ClassVar[int]
    FP4_DN_KS_FIELD_NUMBER: _ClassVar[int]
    FP4_GEMM_MIN_PAIRS_FIELD_NUMBER: _ClassVar[int]
    FP4_GG_BM_FIELD_NUMBER: _ClassVar[int]
    FP4_GG_BN_FIELD_NUMBER: _ClassVar[int]
    FP4_GG_BK_FIELD_NUMBER: _ClassVar[int]
    FP4_GG_WARPS_FIELD_NUMBER: _ClassVar[int]
    FP4_GG_STAGES_FIELD_NUMBER: _ClassVar[int]
    FP4_GD_BM_FIELD_NUMBER: _ClassVar[int]
    FP4_GD_BN_FIELD_NUMBER: _ClassVar[int]
    FP4_GD_BK_FIELD_NUMBER: _ClassVar[int]
    FP4_GD_WARPS_FIELD_NUMBER: _ClassVar[int]
    FP4_GD_STAGES_FIELD_NUMBER: _ClassVar[int]
    FP4_FUSE_EPI_FIELD_NUMBER: _ClassVar[int]
    ALLREDUCE_SCATTER_FIELD_NUMBER: _ClassVar[int]
    ALLREDUCE_SCATTER_MIN_ROWS_FIELD_NUMBER: _ClassVar[int]
    ALLREDUCE_PUSH_BF16_FIELD_NUMBER: _ClassVar[int]
    ALLREDUCE_PULL_FIELD_NUMBER: _ClassVar[int]
    DRAFT_ALLREDUCE_PULL_FIELD_NUMBER: _ClassVar[int]
    ALLREDUCE_BLOCK_FIELD_NUMBER: _ClassVar[int]
    TP_PARTIALS_BF16_FIELD_NUMBER: _ClassVar[int]
    GEMV_PUSH_BN_FIELD_NUMBER: _ClassVar[int]
    GEMV_ROWS_MAX: int
    FP4_DIRECT_MAX: int
    FFN_FRONT_EB: int
    FFN_FRONT_BD: int
    FRONT_SPLIT_BD: int
    FRONT_SPLIT_EB: int
    FRONT_SPLIT_RB: int
    FP4_KSPLIT: int
    FP4_NATIVE: bool
    FP4_DIRECT: bool
    MXFP8_RECODE: bool
    FP4_GU_BN: int
    FP4_GU_BK: int
    FP4_GU_WARPS: int
    FP4_GU_STAGES: int
    FP4_DN_BN: int
    FP4_DN_BK: int
    FP4_DN_WARPS: int
    FP4_DN_STAGES: int
    FOLD_POST: bool
    FOLD_POST_RB: int
    FP4_ONE_LAUNCH: bool
    FP4_ONE_PER_SM: int
    FP4_ONE_WARPS: int
    FP4_ONE_STAGES: int
    FP4_PDL: int
    PDL_ALL: bool
    FP4_DIRECT_COMPACT: bool
    FP4_DIRECT_GROUP: bool
    FP4_DIRECT_FIRST: bool
    SHARED_FFN_GEMM_MIN_ROWS: int
    FP4_GU_KS: int
    FP4_DN_KS: int
    FP4_GEMM_MIN_PAIRS: int
    FP4_GG_BM: int
    FP4_GG_BN: int
    FP4_GG_BK: int
    FP4_GG_WARPS: int
    FP4_GG_STAGES: int
    FP4_GD_BM: int
    FP4_GD_BN: int
    FP4_GD_BK: int
    FP4_GD_WARPS: int
    FP4_GD_STAGES: int
    FP4_FUSE_EPI: int
    ALLREDUCE_SCATTER: bool
    ALLREDUCE_SCATTER_MIN_ROWS: int
    ALLREDUCE_PUSH_BF16: bool
    ALLREDUCE_PULL: bool
    DRAFT_ALLREDUCE_PULL: bool
    ALLREDUCE_BLOCK: int
    TP_PARTIALS_BF16: bool
    GEMV_PUSH_BN: int
    def __init__(self, GEMV_ROWS_MAX: _Optional[int] = ..., FP4_DIRECT_MAX: _Optional[int] = ..., FFN_FRONT_EB: _Optional[int] = ..., FFN_FRONT_BD: _Optional[int] = ..., FRONT_SPLIT_BD: _Optional[int] = ..., FRONT_SPLIT_EB: _Optional[int] = ..., FRONT_SPLIT_RB: _Optional[int] = ..., FP4_KSPLIT: _Optional[int] = ..., FP4_NATIVE: _Optional[bool] = ..., FP4_DIRECT: _Optional[bool] = ..., MXFP8_RECODE: _Optional[bool] = ..., FP4_GU_BN: _Optional[int] = ..., FP4_GU_BK: _Optional[int] = ..., FP4_GU_WARPS: _Optional[int] = ..., FP4_GU_STAGES: _Optional[int] = ..., FP4_DN_BN: _Optional[int] = ..., FP4_DN_BK: _Optional[int] = ..., FP4_DN_WARPS: _Optional[int] = ..., FP4_DN_STAGES: _Optional[int] = ..., FOLD_POST: _Optional[bool] = ..., FOLD_POST_RB: _Optional[int] = ..., FP4_ONE_LAUNCH: _Optional[bool] = ..., FP4_ONE_PER_SM: _Optional[int] = ..., FP4_ONE_WARPS: _Optional[int] = ..., FP4_ONE_STAGES: _Optional[int] = ..., FP4_PDL: _Optional[int] = ..., PDL_ALL: _Optional[bool] = ..., FP4_DIRECT_COMPACT: _Optional[bool] = ..., FP4_DIRECT_GROUP: _Optional[bool] = ..., FP4_DIRECT_FIRST: _Optional[bool] = ..., SHARED_FFN_GEMM_MIN_ROWS: _Optional[int] = ..., FP4_GU_KS: _Optional[int] = ..., FP4_DN_KS: _Optional[int] = ..., FP4_GEMM_MIN_PAIRS: _Optional[int] = ..., FP4_GG_BM: _Optional[int] = ..., FP4_GG_BN: _Optional[int] = ..., FP4_GG_BK: _Optional[int] = ..., FP4_GG_WARPS: _Optional[int] = ..., FP4_GG_STAGES: _Optional[int] = ..., FP4_GD_BM: _Optional[int] = ..., FP4_GD_BN: _Optional[int] = ..., FP4_GD_BK: _Optional[int] = ..., FP4_GD_WARPS: _Optional[int] = ..., FP4_GD_STAGES: _Optional[int] = ..., FP4_FUSE_EPI: _Optional[int] = ..., ALLREDUCE_SCATTER: _Optional[bool] = ..., ALLREDUCE_SCATTER_MIN_ROWS: _Optional[int] = ..., ALLREDUCE_PUSH_BF16: _Optional[bool] = ..., ALLREDUCE_PULL: _Optional[bool] = ..., DRAFT_ALLREDUCE_PULL: _Optional[bool] = ..., ALLREDUCE_BLOCK: _Optional[int] = ..., TP_PARTIALS_BF16: _Optional[bool] = ..., GEMV_PUSH_BN: _Optional[int] = ...) -> None: ...

class PlanSite(_message.Message):
    __slots__ = ("id", "kind", "shape", "row_counts", "strategy")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNTS_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    row_counts: _containers.RepeatedScalarFieldContainer[int]
    strategy: str
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., row_counts: _Optional[_Iterable[int]] = ..., strategy: _Optional[str] = ...) -> None: ...

class PlanFile(_message.Message):
    __slots__ = ("artifact_md5", "tree_commit", "sites")
    ARTIFACT_MD5_FIELD_NUMBER: _ClassVar[int]
    TREE_COMMIT_FIELD_NUMBER: _ClassVar[int]
    SITES_FIELD_NUMBER: _ClassVar[int]
    artifact_md5: str
    tree_commit: str
    sites: _containers.RepeatedCompositeFieldContainer[PlanSite]
    def __init__(self, artifact_md5: _Optional[str] = ..., tree_commit: _Optional[str] = ..., sites: _Optional[_Iterable[_Union[PlanSite, _Mapping]]] = ...) -> None: ...

class RunRecord(_message.Message):
    __slots__ = ("plan_hash", "tree_commit", "artifact_md5", "knobs", "median_step_ms", "tokens_a_stream_a_step", "accepted_drafts_a_step", "checks", "per_card_peak_memory", "build_time_s", "per_class_device_times")
    class ChecksEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bool
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bool] = ...) -> None: ...
    class PerClassDeviceTimesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    PLAN_HASH_FIELD_NUMBER: _ClassVar[int]
    TREE_COMMIT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_MD5_FIELD_NUMBER: _ClassVar[int]
    KNOBS_FIELD_NUMBER: _ClassVar[int]
    MEDIAN_STEP_MS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_A_STREAM_A_STEP_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_DRAFTS_A_STEP_FIELD_NUMBER: _ClassVar[int]
    CHECKS_FIELD_NUMBER: _ClassVar[int]
    PER_CARD_PEAK_MEMORY_FIELD_NUMBER: _ClassVar[int]
    BUILD_TIME_S_FIELD_NUMBER: _ClassVar[int]
    PER_CLASS_DEVICE_TIMES_FIELD_NUMBER: _ClassVar[int]
    plan_hash: str
    tree_commit: str
    artifact_md5: str
    knobs: Knobs
    median_step_ms: float
    tokens_a_stream_a_step: float
    accepted_drafts_a_step: float
    checks: _containers.ScalarMap[str, bool]
    per_card_peak_memory: _containers.RepeatedScalarFieldContainer[int]
    build_time_s: float
    per_class_device_times: _containers.ScalarMap[str, float]
    def __init__(self, plan_hash: _Optional[str] = ..., tree_commit: _Optional[str] = ..., artifact_md5: _Optional[str] = ..., knobs: _Optional[_Union[Knobs, _Mapping]] = ..., median_step_ms: _Optional[float] = ..., tokens_a_stream_a_step: _Optional[float] = ..., accepted_drafts_a_step: _Optional[float] = ..., checks: _Optional[_Mapping[str, bool]] = ..., per_card_peak_memory: _Optional[_Iterable[int]] = ..., build_time_s: _Optional[float] = ..., per_class_device_times: _Optional[_Mapping[str, float]] = ...) -> None: ...
