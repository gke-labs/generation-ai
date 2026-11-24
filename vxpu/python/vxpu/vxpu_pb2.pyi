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

from . import plan_pb2 as _plan_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Telemetry(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TELEMETRY_NONE: _ClassVar[Telemetry]
    TELEMETRY_SUMMARY: _ClassVar[Telemetry]
    TELEMETRY_TRACE: _ClassVar[Telemetry]

class LoadState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOAD_STATE_NONE: _ClassVar[LoadState]
    LOAD_STATE_LOADING: _ClassVar[LoadState]
    LOAD_STATE_READY: _ClassVar[LoadState]
    LOAD_STATE_FAILED: _ClassVar[LoadState]
TELEMETRY_NONE: Telemetry
TELEMETRY_SUMMARY: Telemetry
TELEMETRY_TRACE: Telemetry
LOAD_STATE_NONE: LoadState
LOAD_STATE_LOADING: LoadState
LOAD_STATE_READY: LoadState
LOAD_STATE_FAILED: LoadState

class LoadModelRequest(_message.Message):
    __slots__ = ("manifest_json", "binding_json", "prefill_graph", "decode_graph", "plan", "plan_file")
    class PlanEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MANIFEST_JSON_FIELD_NUMBER: _ClassVar[int]
    BINDING_JSON_FIELD_NUMBER: _ClassVar[int]
    PREFILL_GRAPH_FIELD_NUMBER: _ClassVar[int]
    DECODE_GRAPH_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    PLAN_FILE_FIELD_NUMBER: _ClassVar[int]
    manifest_json: str
    binding_json: str
    prefill_graph: bytes
    decode_graph: bytes
    plan: _containers.ScalarMap[str, str]
    plan_file: _plan_pb2.PlanFile
    def __init__(self, manifest_json: _Optional[str] = ..., binding_json: _Optional[str] = ..., prefill_graph: _Optional[bytes] = ..., decode_graph: _Optional[bytes] = ..., plan: _Optional[_Mapping[str, str]] = ..., plan_file: _Optional[_Union[_plan_pb2.PlanFile, _Mapping]] = ...) -> None: ...

class LoadModelResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: _Optional[bool] = ...) -> None: ...

class RegisterModelRequest(_message.Message):
    __slots__ = ("model", "revision", "plan", "golden", "plan_file")
    class PlanEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    GOLDEN_FIELD_NUMBER: _ClassVar[int]
    PLAN_FILE_FIELD_NUMBER: _ClassVar[int]
    model: str
    revision: str
    plan: _containers.ScalarMap[str, str]
    golden: _plan_pb2.GoldenFile
    plan_file: _plan_pb2.PlanFile
    def __init__(self, model: _Optional[str] = ..., revision: _Optional[str] = ..., plan: _Optional[_Mapping[str, str]] = ..., golden: _Optional[_Union[_plan_pb2.GoldenFile, _Mapping]] = ..., plan_file: _Optional[_Union[_plan_pb2.PlanFile, _Mapping]] = ...) -> None: ...

class RegisterModelResponse(_message.Message):
    __slots__ = ("model", "revision", "commit", "artifact", "compiled", "weight_bytes")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    COMMIT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    COMPILED_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_BYTES_FIELD_NUMBER: _ClassVar[int]
    model: str
    revision: str
    commit: str
    artifact: str
    compiled: bool
    weight_bytes: int
    def __init__(self, model: _Optional[str] = ..., revision: _Optional[str] = ..., commit: _Optional[str] = ..., artifact: _Optional[str] = ..., compiled: _Optional[bool] = ..., weight_bytes: _Optional[int] = ...) -> None: ...

class NewSessionRequest(_message.Message):
    __slots__ = ("model", "revision")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    model: str
    revision: str
    def __init__(self, model: _Optional[str] = ..., revision: _Optional[str] = ...) -> None: ...

class NewSessionResponse(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class ChatRequest(_message.Message):
    __slots__ = ("session_id", "text", "max_new_tokens", "raw_prompt")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    RAW_PROMPT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    text: str
    max_new_tokens: int
    raw_prompt: bool
    def __init__(self, session_id: _Optional[str] = ..., text: _Optional[str] = ..., max_new_tokens: _Optional[int] = ..., raw_prompt: _Optional[bool] = ...) -> None: ...

class ChatResponse(_message.Message):
    __slots__ = ("text", "session_tokens", "new_prompt_tokens", "generated", "prefill_ms", "ms_per_token")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NEW_PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    GENERATED_FIELD_NUMBER: _ClassVar[int]
    PREFILL_MS_FIELD_NUMBER: _ClassVar[int]
    MS_PER_TOKEN_FIELD_NUMBER: _ClassVar[int]
    text: str
    session_tokens: int
    new_prompt_tokens: int
    generated: int
    prefill_ms: float
    ms_per_token: float
    def __init__(self, text: _Optional[str] = ..., session_tokens: _Optional[int] = ..., new_prompt_tokens: _Optional[int] = ..., generated: _Optional[int] = ..., prefill_ms: _Optional[float] = ..., ms_per_token: _Optional[float] = ...) -> None: ...

class ExecuteChatRequest(_message.Message):
    __slots__ = ("request", "telemetry")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_FIELD_NUMBER: _ClassVar[int]
    request: ChatRequest
    telemetry: Telemetry
    def __init__(self, request: _Optional[_Union[ChatRequest, _Mapping]] = ..., telemetry: _Optional[_Union[Telemetry, str]] = ...) -> None: ...

class ExecuteChatResponse(_message.Message):
    __slots__ = ("response", "telemetry_json")
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_JSON_FIELD_NUMBER: _ClassVar[int]
    response: ChatResponse
    telemetry_json: str
    def __init__(self, response: _Optional[_Union[ChatResponse, _Mapping]] = ..., telemetry_json: _Optional[str] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("session_id", "input_ids", "max_new_tokens", "do_sample", "temperature", "top_k", "top_p", "repetition_penalty", "eos_token_id", "seed", "ignore_eos")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    DO_SAMPLE_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    IGNORE_EOS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    input_ids: _containers.RepeatedScalarFieldContainer[int]
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    eos_token_id: _containers.RepeatedScalarFieldContainer[int]
    seed: int
    ignore_eos: bool
    def __init__(self, session_id: _Optional[str] = ..., input_ids: _Optional[_Iterable[int]] = ..., max_new_tokens: _Optional[int] = ..., do_sample: _Optional[bool] = ..., temperature: _Optional[float] = ..., top_k: _Optional[int] = ..., top_p: _Optional[float] = ..., repetition_penalty: _Optional[float] = ..., eos_token_id: _Optional[_Iterable[int]] = ..., seed: _Optional[int] = ..., ignore_eos: _Optional[bool] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("token_ids", "done", "finish_reason", "prompt_tokens", "prefilled_tokens", "generated", "prefill_ms", "ms_per_token")
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PREFILLED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    GENERATED_FIELD_NUMBER: _ClassVar[int]
    PREFILL_MS_FIELD_NUMBER: _ClassVar[int]
    MS_PER_TOKEN_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    done: bool
    finish_reason: str
    prompt_tokens: int
    prefilled_tokens: int
    generated: int
    prefill_ms: float
    ms_per_token: float
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., done: _Optional[bool] = ..., finish_reason: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., prefilled_tokens: _Optional[int] = ..., generated: _Optional[int] = ..., prefill_ms: _Optional[float] = ..., ms_per_token: _Optional[float] = ...) -> None: ...

class ExecuteRequest(_message.Message):
    __slots__ = ("request", "telemetry")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_FIELD_NUMBER: _ClassVar[int]
    request: GenerateRequest
    telemetry: Telemetry
    def __init__(self, request: _Optional[_Union[GenerateRequest, _Mapping]] = ..., telemetry: _Optional[_Union[Telemetry, str]] = ...) -> None: ...

class ExecuteResponse(_message.Message):
    __slots__ = ("response", "telemetry_json")
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_JSON_FIELD_NUMBER: _ClassVar[int]
    response: GenerateResponse
    telemetry_json: str
    def __init__(self, response: _Optional[_Union[GenerateResponse, _Mapping]] = ..., telemetry_json: _Optional[str] = ...) -> None: ...

class GetModelStatsRequest(_message.Message):
    __slots__ = ("model", "records")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    model: str
    records: bool
    def __init__(self, model: _Optional[str] = ..., records: _Optional[bool] = ...) -> None: ...

class GetModelStatsResponse(_message.Message):
    __slots__ = ("summary_json", "records_jsonl")
    SUMMARY_JSON_FIELD_NUMBER: _ClassVar[int]
    RECORDS_JSONL_FIELD_NUMBER: _ClassVar[int]
    summary_json: str
    records_jsonl: str
    def __init__(self, summary_json: _Optional[str] = ..., records_jsonl: _Optional[str] = ...) -> None: ...

class DescribeRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DescribeResponse(_message.Message):
    __slots__ = ("hardware_json",)
    HARDWARE_JSON_FIELD_NUMBER: _ClassVar[int]
    hardware_json: str
    def __init__(self, hardware_json: _Optional[str] = ...) -> None: ...

class ExplainModelRequest(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: str
    def __init__(self, model: _Optional[str] = ...) -> None: ...

class ExplainModelResponse(_message.Message):
    __slots__ = ("plan_json",)
    PLAN_JSON_FIELD_NUMBER: _ClassVar[int]
    plan_json: str
    def __init__(self, plan_json: _Optional[str] = ...) -> None: ...

class DescribeNodesRequest(_message.Message):
    __slots__ = ("manifest_json", "binding_json", "decode_graph", "prefill_graph")
    MANIFEST_JSON_FIELD_NUMBER: _ClassVar[int]
    BINDING_JSON_FIELD_NUMBER: _ClassVar[int]
    DECODE_GRAPH_FIELD_NUMBER: _ClassVar[int]
    PREFILL_GRAPH_FIELD_NUMBER: _ClassVar[int]
    manifest_json: str
    binding_json: str
    decode_graph: bytes
    prefill_graph: bytes
    def __init__(self, manifest_json: _Optional[str] = ..., binding_json: _Optional[str] = ..., decode_graph: _Optional[bytes] = ..., prefill_graph: _Optional[bytes] = ...) -> None: ...

class DescribeNodesResponse(_message.Message):
    __slots__ = ("nodes_json", "cuts_json")
    NODES_JSON_FIELD_NUMBER: _ClassVar[int]
    CUTS_JSON_FIELD_NUMBER: _ClassVar[int]
    nodes_json: str
    cuts_json: str
    def __init__(self, nodes_json: _Optional[str] = ..., cuts_json: _Optional[str] = ...) -> None: ...

class ServeNodesRequest(_message.Message):
    __slots__ = ("manifest_json", "binding_json", "decode_graph", "nodes")
    MANIFEST_JSON_FIELD_NUMBER: _ClassVar[int]
    BINDING_JSON_FIELD_NUMBER: _ClassVar[int]
    DECODE_GRAPH_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    manifest_json: str
    binding_json: str
    decode_graph: bytes
    nodes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, manifest_json: _Optional[str] = ..., binding_json: _Optional[str] = ..., decode_graph: _Optional[bytes] = ..., nodes: _Optional[_Iterable[str]] = ...) -> None: ...

class ServeNodesResponse(_message.Message):
    __slots__ = ("port",)
    PORT_FIELD_NUMBER: _ClassVar[int]
    port: int
    def __init__(self, port: _Optional[int] = ...) -> None: ...

class ServeRangeRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServeRangeResponse(_message.Message):
    __slots__ = ("port", "layer_range", "state")
    PORT_FIELD_NUMBER: _ClassVar[int]
    LAYER_RANGE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    port: int
    layer_range: str
    state: LoadState
    def __init__(self, port: _Optional[int] = ..., layer_range: _Optional[str] = ..., state: _Optional[_Union[LoadState, str]] = ...) -> None: ...

class GetLoadRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetLoadResponse(_message.Message):
    __slots__ = ("state", "error", "bytes_rehydrated")
    STATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    BYTES_REHYDRATED_FIELD_NUMBER: _ClassVar[int]
    state: LoadState
    error: str
    bytes_rehydrated: int
    def __init__(self, state: _Optional[_Union[LoadState, str]] = ..., error: _Optional[str] = ..., bytes_rehydrated: _Optional[int] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("role", "content")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class TokenizeChatRequest(_message.Message):
    __slots__ = ("model", "messages", "add_generation_prompt", "chat_template_kwargs_json")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    ADD_GENERATION_PROMPT_FIELD_NUMBER: _ClassVar[int]
    CHAT_TEMPLATE_KWARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    model: str
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    add_generation_prompt: bool
    chat_template_kwargs_json: str
    def __init__(self, model: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., add_generation_prompt: _Optional[bool] = ..., chat_template_kwargs_json: _Optional[str] = ...) -> None: ...

class TokenizeChatResponse(_message.Message):
    __slots__ = ("input_ids",)
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    input_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, input_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class GetVocabRequest(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: str
    def __init__(self, model: _Optional[str] = ...) -> None: ...

class GetVocabResponse(_message.Message):
    __slots__ = ("pieces", "special_token_ids")
    PIECES_FIELD_NUMBER: _ClassVar[int]
    SPECIAL_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    pieces: _containers.RepeatedScalarFieldContainer[bytes]
    special_token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, pieces: _Optional[_Iterable[bytes]] = ..., special_token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class CompileRequest(_message.Message):
    __slots__ = ("artifact", "plan", "machine_kind", "software_version", "device", "packages", "parallel_builds", "model")
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    MACHINE_KIND_FIELD_NUMBER: _ClassVar[int]
    SOFTWARE_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    PACKAGES_FIELD_NUMBER: _ClassVar[int]
    PARALLEL_BUILDS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    artifact: str
    plan: _plan_pb2.PlanFile
    machine_kind: str
    software_version: str
    device: str
    packages: _containers.RepeatedCompositeFieldContainer[_plan_pb2.Package]
    parallel_builds: int
    model: str
    def __init__(self, artifact: _Optional[str] = ..., plan: _Optional[_Union[_plan_pb2.PlanFile, _Mapping]] = ..., machine_kind: _Optional[str] = ..., software_version: _Optional[str] = ..., device: _Optional[str] = ..., packages: _Optional[_Iterable[_Union[_plan_pb2.Package, _Mapping]]] = ..., parallel_builds: _Optional[int] = ..., model: _Optional[str] = ...) -> None: ...

class CompileResponse(_message.Message):
    __slots__ = ("report",)
    REPORT_FIELD_NUMBER: _ClassVar[int]
    report: _plan_pb2.PackageReport
    def __init__(self, report: _Optional[_Union[_plan_pb2.PackageReport, _Mapping]] = ...) -> None: ...
