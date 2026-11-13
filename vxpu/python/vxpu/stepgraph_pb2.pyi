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
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Answer(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANSWER_UNSPECIFIED: _ClassVar[Answer]
    ANSWER_CPU: _ClassVar[Answer]
    ANSWER_GPU_POOL: _ClassVar[Answer]
    ANSWER_PEER_GPU: _ClassVar[Answer]
    ANSWER_REMOTE: _ClassVar[Answer]

class Execution(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXECUTION_UNSPECIFIED: _ClassVar[Execution]
    EXECUTION_GRAPH: _ClassVar[Execution]
    EXECUTION_EXCHANGE: _ClassVar[Execution]

class Stream(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STREAM_UNSPECIFIED: _ClassVar[Stream]
    STREAM_MAIN: _ClassVar[Stream]
    STREAM_PARALLEL: _ClassVar[Stream]
ANSWER_UNSPECIFIED: Answer
ANSWER_CPU: Answer
ANSWER_GPU_POOL: Answer
ANSWER_PEER_GPU: Answer
ANSWER_REMOTE: Answer
EXECUTION_UNSPECIFIED: Execution
EXECUTION_GRAPH: Execution
EXECUTION_EXCHANGE: Execution
STREAM_UNSPECIFIED: Stream
STREAM_MAIN: Stream
STREAM_PARALLEL: Stream

class StepGraphDescription(_message.Message):
    __slots__ = ("nodes", "inputs", "outputs", "host_mappings")
    NODES_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    HOST_MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[StepNode]
    inputs: _containers.RepeatedScalarFieldContainer[str]
    outputs: _containers.RepeatedCompositeFieldContainer[ValueRef]
    host_mappings: _containers.RepeatedCompositeFieldContainer[HostMapping]
    def __init__(self, nodes: _Optional[_Iterable[_Union[StepNode, _Mapping]]] = ..., inputs: _Optional[_Iterable[str]] = ..., outputs: _Optional[_Iterable[_Union[ValueRef, _Mapping]]] = ..., host_mappings: _Optional[_Iterable[_Union[HostMapping, _Mapping]]] = ...) -> None: ...

class HostMapping(_message.Message):
    __slots__ = ("name", "file_backed")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_BACKED_FIELD_NUMBER: _ClassVar[int]
    name: str
    file_backed: bool
    def __init__(self, name: _Optional[str] = ..., file_backed: _Optional[bool] = ...) -> None: ...

class StepNode(_message.Message):
    __slots__ = ("name", "kind", "layers", "reads", "writes", "host", "execution", "answer")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    READS_FIELD_NUMBER: _ClassVar[int]
    WRITES_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: str
    layers: _containers.RepeatedScalarFieldContainer[int]
    reads: _containers.RepeatedCompositeFieldContainer[ValueRef]
    writes: _containers.RepeatedScalarFieldContainer[str]
    host: HostCall
    execution: Execution
    answer: Answer
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[str] = ..., layers: _Optional[_Iterable[int]] = ..., reads: _Optional[_Iterable[_Union[ValueRef, _Mapping]]] = ..., writes: _Optional[_Iterable[str]] = ..., host: _Optional[_Union[HostCall, _Mapping]] = ..., execution: _Optional[_Union[Execution, str]] = ..., answer: _Optional[_Union[Answer, str]] = ...) -> None: ...

class HostCall(_message.Message):
    __slots__ = ("collection", "index")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    collection: str
    index: int
    def __init__(self, collection: _Optional[str] = ..., index: _Optional[int] = ...) -> None: ...

class EdgeRef(_message.Message):
    __slots__ = ("node", "index")
    NODE_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    node: str
    index: int
    def __init__(self, node: _Optional[str] = ..., index: _Optional[int] = ...) -> None: ...

class ValueRef(_message.Message):
    __slots__ = ("input", "resident", "edge", "host_mapping")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    RESIDENT_FIELD_NUMBER: _ClassVar[int]
    EDGE_FIELD_NUMBER: _ClassVar[int]
    HOST_MAPPING_FIELD_NUMBER: _ClassVar[int]
    input: str
    resident: str
    edge: EdgeRef
    host_mapping: str
    def __init__(self, input: _Optional[str] = ..., resident: _Optional[str] = ..., edge: _Optional[_Union[EdgeRef, _Mapping]] = ..., host_mapping: _Optional[str] = ...) -> None: ...

class StepPlan(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodePlan]
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodePlan, _Mapping]]] = ...) -> None: ...

class NodeExecutionStats(_message.Message):
    __slots__ = ("device_time_ms", "sample_count")
    DEVICE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_COUNT_FIELD_NUMBER: _ClassVar[int]
    device_time_ms: float
    sample_count: int
    def __init__(self, device_time_ms: _Optional[float] = ..., sample_count: _Optional[int] = ...) -> None: ...

class NodePlan(_message.Message):
    __slots__ = ("name", "stream", "waits", "stats")
    NAME_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    WAITS_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    name: str
    stream: Stream
    waits: _containers.RepeatedScalarFieldContainer[str]
    stats: NodeExecutionStats
    def __init__(self, name: _Optional[str] = ..., stream: _Optional[_Union[Stream, str]] = ..., waits: _Optional[_Iterable[str]] = ..., stats: _Optional[_Union[NodeExecutionStats, _Mapping]] = ...) -> None: ...
