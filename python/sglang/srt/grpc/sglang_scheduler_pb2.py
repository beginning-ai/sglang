"""Compatibility wrapper for scheduler protobufs.

Standard scheduler messages come from `smg_grpc_proto.generated.sglang_scheduler_pb2`.
We keep only the Qwen3-Omni streaming-audio messages local.
"""

from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
from smg_grpc_proto.generated.sglang_scheduler_pb2 import *  # noqa: F401,F403
from smg_grpc_proto.generated import sglang_scheduler_pb2 as _base_pb2

_sym_db = _symbol_database.Default()


def _make_streaming_audio_descriptor():
    file_proto = _descriptor_pb2.FileDescriptorProto()
    file_proto.name = "sglang_streaming_audio_extension.proto"
    file_proto.package = "sglang.grpc.scheduler"
    file_proto.syntax = "proto3"
    file_proto.dependency.append("sglang_scheduler.proto")

    streaming_audio_config = file_proto.message_type.add()
    streaming_audio_config.name = "StreamingAudioConfig"

    field = streaming_audio_config.field.add()
    field.name = "session_id"
    field.number = 1
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = streaming_audio_config.field.add()
    field.name = "parent_rid"
    field.number = 2
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = streaming_audio_config.field.add()
    field.name = "system_prompt"
    field.number = 3
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = streaming_audio_config.field.add()
    field.name = "sampling_params"
    field.number = 4
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".sglang.grpc.scheduler.SamplingParams"

    field = streaming_audio_config.field.add()
    field.name = "speaker"
    field.number = 5
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    streaming_audio_request = file_proto.message_type.add()
    streaming_audio_request.name = "StreamingAudioRequest"
    request_oneof = streaming_audio_request.oneof_decl.add()
    request_oneof.name = "request"

    field = streaming_audio_request.field.add()
    field.name = "request_id"
    field.number = 1
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = streaming_audio_request.field.add()
    field.name = "config"
    field.number = 2
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".sglang.grpc.scheduler.StreamingAudioConfig"
    field.oneof_index = 0

    field = streaming_audio_request.field.add()
    field.name = "audio_chunk"
    field.number = 3
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    field.oneof_index = 0

    field = streaming_audio_request.field.add()
    field.name = "end_of_input"
    field.number = 4
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
    field.oneof_index = 0

    streaming_audio_complete = file_proto.message_type.add()
    streaming_audio_complete.name = "StreamingAudioComplete"

    field = streaming_audio_complete.field.add()
    field.name = "full_text"
    field.number = 1
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = streaming_audio_complete.field.add()
    field.name = "prompt_tokens"
    field.number = 2
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_INT32

    field = streaming_audio_complete.field.add()
    field.name = "completion_tokens"
    field.number = 3
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_INT32

    field = streaming_audio_complete.field.add()
    field.name = "thinker_token_ids"
    field.number = 4
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_UINT32

    field = streaming_audio_complete.field.add()
    field.name = "talker_token_ids"
    field.number = 5
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_UINT32

    field = streaming_audio_complete.field.add()
    field.name = "turn_rid"
    field.number = 6
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    streaming_audio_error = file_proto.message_type.add()
    streaming_audio_error.name = "StreamingAudioError"

    field = streaming_audio_error.field.add()
    field.name = "message"
    field.number = 1
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = streaming_audio_error.field.add()
    field.name = "code"
    field.number = 2
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    streaming_audio_response = file_proto.message_type.add()
    streaming_audio_response.name = "StreamingAudioResponse"
    response_oneof = streaming_audio_response.oneof_decl.add()
    response_oneof.name = "response"

    field = streaming_audio_response.field.add()
    field.name = "request_id"
    field.number = 1
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = streaming_audio_response.field.add()
    field.name = "text_delta"
    field.number = 2
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field.oneof_index = 0

    field = streaming_audio_response.field.add()
    field.name = "audio_chunk"
    field.number = 3
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    field.oneof_index = 0

    field = streaming_audio_response.field.add()
    field.name = "complete"
    field.number = 4
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".sglang.grpc.scheduler.StreamingAudioComplete"
    field.oneof_index = 0

    field = streaming_audio_response.field.add()
    field.name = "error"
    field.number = 5
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".sglang.grpc.scheduler.StreamingAudioError"
    field.oneof_index = 0

    return _descriptor_pool.Default().AddSerializedFile(file_proto.SerializeToString())


STREAMING_AUDIO_DESCRIPTOR = _make_streaming_audio_descriptor()
_builder.BuildMessageAndEnumDescriptors(STREAMING_AUDIO_DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    STREAMING_AUDIO_DESCRIPTOR,
    "sglang.srt.grpc.sglang_scheduler_pb2",
    globals(),
)

DESCRIPTOR = _base_pb2.DESCRIPTOR

__all__ = [
    *[name for name in dir(_base_pb2) if not name.startswith("_")],
    "StreamingAudioConfig",
    "StreamingAudioRequest",
    "StreamingAudioComplete",
    "StreamingAudioError",
    "StreamingAudioResponse",
    "STREAMING_AUDIO_DESCRIPTOR",
]
