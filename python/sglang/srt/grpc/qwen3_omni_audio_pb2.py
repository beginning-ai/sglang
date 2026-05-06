"""Dynamic protobuf bindings for the Qwen3-Omni audio gRPC API."""

from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_sym_db = _symbol_database.Default()


def _add_field(msg, name, number, field_type, *, type_name=None, oneof_index=None):
    field = msg.field.add()
    field.name = name
    field.number = number
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = field_type
    if type_name is not None:
        field.type_name = type_name
    if oneof_index is not None:
        field.oneof_index = oneof_index


def _make_descriptor():
    file_proto = _descriptor_pb2.FileDescriptorProto()
    file_proto.name = "qwen3_omni_audio.proto"
    file_proto.package = "sglang.grpc.qwen3_omni"
    file_proto.syntax = "proto3"

    string = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    bytes_ = _descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    bool_ = _descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
    int32 = _descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    double = _descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    message = _descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE

    config = file_proto.message_type.add()
    config.name = "AudioTurnConfig"
    _add_field(config, "request_id", 1, string)
    _add_field(config, "session_id", 2, string)
    _add_field(config, "parent_rid", 3, string)
    _add_field(config, "system_prompt", 4, string)
    _add_field(config, "speaker", 5, string)
    _add_field(config, "sampling_params_json", 6, string)

    request = file_proto.message_type.add()
    request.name = "AudioTurnRequest"
    request.oneof_decl.add().name = "event"
    _add_field(
        request,
        "config",
        1,
        message,
        type_name=".sglang.grpc.qwen3_omni.AudioTurnConfig",
        oneof_index=0,
    )
    _add_field(request, "pcm16_audio", 2, bytes_, oneof_index=0)
    _add_field(request, "end_of_input", 3, bool_, oneof_index=0)
    _add_field(request, "abort_reason", 4, string, oneof_index=0)

    complete = file_proto.message_type.add()
    complete.name = "AudioTurnComplete"
    _add_field(complete, "turn_rid", 1, string)
    _add_field(complete, "full_text", 2, string)
    _add_field(complete, "prompt_tokens", 3, int32)
    _add_field(complete, "completion_tokens", 4, int32)
    _add_field(complete, "thinker_token_count", 5, int32)
    _add_field(complete, "talker_token_count", 6, int32)

    error = file_proto.message_type.add()
    error.name = "AudioTurnError"
    _add_field(error, "code", 1, string)
    _add_field(error, "message", 2, string)

    event = file_proto.message_type.add()
    event.name = "AudioTurnEvent"
    event.oneof_decl.add().name = "event"
    _add_field(event, "request_id", 1, string)
    _add_field(event, "text_delta", 2, string, oneof_index=0)
    _add_field(event, "pcm16_audio", 3, bytes_, oneof_index=0)
    _add_field(
        event,
        "complete",
        4,
        message,
        type_name=".sglang.grpc.qwen3_omni.AudioTurnComplete",
        oneof_index=0,
    )
    _add_field(
        event,
        "error",
        5,
        message,
        type_name=".sglang.grpc.qwen3_omni.AudioTurnError",
        oneof_index=0,
    )

    rebuild_config = file_proto.message_type.add()
    rebuild_config.name = "RebuildAudioSessionConfig"
    _add_field(rebuild_config, "request_id", 1, string)
    _add_field(rebuild_config, "session_id", 2, string)
    _add_field(rebuild_config, "system_prompt", 3, string)
    _add_field(rebuild_config, "speaker", 4, string)
    _add_field(rebuild_config, "sampling_params_json", 5, string)

    rebuild_audio = file_proto.message_type.add()
    rebuild_audio.name = "RebuildAudioSessionUserAudio"
    _add_field(rebuild_audio, "turn_index", 1, int32)
    _add_field(rebuild_audio, "pcm16_audio", 2, bytes_)

    rebuild_text = file_proto.message_type.add()
    rebuild_text.name = "RebuildAudioSessionAssistantText"
    _add_field(rebuild_text, "turn_index", 1, int32)
    _add_field(rebuild_text, "turn_rid", 2, string)
    _add_field(rebuild_text, "text", 3, string)

    rebuild_req = file_proto.message_type.add()
    rebuild_req.name = "RebuildAudioSessionRequest"
    rebuild_req.oneof_decl.add().name = "event"
    _add_field(
        rebuild_req,
        "config",
        1,
        message,
        type_name=".sglang.grpc.qwen3_omni.RebuildAudioSessionConfig",
        oneof_index=0,
    )
    _add_field(
        rebuild_req,
        "user_audio",
        2,
        message,
        type_name=".sglang.grpc.qwen3_omni.RebuildAudioSessionUserAudio",
        oneof_index=0,
    )
    _add_field(
        rebuild_req,
        "assistant_text",
        3,
        message,
        type_name=".sglang.grpc.qwen3_omni.RebuildAudioSessionAssistantText",
        oneof_index=0,
    )

    rebuild_resp = file_proto.message_type.add()
    rebuild_resp.name = "RebuildAudioSessionResponse"
    _add_field(rebuild_resp, "success", 1, bool_)
    _add_field(rebuild_resp, "message", 2, string)
    _add_field(rebuild_resp, "latest_turn_rid", 3, string)
    _add_field(rebuild_resp, "prefix_turn_count", 4, int32)
    _add_field(rebuild_resp, "prompt_tokens", 5, int32)

    close_req = file_proto.message_type.add()
    close_req.name = "CloseAudioSessionRequest"
    _add_field(close_req, "session_id", 1, string)
    _add_field(close_req, "reason", 2, string)

    close_resp = file_proto.message_type.add()
    close_resp.name = "CloseAudioSessionResponse"
    _add_field(close_resp, "success", 1, bool_)
    _add_field(close_resp, "message", 2, string)
    _add_field(close_resp, "turn_count", 3, int32)
    _add_field(close_resp, "active_request_count", 4, int32)

    heartbeat_req = file_proto.message_type.add()
    heartbeat_req.name = "HeartbeatAudioSessionRequest"
    _add_field(heartbeat_req, "session_id", 1, string)
    _add_field(heartbeat_req, "lease_token", 2, string)
    _add_field(heartbeat_req, "ttl_secs", 3, int32)

    heartbeat_resp = file_proto.message_type.add()
    heartbeat_resp.name = "HeartbeatAudioSessionResponse"
    _add_field(heartbeat_resp, "success", 1, bool_)
    _add_field(heartbeat_resp, "message", 2, string)
    _add_field(heartbeat_resp, "lease_expires_at_unix", 3, double)

    service = file_proto.service.add()
    service.name = "Qwen3OmniAudio"

    method = service.method.add()
    method.name = "AudioTurn"
    method.input_type = ".sglang.grpc.qwen3_omni.AudioTurnRequest"
    method.output_type = ".sglang.grpc.qwen3_omni.AudioTurnEvent"
    method.client_streaming = True
    method.server_streaming = True

    method = service.method.add()
    method.name = "RebuildAudioSession"
    method.input_type = ".sglang.grpc.qwen3_omni.RebuildAudioSessionRequest"
    method.output_type = ".sglang.grpc.qwen3_omni.RebuildAudioSessionResponse"
    method.client_streaming = True

    method = service.method.add()
    method.name = "CloseAudioSession"
    method.input_type = ".sglang.grpc.qwen3_omni.CloseAudioSessionRequest"
    method.output_type = ".sglang.grpc.qwen3_omni.CloseAudioSessionResponse"

    method = service.method.add()
    method.name = "HeartbeatAudioSession"
    method.input_type = ".sglang.grpc.qwen3_omni.HeartbeatAudioSessionRequest"
    method.output_type = ".sglang.grpc.qwen3_omni.HeartbeatAudioSessionResponse"

    return _descriptor_pool.Default().AddSerializedFile(
        file_proto.SerializeToString()
    )


DESCRIPTOR = _make_descriptor()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR,
    "sglang.srt.grpc.qwen3_omni_audio_pb2",
    globals(),
)
