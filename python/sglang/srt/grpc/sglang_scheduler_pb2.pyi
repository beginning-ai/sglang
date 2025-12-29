from smg_grpc_proto.generated.sglang_scheduler_pb2 import *  # noqa: F401,F403
from google.protobuf.message import Message


class StreamingAudioConfig(Message): ...


class StreamingAudioRequest(Message): ...


class StreamingAudioComplete(Message): ...


class StreamingAudioError(Message): ...


class StreamingAudioResponse(Message): ...
