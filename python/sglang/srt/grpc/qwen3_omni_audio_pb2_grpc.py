"""gRPC bindings for the Qwen3-Omni audio API."""

import grpc

from sglang.srt.grpc import qwen3_omni_audio_pb2 as audio__pb2

GRPC_GENERATED_VERSION = "dynamic"
GRPC_VERSION = grpc.__version__


class Qwen3OmniAudioStub:
    def __init__(self, channel):
        self.AudioTurn = channel.stream_stream(
            "/sglang.grpc.qwen3_omni.Qwen3OmniAudio/AudioTurn",
            request_serializer=audio__pb2.AudioTurnRequest.SerializeToString,
            response_deserializer=audio__pb2.AudioTurnEvent.FromString,
        )
        self.RebuildAudioSession = channel.stream_unary(
            "/sglang.grpc.qwen3_omni.Qwen3OmniAudio/RebuildAudioSession",
            request_serializer=(
                audio__pb2.RebuildAudioSessionRequest.SerializeToString
            ),
            response_deserializer=(
                audio__pb2.RebuildAudioSessionResponse.FromString
            ),
        )
        self.CloseAudioSession = channel.unary_unary(
            "/sglang.grpc.qwen3_omni.Qwen3OmniAudio/CloseAudioSession",
            request_serializer=(
                audio__pb2.CloseAudioSessionRequest.SerializeToString
            ),
            response_deserializer=(
                audio__pb2.CloseAudioSessionResponse.FromString
            ),
        )
        self.HeartbeatAudioSession = channel.unary_unary(
            "/sglang.grpc.qwen3_omni.Qwen3OmniAudio/HeartbeatAudioSession",
            request_serializer=(
                audio__pb2.HeartbeatAudioSessionRequest.SerializeToString
            ),
            response_deserializer=(
                audio__pb2.HeartbeatAudioSessionResponse.FromString
            ),
        )


class Qwen3OmniAudioServicer:
    async def AudioTurn(self, request_iterator, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    async def RebuildAudioSession(self, request_iterator, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    async def CloseAudioSession(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    async def HeartbeatAudioSession(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_Qwen3OmniAudioServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "AudioTurn": grpc.stream_stream_rpc_method_handler(
            servicer.AudioTurn,
            request_deserializer=audio__pb2.AudioTurnRequest.FromString,
            response_serializer=audio__pb2.AudioTurnEvent.SerializeToString,
        ),
        "RebuildAudioSession": grpc.stream_unary_rpc_method_handler(
            servicer.RebuildAudioSession,
            request_deserializer=(
                audio__pb2.RebuildAudioSessionRequest.FromString
            ),
            response_serializer=(
                audio__pb2.RebuildAudioSessionResponse.SerializeToString
            ),
        ),
        "CloseAudioSession": grpc.unary_unary_rpc_method_handler(
            servicer.CloseAudioSession,
            request_deserializer=audio__pb2.CloseAudioSessionRequest.FromString,
            response_serializer=(
                audio__pb2.CloseAudioSessionResponse.SerializeToString
            ),
        ),
        "HeartbeatAudioSession": grpc.unary_unary_rpc_method_handler(
            servicer.HeartbeatAudioSession,
            request_deserializer=audio__pb2.HeartbeatAudioSessionRequest.FromString,
            response_serializer=(
                audio__pb2.HeartbeatAudioSessionResponse.SerializeToString
            ),
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "sglang.grpc.qwen3_omni.Qwen3OmniAudio",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))
