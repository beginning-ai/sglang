"""Compatibility wrapper for scheduler gRPC bindings.

Standard scheduler RPCs come from `smg_grpc_proto.generated.sglang_scheduler_pb2_grpc`.
We add the local `StreamingAudio` stream-stream RPC on top.
"""

import grpc

from smg_grpc_proto.generated import sglang_scheduler_pb2_grpc as _base_pb2_grpc

from . import sglang_scheduler_pb2 as sglang__scheduler__pb2

GRPC_GENERATED_VERSION = _base_pb2_grpc.GRPC_GENERATED_VERSION
GRPC_VERSION = _base_pb2_grpc.GRPC_VERSION
SglangScheduler = getattr(_base_pb2_grpc, "SglangScheduler", None)


class SglangSchedulerStub(_base_pb2_grpc.SglangSchedulerStub):
    def __init__(self, channel):
        super().__init__(channel)
        self.StreamingAudio = channel.stream_stream(
            "/sglang.grpc.scheduler.SglangScheduler/StreamingAudio",
            request_serializer=(
                sglang__scheduler__pb2.StreamingAudioRequest.SerializeToString
            ),
            response_deserializer=(
                sglang__scheduler__pb2.StreamingAudioResponse.FromString
            ),
        )


class SglangSchedulerServicer(_base_pb2_grpc.SglangSchedulerServicer):
    async def StreamingAudio(self, request_iterator, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_SglangSchedulerServicer_to_server(servicer, server):
    _base_pb2_grpc.add_SglangSchedulerServicer_to_server(servicer, server)
    rpc_method_handlers = {
        "StreamingAudio": grpc.stream_stream_rpc_method_handler(
            servicer.StreamingAudio,
            request_deserializer=(
                sglang__scheduler__pb2.StreamingAudioRequest.FromString
            ),
            response_serializer=(
                sglang__scheduler__pb2.StreamingAudioResponse.SerializeToString
            ),
        )
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "sglang.grpc.scheduler.SglangScheduler",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))
