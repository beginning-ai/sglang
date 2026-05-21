"""
Thin gRPC server wrapper — delegates to smg-grpc-servicer package.

A lightweight HTTP sidecar is started alongside the gRPC server to expose:
- /metrics (Prometheus, when --enable-metrics is set)
- /start_profile, /stop_profile (profiling control)

The sidecar is started on --grpc-http-sidecar-port (default: --port + 1)
once the gRPC request manager is ready, regardless of whether --enable-metrics
is set.
"""

import asyncio
import inspect
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from aiohttp import web

from sglang.srt.managers.io_struct import (
    GetInternalStateReq,
    ProfileReq,
    ProfileReqType,
)
from sglang.srt.utils.common import get_bool_env_var, is_hip, kill_process_tree

logger = logging.getLogger(__name__)

def _qwen3_omni_audio_prewarm_script() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "test/manual/qwen3_omni_audio_prewarm.py"
    )


def _run_qwen3_omni_launch_prewarm(server_args) -> None:
    script = _qwen3_omni_audio_prewarm_script()
    if not script.is_file():
        raise FileNotFoundError(script)
    max_concurrent = server_args.qwen3_omni_max_concurrent_audio_turns
    if max_concurrent is None:
        raise RuntimeError(
            "Qwen3-Omni launch prewarm requires "
            "--qwen3-omni-max-concurrent-audio-turns."
        )
    batch_sizes = [1, 2, 4, 8, 16, 32]
    batch_sizes = [bs for bs in batch_sizes if bs < max_concurrent]
    batch_sizes.append(max_concurrent)

    env = os.environ.copy()
    python_root = str(Path(__file__).resolve().parents[3])
    manual_root = str(script.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [python_root, manual_root, env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [python_root, manual_root]
    )
    cmd = [
        sys.executable,
        str(script),
        "--host",
        server_args.host,
        "--port",
        str(server_args.port),
        "--batch-sizes",
        ",".join(str(bs) for bs in batch_sizes),
        "--chunk-ms",
        "1000",
        "--speaker",
        "chelsie",
        "--max-new-tokens",
        "64",
        "--talker-max-new-tokens",
        "256",
    ]
    logger.info("Running Qwen3-Omni launch prewarm: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _wait_for_qwen3_omni_grpc_model_info(server_args) -> None:
    import grpc
    from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc

    channel = grpc.insecure_channel(
        f"{server_args.host}:{server_args.port}",
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )
    last_error = None
    for _ in range(120):
        time.sleep(1)
        try:
            stub = sglang_scheduler_pb2_grpc.SglangSchedulerStub(channel)
            stub.GetModelInfo(sglang_scheduler_pb2.GetModelInfoRequest(), timeout=5)
            channel.close()
            return
        except Exception as exc:
            last_error = exc
    channel.close()
    raise RuntimeError(f"Qwen3-Omni gRPC model-info warmup failed: {last_error}")


def _is_qwen3_omni_model(server_args) -> bool:
    hf_config = server_args.get_model_config().hf_config
    architectures = getattr(hf_config, "architectures", []) or []
    return (
        getattr(hf_config, "model_type", None) == "qwen3_omni_moe"
        or "Qwen3OmniMoeForConditionalGeneration" in architectures
    )


def _should_run_qwen3_omni_launch_prewarm(server_args) -> bool:
    return _is_qwen3_omni_model(server_args) and is_hip()


def _install_qwen3_omni_audio_service(server_args):
    """Attach the local Qwen3-Omni audio service to the SMG gRPC server."""
    import zmq
    from smg_grpc_proto import sglang_scheduler_pb2_grpc
    from smg_grpc_servicer.sglang.request_manager import GrpcRequestManager

    from sglang.srt.managers.io_struct import (
        AbortReq,
        AudioSessionRebuildReqOutput,
        BatchEmbeddingOutput,
        BatchTokenIDOutput,
        CloseAudioSessionReqOutput,
        GetInternalStateReqOutput,
        GetLoadsReqOutput,
        HeartbeatAudioSessionReqOutput,
        HealthCheckOutput,
    )
    from sglang.srt.observability.req_time_stats import real_time
    from sglang.utils import get_exception_traceback

    from sglang.srt.grpc.audio_request_manager import (
        AudioStreamRequestManager,
        install_audio_output_routing,
    )
    from sglang.srt.grpc.qwen3_omni_audio_pb2_grpc import (
        add_Qwen3OmniAudioServicer_to_server,
    )
    from sglang.srt.grpc.qwen3_omni_audio_servicer import Qwen3OmniAudioServicer

    if not getattr(GrpcRequestManager.handle_loop, "_qwen3_omni_audio_wrapped", False):
        async def handle_loop_with_audio_responses(self):
            while not self.gracefully_exit:
                try:
                    recv_obj = await self.recv_from_scheduler.recv_pyobj()
                    self.last_receive_tstamp = real_time()

                    if self.is_pause:
                        async with self.is_pause_cond:
                            while self.is_pause:
                                await self.is_pause_cond.wait()

                    if isinstance(recv_obj, BatchTokenIDOutput):
                        await self._handle_batch_output(recv_obj)
                    elif isinstance(recv_obj, BatchEmbeddingOutput):
                        await self._handle_embedding_output(recv_obj)
                    elif isinstance(recv_obj, HealthCheckOutput):
                        await self._handle_health_check_output(recv_obj)
                    elif isinstance(recv_obj, AbortReq):
                        await self._handle_abort_req(recv_obj)
                    elif isinstance(recv_obj, GetLoadsReqOutput):
                        self.get_loads_communicator.handle_recv(recv_obj)
                    elif isinstance(recv_obj, GetInternalStateReqOutput):
                        self.get_internal_state_communicator.handle_recv(recv_obj)
                    elif isinstance(
                        recv_obj,
                        (
                            CloseAudioSessionReqOutput,
                            HeartbeatAudioSessionReqOutput,
                            AudioSessionRebuildReqOutput,
                        ),
                    ):
                        audio_manager = getattr(
                            self,
                            "qwen3_omni_audio_manager",
                            None,
                        )
                        if audio_manager is not None:
                            await audio_manager.route_scheduler_output(recv_obj)
                        else:
                            logger.warning("Audio session response before service init")
                    else:
                        logger.warning("Unknown output type: %s", type(recv_obj))
                except zmq.error.Again:
                    if self.gracefully_exit:
                        break
                    continue
                except zmq.error.ZMQError as e:
                    if self.gracefully_exit:
                        logger.debug("ZMQ recv interrupted during shutdown: %s", e)
                        break
                    logger.error("ZMQ error in handle loop: %s\n%s", e, get_exception_traceback())
                    break
                except Exception as e:
                    logger.error("Handle loop error: %s\n%s", e, get_exception_traceback())
                    if self.gracefully_exit:
                        break

        handle_loop_with_audio_responses._qwen3_omni_audio_wrapped = True
        GrpcRequestManager.handle_loop = handle_loop_with_audio_responses

    original_add = sglang_scheduler_pb2_grpc.add_SglangSchedulerServicer_to_server
    if getattr(original_add, "_qwen3_omni_audio_wrapped", False):
        return

    def add_scheduler_and_audio_servicers(servicer, grpc_server):
        original_add(servicer, grpc_server)
        audio_manager = AudioStreamRequestManager(server_args)
        servicer.request_manager.qwen3_omni_audio_manager = audio_manager
        install_audio_output_routing(servicer.request_manager, audio_manager)
        add_Qwen3OmniAudioServicer_to_server(
            Qwen3OmniAudioServicer(
                request_manager=servicer.request_manager,
                audio_manager=audio_manager,
                server_args=server_args,
            ),
            grpc_server,
        )

    add_scheduler_and_audio_servicers._qwen3_omni_audio_wrapped = True
    sglang_scheduler_pb2_grpc.add_SglangSchedulerServicer_to_server = (
        add_scheduler_and_audio_servicers
    )


async def _start_sidecar_server(host: str, port: int, app):
    """Start the aiohttp sidecar and return the runner for cleanup."""
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
    except BaseException:
        await runner.cleanup()
        raise
    logger.info("HTTP sidecar server started on http://%s:%d", host, port)
    return runner


def _add_metrics_routes(app):
    """Add Prometheus /metrics endpoint to the aiohttp app."""
    from prometheus_client import (
        CollectorRegistry,
        multiprocess,
    )
    from prometheus_client.openmetrics.exposition import (
        CONTENT_TYPE_LATEST,
        generate_latest,
    )
    from sglang.srt.observability.qwen3_omni_audio_metrics import (
        QWEN3_OMNI_AUDIO_REGISTRY,
    )

    async def metrics_handler(request):
        try:
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            data = generate_latest(registry)
            qwen3_omni_data = generate_latest(QWEN3_OMNI_AUDIO_REGISTRY)
            if qwen3_omni_data != b"# EOF\n":
                data = data.removesuffix(b"# EOF\n") + qwen3_omni_data
            return web.Response(
                body=data,
                headers={"Content-Type": CONTENT_TYPE_LATEST},
            )
        except Exception:
            logger.exception("Failed to generate Prometheus metrics")
            return web.Response(status=500, text="Failed to generate metrics")

    app.router.add_get("/metrics", metrics_handler)


def _check_communicator_results(results, action):
    """Return a web.Response error if results indicate failure, else None."""
    if not results:
        return web.Response(status=500, text="No response from scheduler\n")
    failures = [r for r in results if not r.success]
    if failures:
        msgs = " | ".join(r.message for r in failures)
        return web.Response(status=500, text=f"{action} failed: {msgs}\n")
    return None


def _add_admin_routes(app, request_manager):
    """Add admin endpoints to the aiohttp app.

    Endpoints: /start_profile, /stop_profile.
    Business logic (request construction, env var handling, response interpretation)
    lives here; request_manager only provides the transport to the scheduler.
    """

    async def start_profile_handler(request):
        try:
            if request.content_length and request.content_length > 0:
                try:
                    body = await request.json()
                except json.JSONDecodeError as e:
                    return web.Response(
                        status=400,
                        text=f"Invalid JSON in request body: {e}",
                    )
            else:
                body = {}

            # Build ProfileReq with env var overrides (same as tokenizer_communicator_mixin)
            with_stack = body.get("with_stack")
            env_with_stack = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
            with_stack = (with_stack is not False) and env_with_stack
            record_shapes = body.get("record_shapes")
            env_record_shapes = get_bool_env_var("SGLANG_PROFILE_RECORD_SHAPES", "true")
            record_shapes = (record_shapes is not False) and env_record_shapes

            req = ProfileReq(
                type=ProfileReqType.START_PROFILE,
                output_dir=body.get("output_dir"),
                start_step=body.get("start_step"),
                num_steps=body.get("num_steps"),
                activities=body.get("activities"),
                with_stack=with_stack,
                record_shapes=record_shapes,
                profile_by_stage=body.get("profile_by_stage", False),
                profile_id=str(time.time()),
                merge_profiles=body.get("merge_profiles", False),
                profile_prefix=body.get("profile_prefix"),
                profile_stages=body.get("profile_stages"),
            )
            results = await request_manager.send_communicator_req(
                req, "profile_communicator", timeout=600.0
            )
            err = _check_communicator_results(results, "Start Profile")
            if err:
                return err
            return web.Response(text="Start profiling.\n")
        except Exception as e:
            logger.exception("Failed to start profile")
            return web.Response(
                status=500,
                text=f"Internal error: {type(e).__name__}. Check server logs.\n",
            )

    async def stop_profile_handler(request):
        try:
            req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
            results = await request_manager.send_communicator_req(
                req, "profile_communicator", timeout=600.0
            )
            err = _check_communicator_results(results, "Stop profile")
            if err:
                return err
            return web.Response(text="Stop profiling. This will take some time.\n")
        except Exception as e:
            logger.exception("Failed to stop profile")
            return web.Response(
                status=500,
                text=f"Internal error: {type(e).__name__}. Check server logs.\n",
            )

    async def internal_state_handler(request):
        try:
            results = await request_manager.send_communicator_req(
                GetInternalStateReq(),
                "get_internal_state_communicator",
                timeout=30.0,
            )
            states = []
            for result in results:
                state = result.internal_state
                states.append(
                    {
                        "memory_usage": state.get("memory_usage"),
                        "qwen3_omni_audio": state.get("qwen3_omni_audio"),
                    }
                )
            return web.json_response({"states": states})
        except Exception:
            logger.exception("Failed to collect internal state")
            return web.Response(
                status=500,
                text="Failed to collect internal state. Check server logs.\n",
            )

    app.router.add_post("/start_profile", start_profile_handler)
    app.router.add_post("/stop_profile", stop_profile_handler)
    app.router.add_get("/internal_state", internal_state_handler)


async def serve_grpc(server_args, model_info=None):
    """Start the standalone gRPC server with integrated scheduler."""
    try:
        import smg_grpc_servicer.sglang.server as smg_grpc_server
    except ImportError as e:
        raise ImportError(
            "gRPC mode requires the smg-grpc-servicer package. "
            "If not installed, run: pip install smg-grpc-servicer[sglang]. "
            "If already installed, there may be a broken import due to a "
            "version mismatch — see the chained exception above for details."
        ) from e

    _install_qwen3_omni_audio_service(server_args)
    _serve_grpc = smg_grpc_server.serve_grpc

    qwen3_omni_launch_prewarm = _should_run_qwen3_omni_launch_prewarm(server_args)
    if qwen3_omni_launch_prewarm and server_args.skip_server_warmup:
        raise RuntimeError(
            "Qwen3-Omni gRPC launch prewarm is required for readiness; "
            "remove --skip-server-warmup."
        )
    if qwen3_omni_launch_prewarm:
        def _execute_grpc_server_warmup_with_qwen3_omni_audio(srv_args):
            try:
                _wait_for_qwen3_omni_grpc_model_info(srv_args)
                _run_qwen3_omni_launch_prewarm(srv_args)
            except Exception:
                logger.exception("Qwen3-Omni launch prewarm failed")
                kill_process_tree(os.getpid())
                return False
            return True

        smg_grpc_server._execute_grpc_server_warmup = (
            _execute_grpc_server_warmup_with_qwen3_omni_audio
        )

    sidecar_app = web.Application()
    sidecar_runner = None
    sidecar_port = (
        server_args.grpc_http_sidecar_port
        if server_args.grpc_http_sidecar_port is not None
        else server_args.port + 1
    )

    # Metrics setup: must set PROMETHEUS_MULTIPROC_DIR before scheduler
    # processes import prometheus_client, since the env var is inherited
    # at fork time.
    if server_args.enable_metrics:
        try:
            from sglang.srt.observability.func_timer import enable_func_timer
            from sglang.srt.utils import set_prometheus_multiproc_dir

            set_prometheus_multiproc_dir()
            enable_func_timer()
            _add_metrics_routes(sidecar_app)
        except Exception as e:
            logger.error(
                "Failed to set up metrics: %s. Continuing without metrics.",
                e,
                exc_info=True,
            )

    async def _on_request_manager_ready(request_manager, srv_args, sched_info):
        nonlocal sidecar_runner
        try:
            _add_admin_routes(sidecar_app, request_manager)
        except Exception as e:
            logger.error(
                "Failed to set up admin routes: %s. "
                "Continuing without admin endpoints.",
                e,
                exc_info=True,
            )
        try:
            sidecar_runner = await _start_sidecar_server(
                server_args.host, sidecar_port, sidecar_app
            )
        except OSError as e:
            logger.error(
                "Failed to start HTTP sidecar server: %s. "
                "Continuing without metrics/profile endpoints.",
                e,
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                "Unexpected error starting HTTP sidecar server: %s. "
                "Continuing without metrics/profile endpoints.",
                e,
                exc_info=True,
            )

    # Published smg-grpc-servicer releases through 0.5.3 accept only
    # (server_args, model_info) and reject the on_request_manager_ready hook.
    # Patch the servicer class in that case so the sidecar still starts when the
    # request manager becomes available.
    serve_kwargs: dict = {}
    sidecar_supported = (
        "on_request_manager_ready" in inspect.signature(_serve_grpc).parameters
    )
    if sidecar_supported:
        serve_kwargs["on_request_manager_ready"] = _on_request_manager_ready
    else:
        from smg_grpc_proto import sglang_scheduler_pb2

        original_servicer_cls = smg_grpc_server.SGLangSchedulerServicer
        model_info_fields = (
            sglang_scheduler_pb2.GetModelInfoResponse.DESCRIPTOR.fields_by_name
        )

        class SidecarAwareSGLangSchedulerServicer(original_servicer_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                asyncio.get_running_loop().create_task(
                    _on_request_manager_ready(
                        self.request_manager,
                        self.server_args,
                        self.scheduler_info,
                    )
                )

            async def GetModelInfo(self, request, context):
                if "default_sampling_params_json" in model_info_fields:
                    return await super().GetModelInfo(request, context)

                is_generation = self.scheduler_info.get("is_generation")
                if is_generation is None:
                    is_generation = not self.server_args.is_embedding

                return sglang_scheduler_pb2.GetModelInfoResponse(
                    model_path=self.server_args.model_path,
                    tokenizer_path=self.server_args.tokenizer_path or "",
                    is_generation=is_generation,
                    preferred_sampling_params=self._preferred_sampling_params_json(),
                    weight_version=self.server_args.weight_version or "",
                    served_model_name=self.server_args.served_model_name,
                    max_context_length=self.model_info["max_context_length"],
                    vocab_size=self.model_info["vocab_size"],
                    supports_vision=self.model_info["supports_vision"],
                    model_type=self.model_info.get("model_type") or "",
                    architectures=self.model_info.get("architectures") or [],
                    eos_token_ids=self.model_info["eos_token_ids"],
                    pad_token_id=self.model_info["pad_token_id"],
                    bos_token_id=self.model_info["bos_token_id"],
                    max_req_input_len=self.model_info["max_req_input_len"],
                    id2label_json=self.model_info.get("id2label_json") or "",
                    num_labels=self.model_info.get("num_labels") or 0,
                )

        smg_grpc_server.SGLangSchedulerServicer = SidecarAwareSGLangSchedulerServicer

    try:
        await _serve_grpc(server_args, model_info, **serve_kwargs)
    finally:
        if sidecar_runner is not None:
            try:
                await sidecar_runner.cleanup()
            except Exception as e:
                logger.exception(
                    "Failed to cleanly shut down HTTP sidecar server: %s",
                    e,
                )
