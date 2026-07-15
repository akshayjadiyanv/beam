#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pytype: skip-file

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Optional

from openai import AsyncOpenAI
from openai import OpenAI

from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference._dynamo_runtime import DynamoRuntimeConfig
from apache_beam.ml.inference._dynamo_runtime import DynamoVLLMEngineSpec
from apache_beam.ml.inference._dynamo_runtime import _DynamoLocalRuntime
from apache_beam.ml.inference._dynamo_runtime import single_engine_config
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.utils import subprocess_server

try:
  # VLLM logging config breaks beam logging.
  os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
  import vllm  # pylint: disable=unused-import
  logging.info('vllm module successfully imported.')
  os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
except ModuleNotFoundError:
  msg = 'vllm module was not found. This is ok as long as the specified ' \
    'runner has vllm dependencies installed.'
  logging.warning(msg)

__all__ = [
    'OpenAIChatMessage',
    'VLLMCompletionsModelHandler',
    'VLLMChatModelHandler',
    'DynamoRuntimeConfig',
    'DynamoVLLMEngineSpec',
]


@dataclass(frozen=True)
class OpenAIChatMessage():
  """"
  Dataclass containing previous chat messages in conversation.
  Role is the entity that sent the message (either 'user' or 'system').
  Content is the contents of the message.
  """
  role: str
  content: str


def start_process(
    cmd,
    port: Optional[int] = None,
    env: Optional[dict] = None,
    component_label: Optional[str] = None) -> tuple[subprocess.Popen, int]:
  """Launch ``cmd`` in its own session, substituting ``{{PORT}}``.

  Args:
    cmd: Command list. Any ``{{PORT}}`` token is replaced with ``port``.
    port: Port to substitute/return. When ``None`` a free port is picked (the
      original single-server behaviour). Callers managing a multi-process
      topology pre-allocate every port and pass it explicitly so the whole
      topology's ports are guaranteed distinct.
    env: Child environment. When ``None`` the child inherits the SDK process
      environment. Passing an explicit copy avoids leaking topology-local state
      (``CUDA_VISIBLE_DEVICES``, ``DYN_SYSTEM_PORT``, ``ETCD_ENDPOINTS``) into
      the parent or sibling engines.
    component_label: Human-readable label for logs.
  """
  if port is None:
    port, = subprocess_server.pick_port(None)
  cmd = [arg.replace('{{PORT}}', str(port)) for arg in cmd]  # pylint: disable=not-an-iterable
  label = f' [{component_label}]' if component_label else ''
  logging.info("Starting service%s with %s", label, str(cmd).replace("',", "'"))
  # start_new_session makes the child a process-group leader so cleanup can
  # signal the whole group (vLLM + CUDA descendants) and not leak GPU memory.
  process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      env=env,
      start_new_session=True)

  # Emit the output of this command as info level logging.
  def log_stdout():
    line = process.stdout.readline()
    while line:
      # The log obtained from stdout is bytes, decode it into string.
      # Remove newline via rstrip() to not print an empty line.
      logging.info(line.decode(errors='backslashreplace').rstrip())
      line = process.stdout.readline()

  t = threading.Thread(target=log_stdout)
  t.daemon = True
  t.start()
  return process, port


def getVLLMClient(port) -> OpenAI:
  openai_api_key = "EMPTY"
  openai_api_base = f"http://localhost:{port}/v1"
  return OpenAI(
      api_key=openai_api_key,
      base_url=openai_api_base,
  )


def getAsyncVLLMClient(port) -> AsyncOpenAI:
  openai_api_key = "EMPTY"
  openai_api_base = f"http://localhost:{port}/v1"
  return AsyncOpenAI(
      api_key=openai_api_key,
      base_url=openai_api_base,
  )


def _append_kwargs(cmd: list[str], kwargs: dict[str, Optional[str]]) -> None:
  for k, v in kwargs.items():
    cmd.append(f'--{k}')
    # Only add values for commands with value part.
    if v is not None:
      cmd.append(v)


def _resolve_dynamo_config(
    use_dynamo: bool,
    dynamo_runtime_config: Optional[DynamoRuntimeConfig],
    dynamo_frontend_kwargs: Optional[dict[str, Optional[str]]],
) -> Optional[DynamoRuntimeConfig]:
  """Normalize the two Dynamo config surfaces into one validated config.

  ``use_dynamo=True`` without a typed config maps to the legacy single-engine
  topology. The typed ``dynamo_runtime_config`` and the legacy
  ``dynamo_frontend_kwargs`` escape hatch are mutually exclusive so they cannot
  silently contradict each other about router/event behaviour. Validation runs
  at handler-construction time to fail before any Dataflow launch.
  """
  if not use_dynamo:
    if dynamo_runtime_config is not None or dynamo_frontend_kwargs:
      raise ValueError(
          'dynamo_runtime_config and dynamo_frontend_kwargs require '
          'use_dynamo=True.')
    return None
  if dynamo_runtime_config is not None and dynamo_frontend_kwargs:
    raise ValueError(
        'Pass either dynamo_runtime_config or dynamo_frontend_kwargs, not '
        'both; the typed config already owns the frontend configuration.')
  config = dynamo_runtime_config or single_engine_config(dynamo_frontend_kwargs)
  config.validate()
  return config


class _VLLMModelServer():
  """Owns the per-worker inference server.

  Two adapters live behind one interface: the native vLLM OpenAI api_server
  (a single process) and the embedded Dynamo topology (etcd + frontend + N
  GPU-pinned engines), the latter delegated to :class:`_DynamoLocalRuntime`.
  ``run_inference`` only depends on :meth:`get_server_port` and
  :meth:`check_connectivity`, so both adapters share that surface.
  """
  def __init__(
      self,
      model_name: str,
      vllm_server_kwargs: dict[str, Optional[str]],
      use_dynamo: bool = False,
      dynamo_config: Optional[DynamoRuntimeConfig] = None):
    self._model_name = model_name
    self._vllm_server_kwargs = vllm_server_kwargs
    self._use_dynamo = use_dynamo
    self._dynamo_config = dynamo_config
    self._server_started = False
    self._server_process = None  # native vLLM api_server
    self._server_port: int = -1
    self._runtime: Optional[_DynamoLocalRuntime] = None  # embedded Dynamo
    self._server_process_lock = threading.RLock()

    self.start_server()

  @staticmethod
  def _stop_process(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
      return
    # A process may exit between poll() and terminate() / kill(), in which
    # case the OS raises ProcessLookupError (or another OSError). Treat that
    # as already-stopped so we don't bail out of the broader cleanup.
    try:
      process.terminate()
      try:
        process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    except OSError:
      pass

  def _stop_processes(self) -> None:
    if self._runtime is not None:
      self._runtime.stop()
      self._runtime = None
    self._stop_process(self._server_process)
    self._server_process = None
    self._server_started = False
    self._server_port = -1

  def _process_status(self) -> str:
    if self._runtime is not None:
      return self._runtime.status()
    if self._server_process is not None:
      return 'frontend/server exit code: %s' % self._server_process.poll()
    return 'no process status available'

  def __del__(self):
    # __del__ may run during interpreter shutdown when module globals can
    # already be torn down; swallow any cleanup failures so we don't print
    # a noisy traceback.
    try:
      self._stop_processes()
    except Exception:  # pylint: disable=broad-except
      pass

  def start_server(self, retries=3):
    with self._server_process_lock:
      if not self._server_started:
        self._stop_processes()
        if self._use_dynamo:
          # The whole embedded topology (etcd + frontend + engines) is owned by
          # the runtime; the frontend is the OpenAI-compatible local endpoint.
          self._runtime = _DynamoLocalRuntime(
              self._model_name,
              self._dynamo_config,
              self._vllm_server_kwargs,
              start_process,
              _append_kwargs)
          self._runtime.start()
          self._server_port = self._runtime.frontend_port
        else:
          server_cmd = [
              sys.executable,
              '-m',
              'vllm.entrypoints.openai.api_server',
              '--model',
              self._model_name,
              '--port',
              '{{PORT}}',
          ]
          _append_kwargs(server_cmd, self._vllm_server_kwargs)
          self._server_process, self._server_port = start_process(
              server_cmd, component_label='vllm')

      self.check_connectivity(retries)

  def get_server_port(self) -> int:
    if not self._server_started:
      self.start_server()
    return self._server_port

  def check_connectivity(self, retries=3, timeout_secs=600):
    if self._use_dynamo:
      self._check_connectivity_dynamo(retries, timeout_secs)
    else:
      self._check_connectivity_native(retries, timeout_secs)

  def _check_connectivity_dynamo(self, retries, timeout_secs):
    if self._runtime is None:
      # The topology was torn down (e.g. a prior failure); rebuild it.
      self.start_server(retries)
      return
    try:
      self._runtime.wait_until_ready(timeout_secs)
    except Exception as e:  # pylint: disable=broad-except
      process_status = self._process_status()
      self._stop_processes()
      if retries == 0:
        raise Exception(
            "Failed to start embedded Dynamo topology. " + process_status +
            ". Next time a request is tried, the server will be restarted"
        ) from e
      self.start_server(retries - 1)
      return
    for name, url in self._runtime.metrics_endpoints().items():
      logging.info('Dynamo metrics endpoint %s: %s', name, url)
    self._server_started = True

  def _check_connectivity_native(self, retries, timeout_secs):
    start_time = time.time()
    with getVLLMClient(self._server_port) as client:
      while (time.time() - start_time < timeout_secs and
             self._server_process.poll() is None):
        try:
          models = client.models.list().data
          logging.info('models: %s' % models)
          if len(models) > 0:
            self._server_started = True
            return
        except:  # pylint: disable=bare-except
          pass
        # Sleep while bringing up the process
        time.sleep(5)

      process_status = self._process_status()
      self._stop_processes()
      if retries == 0:
        raise Exception(
            "Failed to start vLLM server. Process status: "
            f"{process_status}. Next time a request is tried, the server "
            "will be restarted")
      else:
        self.start_server(retries - 1)


class VLLMCompletionsModelHandler(ModelHandler[str,
                                               PredictionResult,
                                               _VLLMModelServer]):
  def __init__(
      self,
      model_name: str,
      vllm_server_kwargs: Optional[dict[str, Optional[str]]] = None,
      *,
      use_dynamo: bool = False,
      dynamo_runtime_config: Optional[DynamoRuntimeConfig] = None,
      dynamo_frontend_kwargs: Optional[dict[str, Optional[str]]] = None,
      min_batch_size: Optional[int] = None,
      max_batch_size: Optional[int] = None,
      max_batch_duration_secs: Optional[int] = None,
      max_batch_weight: Optional[int] = None,
      element_size_fn: Optional[Callable[[Any], int]] = None,
      batch_length_fn: Optional[Callable[[Any], int]] = None,
      batch_bucket_boundaries: Optional[list[int]] = None):
    """Implementation of the ModelHandler interface for vLLM using text as
    input.

    Example Usage::

      pcoll | RunInference(VLLMModelHandler(model_name='facebook/opt-125m'))

    Args:
      model_name: The vLLM model. See
        https://docs.vllm.ai/en/latest/models/supported_models.html for
        supported models.
      vllm_server_kwargs: Any additional kwargs to be passed into your vllm
        server when it is being created. When ``use_dynamo`` is disabled,
        this is invoked using ``python -m vllm.entrypoints.openai.api_server
        <beam provided args> <vllm_server_kwargs>``. When ``use_dynamo`` is
        enabled, these kwargs are passed to the ``dynamo.vllm`` worker
        process. For example, you could pass ``{'echo': 'true'}`` to prepend
        new messages with the previous message. On ~16GB GPUs, pass lower
        ``max-num-seqs`` and ``gpu-memory-utilization`` values (see
        ``apache_beam.examples.inference.vllm_text_completion``). For a list
        of possible kwargs, see
        https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-completions-api
      use_dynamo: Whether to use NVIDIA Dynamo as the underlying vLLM engine.
        Requires installing Dynamo in your runtime environment
        (``pip install ai-dynamo[vllm]``). By default this is an opt-in
        single-engine embedded mode; pass ``dynamo_runtime_config`` to run
        multiple GPU-pinned engines with KV-aware routing. Disaggregated
        prefill/decode, KVBM offload across nodes, the Planner, and Grove are
        not active in embedded mode. Dynamo also requires an etcd-style
        discovery service: when ``ETCD_ENDPOINTS`` is unset, Beam starts a
        local etcd, which requires the ``etcd`` binary in the worker
        environment.
      dynamo_runtime_config: A :class:`DynamoRuntimeConfig` describing the
        whole local Dynamo topology -- the engines and their GPU placement,
        the router mode (``round-robin`` or ``kv``), and how KV-cache state
        reaches the router (``disabled``/``approximate``/``zmq``). When
        omitted, ``use_dynamo=True`` maps to the legacy one-engine,
        round-robin, KV-events-disabled behaviour. Mutually exclusive with
        ``dynamo_frontend_kwargs``.
      dynamo_frontend_kwargs: Legacy escape hatch for extra
        ``dynamo.frontend`` flags in single-engine mode. Prefer
        ``dynamo_runtime_config`` for anything beyond ad-hoc frontend flags;
        the two may not be combined.
      min_batch_size: optional. the minimum batch size to use when batching
        inputs.
      max_batch_size: optional. the maximum batch size to use when batching
        inputs.
      max_batch_duration_secs: optional. the maximum amount of time to buffer
        a batch before emitting; used in streaming contexts.
      max_batch_weight: optional. the maximum total weight of a batch.
      element_size_fn: optional. a function that returns the size (weight) of
        an element.
      batch_length_fn: optional. a callable that returns the length of an
        element for length-aware batching.
      batch_bucket_boundaries: optional. a sorted list of positive boundary
        values for length-aware batching buckets.
    """
    super().__init__(
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        max_batch_duration_secs=max_batch_duration_secs,
        max_batch_weight=max_batch_weight,
        element_size_fn=element_size_fn,
        batch_length_fn=batch_length_fn,
        batch_bucket_boundaries=batch_bucket_boundaries)
    self._model_name = model_name
    self._vllm_server_kwargs: dict[str, Optional[str]] = dict(
        vllm_server_kwargs or {})
    self._use_dynamo = use_dynamo
    self._dynamo_config = _resolve_dynamo_config(
        use_dynamo, dynamo_runtime_config, dynamo_frontend_kwargs)

  def load_model(self) -> _VLLMModelServer:
    return _VLLMModelServer(
        self._model_name,
        self._vllm_server_kwargs,
        self._use_dynamo,
        self._dynamo_config)

  async def _async_run_inference(
      self,
      batch: Sequence[str],
      model: _VLLMModelServer,
      inference_args: Optional[dict[str, Any]] = None
  ) -> Iterable[PredictionResult]:
    inference_args = inference_args or {}

    async with getAsyncVLLMClient(model.get_server_port()) as client:
      try:
        async_predictions = [
            client.completions.create(
                model=self._model_name, prompt=prompt, **inference_args)
            for prompt in batch
        ]
        responses = await asyncio.gather(*async_predictions)
      except Exception as e:
        model.check_connectivity()
        raise e

    return [PredictionResult(x, y) for x, y in zip(batch, responses)]

  def run_inference(
      self,
      batch: Sequence[str],
      model: _VLLMModelServer,
      inference_args: Optional[dict[str, Any]] = None
  ) -> Iterable[PredictionResult]:
    """Runs inferences on a batch of text strings.

    Args:
      batch: A sequence of examples as text strings.
      model: A _VLLMModelServer containing info for connecting to the server.
      inference_args: Any additional arguments for an inference.

    Returns:
      An Iterable of type PredictionResult.
    """
    return asyncio.run(self._async_run_inference(batch, model, inference_args))

  def validate_inference_args(self, inference_args: Optional[dict[str, Any]]):
    # Override the base validator so OpenAI-compatible request kwargs such as
    # ``max_tokens`` can be passed through ``RunInference`` to the vLLM /
    # Dynamo server.
    pass

  def share_model_across_processes(self) -> bool:
    return True


class VLLMChatModelHandler(ModelHandler[Sequence[OpenAIChatMessage],
                                        PredictionResult,
                                        _VLLMModelServer]):
  def __init__(
      self,
      model_name: str,
      chat_template_path: Optional[str] = None,
      vllm_server_kwargs: Optional[dict[str, Optional[str]]] = None,
      *,
      use_dynamo: bool = False,
      dynamo_runtime_config: Optional[DynamoRuntimeConfig] = None,
      dynamo_frontend_kwargs: Optional[dict[str, Optional[str]]] = None,
      min_batch_size: Optional[int] = None,
      max_batch_size: Optional[int] = None,
      max_batch_duration_secs: Optional[int] = None,
      max_batch_weight: Optional[int] = None,
      element_size_fn: Optional[Callable[[Any], int]] = None,
      batch_length_fn: Optional[Callable[[Any], int]] = None,
      batch_bucket_boundaries: Optional[list[int]] = None):
    """ Implementation of the ModelHandler interface for vLLM using previous
    messages as input.

    Example Usage::

      pcoll | RunInference(VLLMModelHandler(model_name='facebook/opt-125m'))

    Args:
      model_name: The vLLM model. See
        https://docs.vllm.ai/en/latest/models/supported_models.html for
        supported models.
      chat_template_path: Path to a chat template. This file must be accessible
        from your runner's execution environment, so it is recommended to use
        a cloud based file storage system (e.g. Google Cloud Storage).
        For info on chat templates, see:
        https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-template
      vllm_server_kwargs: Any additional kwargs to be passed into your vllm
        server when it is being created. When ``use_dynamo`` is disabled,
        this is invoked using ``python -m vllm.entrypoints.openai.api_server
        <beam provided args> <vllm_server_kwargs>``. When ``use_dynamo`` is
        enabled, these kwargs are passed to the ``dynamo.vllm`` worker
        process. For example, you could pass ``{'echo': 'true'}`` to prepend
        new messages with the previous message. For a list of possible
        kwargs, see
        https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-api
      use_dynamo: Whether to use NVIDIA Dynamo as the underlying vLLM engine.
        Requires installing Dynamo in your runtime environment
        (``pip install ai-dynamo[vllm]``). By default this is an opt-in
        single-engine embedded mode; pass ``dynamo_runtime_config`` to run
        multiple GPU-pinned engines with KV-aware routing. Disaggregated
        prefill/decode, KVBM offload across nodes, the Planner, and Grove are
        not active in embedded mode. Dynamo also requires an etcd-style
        discovery service: when ``ETCD_ENDPOINTS`` is unset, Beam starts a
        local etcd, which requires the ``etcd`` binary in the worker
        environment.
      dynamo_runtime_config: A :class:`DynamoRuntimeConfig` describing the
        whole local Dynamo topology -- the engines and their GPU placement,
        the router mode (``round-robin`` or ``kv``), and how KV-cache state
        reaches the router (``disabled``/``approximate``/``zmq``). When
        omitted, ``use_dynamo=True`` maps to the legacy one-engine,
        round-robin, KV-events-disabled behaviour. Mutually exclusive with
        ``dynamo_frontend_kwargs``.
      dynamo_frontend_kwargs: Legacy escape hatch for extra
        ``dynamo.frontend`` flags in single-engine mode. Prefer
        ``dynamo_runtime_config`` for anything beyond ad-hoc frontend flags;
        the two may not be combined.
      min_batch_size: optional. the minimum batch size to use when batching
        inputs.
      max_batch_size: optional. the maximum batch size to use when batching
        inputs.
      max_batch_duration_secs: optional. the maximum amount of time to buffer
        a batch before emitting; used in streaming contexts.
      max_batch_weight: optional. the maximum total weight of a batch.
      element_size_fn: optional. a function that returns the size (weight) of
        an element.
      batch_length_fn: optional. a callable that returns the length of an
        element for length-aware batching.
      batch_bucket_boundaries: optional. a sorted list of positive boundary
        values for length-aware batching buckets.
    """
    super().__init__(
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        max_batch_duration_secs=max_batch_duration_secs,
        max_batch_weight=max_batch_weight,
        element_size_fn=element_size_fn,
        batch_length_fn=batch_length_fn,
        batch_bucket_boundaries=batch_bucket_boundaries)
    self._model_name = model_name
    self._vllm_server_kwargs: dict[str, Optional[str]] = dict(
        vllm_server_kwargs or {})
    self._dynamo_config = _resolve_dynamo_config(
        use_dynamo, dynamo_runtime_config, dynamo_frontend_kwargs)
    self._chat_template_path = chat_template_path
    self._chat_file = f'template-{uuid.uuid4().hex}.jinja'
    self._use_dynamo = use_dynamo

  def load_model(self) -> _VLLMModelServer:
    chat_template_contents = ''
    if self._chat_template_path is not None:
      local_chat_template_path = os.path.join(os.getcwd(), self._chat_file)
      if not os.path.exists(local_chat_template_path):
        with FileSystems.open(self._chat_template_path) as fin:
          chat_template_contents = fin.read().decode()
        with open(local_chat_template_path, 'a') as f:
          f.write(chat_template_contents)
      self._vllm_server_kwargs['chat_template'] = local_chat_template_path

    return _VLLMModelServer(
        self._model_name,
        self._vllm_server_kwargs,
        self._use_dynamo,
        self._dynamo_config)

  async def _async_run_inference(
      self,
      batch: Sequence[Sequence[OpenAIChatMessage]],
      model: _VLLMModelServer,
      inference_args: Optional[dict[str, Any]] = None
  ) -> Iterable[PredictionResult]:
    inference_args = inference_args or {}

    async with getAsyncVLLMClient(model.get_server_port()) as client:
      try:
        async_predictions = [
            client.chat.completions.create(
                model=self._model_name,
                messages=[{
                    "role": message.role, "content": message.content
                } for message in messages],
                **inference_args) for messages in batch
        ]
        predictions = await asyncio.gather(*async_predictions)
      except Exception as e:
        model.check_connectivity()
        raise e

    return [PredictionResult(x, y) for x, y in zip(batch, predictions)]

  def run_inference(
      self,
      batch: Sequence[Sequence[OpenAIChatMessage]],
      model: _VLLMModelServer,
      inference_args: Optional[dict[str, Any]] = None
  ) -> Iterable[PredictionResult]:
    """Runs inferences on a batch of text strings.

    Args:
      batch: A sequence of examples as OpenAI messages.
      model: A _VLLMModelServer for connecting to the spun up server.
      inference_args: Any additional arguments for an inference.

    Returns:
      An Iterable of type PredictionResult.
    """
    return asyncio.run(self._async_run_inference(batch, model, inference_args))

  def validate_inference_args(self, inference_args: Optional[dict[str, Any]]):
    # Override the base validator so OpenAI-compatible request kwargs such as
    # ``max_tokens`` can be passed through ``RunInference`` to the vLLM /
    # Dynamo server.
    pass

  def share_model_across_processes(self) -> bool:
    return True
