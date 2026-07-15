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

"""Internal lifecycle manager for embedded multi-engine NVIDIA Dynamo.

This module owns the *local* Dynamo topology that Beam launches inside a single
SDK worker process: one managed etcd (unless an external endpoint is supplied),
one ``dynamo.frontend`` router, and one or more GPU-pinned ``dynamo.vllm``
aggregated engines. It is intentionally private (underscore-prefixed) --
``vllm_inference`` exposes only the typed :class:`DynamoRuntimeConfig` /
:class:`DynamoVLLMEngineSpec` and delegates process supervision here so the
handlers do not duplicate topology logic.

The single-engine, round-robin, KV-events-disabled configuration reproduces the
original embedded Dynamo behaviour byte-for-byte. Multi-engine KV-aware routing
is opt-in through :class:`DynamoRuntimeConfig`.

References:
  * Canonical two-GPU aggregated router launcher (``agg_router.sh``):
    https://raw.githubusercontent.com/ai-dynamo/dynamo/main/examples/backends/vllm/launch/agg_router.sh
  * KV-aware routing / event transport modes:
    https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing
  * Local health checks (DYN_SYSTEM_PORT / endpoint health status):
    https://docs.nvidia.com/dynamo/latest/user-guides/observability-local/health-checks
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional

from apache_beam.utils import subprocess_server

_LOGGER = logging.getLogger(__name__)

# Router modes understood by ``dynamo.frontend --router-mode``.
_ROUTER_MODES = ('round-robin', 'kv')

# KV event transport modes. ``disabled`` keeps the router purely load-based;
# ``approximate`` uses ``--router-mode kv`` without a real event stream
# (predicted overlap only); ``zmq`` publishes real vLLM KV-cache events over a
# per-engine ZMQ endpoint so the router can combine prefix locality with load.
_KV_EVENT_MODES = ('disabled', 'approximate', 'zmq')

# Discovery/request/event plane defaults proven on the merged smoke test. These
# are the transports every engine and the frontend must agree on.
_PLANE_DEFAULTS: dict[str, Optional[str]] = {
    'discovery-backend': 'etcd',
    'request-plane': 'tcp',
    'event-plane': 'zmq',
}

# Dynamo parses DYN_SYSTEM_PORT as a signed 16-bit integer, so the per-engine
# system/health port must be <= 32767. OS-assigned ephemeral ports (what
# subprocess_server.pick_port returns) live in the 32768-60999 range on Linux
# and overflow, so engine system ports are drawn from the low i16 range below.
_MAX_I16_PORT = 32767
_SYSTEM_PORT_RANGE_START = 20000

# Frontend/engine keys whose values the runtime computes from the typed config.
# Passing them again through ``frontend_kwargs`` would let the caller silently
# contradict ``router_mode`` / ``kv_event_mode``, so we reject them up front.
_RUNTIME_OWNED_FRONTEND_KEYS = frozenset(
    {'router-mode', 'no-router-kv-events', 'http-port'})
_RUNTIME_OWNED_ENGINE_KEYS = frozenset({'kv-events-config'})


@dataclass(frozen=True)
class DynamoVLLMEngineSpec:
  """One logical ``dynamo.vllm`` aggregated engine.

  Args:
    gpu_devices: GPU ordinals this engine may use, e.g. ``('0',)`` for a single
      GPU or ``('0', '1')`` for a tensor/pipeline-parallel engine. Rendered into
      ``CUDA_VISIBLE_DEVICES`` for the engine process. May be empty only when
      the whole topology has exactly one engine (legacy single-engine mode),
      in which case the engine inherits the ambient GPU visibility.
    vllm_server_kwargs: Per-engine overrides layered on top of the handler-wide
      ``vllm_server_kwargs``. Use this to, for example, give two otherwise
      identical replicas different ``gpu-memory-utilization`` values.
  """
  gpu_devices: tuple[str, ...] = ()
  vllm_server_kwargs: Mapping[str, Optional[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamoRuntimeConfig:
  """Declarative description of one local embedded Dynamo topology.

  A single config owns the whole topology and its invariants: engine count and
  GPU placement, the router mode, and how KV-cache state reaches the router.
  ``VLLMCompletionsModelHandler`` / ``VLLMChatModelHandler`` cross into the
  runtime through this object instead of juggling several raw kwarg dicts.

  Args:
    engines: One or more :class:`DynamoVLLMEngineSpec`. Each entry is exactly
      one ``dynamo.vllm`` process. Two entries on GPUs ``0`` and ``1`` give the
      router a real placement choice; a single entry reproduces legacy embedded
      mode.
    router_mode: ``'round-robin'`` cycles workers; ``'kv'`` selects a worker
      from cache-prefix overlap and projected load.
    kv_event_mode: ``'disabled'`` (no KV events), ``'approximate'`` (KV router
      without a real event stream), or ``'zmq'`` (real per-engine KV-cache
      events; requires ``router_mode='kv'``).
    frontend_kwargs: Extra ``dynamo.frontend`` flags. May not contain keys the
      runtime owns (``router-mode``, ``no-router-kv-events``, ``http-port``).
  """
  engines: tuple[DynamoVLLMEngineSpec, ...]
  router_mode: str = 'round-robin'
  kv_event_mode: str = 'disabled'
  frontend_kwargs: Mapping[str, Optional[str]] = field(default_factory=dict)

  def validate(self) -> None:
    """Fail fast, before any GPU work or Dataflow launch, on a bad topology."""
    if not self.engines:
      raise ValueError('DynamoRuntimeConfig requires at least one engine.')
    if self.router_mode not in _ROUTER_MODES:
      raise ValueError(
          f'router_mode must be one of {_ROUTER_MODES}, got '
          f'{self.router_mode!r}.')
    if self.kv_event_mode not in _KV_EVENT_MODES:
      raise ValueError(
          f'kv_event_mode must be one of {_KV_EVENT_MODES}, got '
          f'{self.kv_event_mode!r}.')

    # Real/approximate KV events only make sense behind the KV router.
    if self.kv_event_mode != 'disabled' and self.router_mode != 'kv':
      raise ValueError(
          f'kv_event_mode={self.kv_event_mode!r} requires '
          "router_mode='kv'; round-robin cannot consume KV events.")

    owned = _RUNTIME_OWNED_FRONTEND_KEYS.intersection(self.frontend_kwargs)
    if owned:
      raise ValueError(
          f'frontend_kwargs may not set runtime-owned keys {sorted(owned)}; '
          'control these through router_mode / kv_event_mode instead.')

    multi_engine = len(self.engines) > 1
    seen_devices: dict[str, int] = {}
    for idx, engine in enumerate(self.engines):
      overlap = _RUNTIME_OWNED_ENGINE_KEYS.intersection(
          engine.vllm_server_kwargs)
      if overlap:
        raise ValueError(
            f'engine[{idx}] vllm_server_kwargs may not set runtime-owned keys '
            f'{sorted(overlap)}; control these through kv_event_mode.')

      if multi_engine and not engine.gpu_devices:
        raise ValueError(
            f'engine[{idx}] must declare gpu_devices explicitly in a '
            'multi-engine topology so placement is non-overlapping.')

      for device in engine.gpu_devices:
        if device in seen_devices:
          owner = seen_devices[device]
          raise ValueError(
              f'GPU {device!r} is assigned to both engine[{owner}] and '
              f'engine[{idx}]; engine GPU assignments must be disjoint.')
        seen_devices[device] = idx

      # A tensor/pipeline-parallel engine must be given exactly as many GPUs as
      # its parallel width. TP/PP is a vLLM feature, not a Dynamo benefit, but
      # if the caller asks for it the device math still has to add up.
      declared = len(engine.gpu_devices)
      if declared:
        tp = _int_kwarg(engine.vllm_server_kwargs, 'tensor-parallel-size', 1)
        pp = _int_kwarg(engine.vllm_server_kwargs, 'pipeline-parallel-size', 1)
        if tp * pp != declared:
          raise ValueError(
              f'engine[{idx}] declares {declared} GPU(s) but '
              f'tensor-parallel-size*pipeline-parallel-size={tp * pp}; they '
              'must match.')


def _int_kwarg(
    kwargs: Mapping[str, Optional[str]], key: str, default: int) -> int:
  value = kwargs.get(key)
  if value is None:
    return default
  try:
    return int(value)
  except (TypeError, ValueError):
    raise ValueError(f'{key} must be an integer, got {value!r}.')


def single_engine_config(
    frontend_kwargs: Optional[Mapping[str, Optional[str]]] = None
) -> DynamoRuntimeConfig:
  """Legacy embedded topology: one engine, round-robin, KV events disabled.

  ``frontend_kwargs`` are extra ``dynamo.frontend`` flags. To preserve the
  original escape-hatch behaviour (where a caller could pass ``router-mode``
  directly) any runtime-owned keys are stripped and applied to the typed fields
  instead of raising.
  """
  frontend_kwargs = dict(frontend_kwargs or {})
  router_mode = frontend_kwargs.pop('router-mode', 'round-robin')
  # ``no-router-kv-events`` is implied by the disabled event mode below; drop a
  # redundant explicit copy so validate() does not reject it.
  frontend_kwargs.pop('no-router-kv-events', None)
  return DynamoRuntimeConfig(
      engines=(DynamoVLLMEngineSpec(), ),
      router_mode=router_mode,
      kv_event_mode='disabled',
      frontend_kwargs=frontend_kwargs)


# Type of the injected process launcher: (cmd, port, env, label) -> (proc, port)
StartProcessFn = Callable[..., "tuple[subprocess.Popen, int]"]


@dataclass
class _ManagedProcess:
  """A supervised child process and everything needed to describe it in logs."""
  label: str
  cmd: list[str]
  process: Optional[subprocess.Popen] = None
  gpu_devices: tuple[str, ...] = ()
  http_port: Optional[int] = None
  system_port: Optional[int] = None
  event_port: Optional[int] = None

  def poll(self) -> Optional[int]:
    return None if self.process is None else self.process.poll()

  def metrics_url(self) -> Optional[str]:
    port = self.http_port if self.http_port is not None else self.system_port
    if port is None:
      return None
    return f'http://localhost:{port}/metrics'


def _http_status(url: str, timeout: float = 2.0) -> Optional[int]:
  """Return the HTTP status for ``url`` or ``None`` if it cannot be reached.

  Factored out so unit tests can patch a single seam instead of the network.
  """
  try:
    with urllib.request.urlopen(url, timeout=timeout) as response:
      return response.status
  except urllib.error.HTTPError as exc:
    # An HTTP error is still a *reachable* server (e.g. 503 not-ready).
    return exc.code
  except Exception:  # pylint: disable=broad-except
    return None


def _http_json(url: str, timeout: float = 2.0) -> Optional[Any]:
  try:
    with urllib.request.urlopen(url, timeout=timeout) as response:
      if response.status >= 400:
        return None
      return json.loads(response.read().decode('utf-8'))
  except Exception:  # pylint: disable=broad-except
    return None


def _pick_low_ports(
    count: int,
    low: int = _SYSTEM_PORT_RANGE_START,
    high: int = _MAX_I16_PORT) -> list[int]:
  """Return ``count`` distinct free TCP ports in ``[low, high]`` (<= i16 max).

  Used for ``DYN_SYSTEM_PORT``, which Dynamo parses as a signed 16-bit int, so
  the ephemeral ports ``subprocess_server.pick_port`` returns (>= 32768)
  overflow. Candidate ports are bound explicitly and the sockets are held open
  until all are chosen, so the returned ports are distinct and currently free
  (subject to the same close-then-reuse race as ``pick_port``).
  """
  sockets: list[socket.socket] = []
  ports: list[int] = []
  try:
    candidate = low
    while len(ports) < count and candidate <= high:
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      try:
        s.bind(('localhost', candidate))
      except OSError:
        s.close()
        candidate += 1
        continue
      sockets.append(s)
      ports.append(candidate)
      candidate += 1
    if len(ports) < count:
      raise RuntimeError(
          f'Could not find {count} free port(s) in [{low}, {high}] for '
          'DYN_SYSTEM_PORT.')
    return ports
  finally:
    for s in sockets:
      s.close()


class _DynamoLocalRuntime:
  """Owns one etcd + frontend + N GPU-pinned engine topology.

  Interface intentionally small: :meth:`start`, :meth:`wait_until_ready`,
  :meth:`is_healthy`, :meth:`status`, :meth:`metrics_endpoints`, :meth:`stop`.
  Port allocation, child environments, discovery, and cleanup are private.
  """
  def __init__(
      self,
      model_name: str,
      config: DynamoRuntimeConfig,
      common_vllm_kwargs: Mapping[str, Optional[str]],
      start_process_fn: StartProcessFn,
      append_kwargs_fn: Callable[[list[str], Mapping[str, Optional[str]]],
                                 None],
  ):
    config.validate()
    self._model_name = model_name
    self._config = config
    self._common_vllm_kwargs = dict(common_vllm_kwargs or {})
    self._start_process = start_process_fn
    self._append_kwargs = append_kwargs_fn

    self._frontend: Optional[_ManagedProcess] = None
    self._engines: list[_ManagedProcess] = []
    self._etcd_process: Optional[subprocess.Popen] = None
    self._etcd_data_dir: Optional[str] = None
    self._etcd_endpoint: Optional[str] = None
    self._manage_etcd = False
    self._frontend_port: int = -1

  # ---- lifecycle ---------------------------------------------------------

  def start(self) -> None:
    """Allocate ports, start etcd (if managed), the frontend, then engines.

    On any failure the partially-started topology is torn down before the
    error propagates, so a half-running set of engines never leaks GPU memory.
    """
    self.stop()  # idempotent: never start on top of a half-running topology
    try:
      self._resolve_etcd()
      ports = self._allocate_ports()
      self._build_frontend(ports['frontend_http'])
      self._build_engines(ports)

      if self._manage_etcd:
        self._start_etcd(ports['etcd_client'], ports['etcd_peer'])
        self._wait_for_etcd()

      self._launch(self._frontend, is_engine=False)
      for engine in self._engines:
        self._launch(engine, is_engine=True)
    except Exception:
      self.stop()
      raise

  def wait_until_ready(self, timeout_secs: int = 600) -> None:
    """Block until *every* engine and the frontend are serving.

    Unlike a plain ``/v1/models`` probe -- which can succeed as soon as the
    first of several engines registers, letting a two-engine run proceed at
    half capacity -- this waits for each engine's system health endpoint and
    then confirms the frontend exposes the model.
    """
    deadline = time.time() + timeout_secs
    pending = list(self._engines)
    while pending and time.time() < deadline:
      if not self._required_children_alive():
        raise RuntimeError(
            'Dynamo topology exited during startup. ' + self.status())
      still_pending = []
      for engine in pending:
        status = _http_status(f'http://localhost:{engine.system_port}/health')
        # 200 means the configured (generate) endpoint is healthy; a reachable
        # but not-ready worker (e.g. 503) keeps us waiting.
        if status == 200:
          _LOGGER.info('Dynamo engine ready: %s', engine.label)
        else:
          still_pending.append(engine)
      pending = still_pending
      if pending:
        time.sleep(3)

    if pending:
      raise RuntimeError(
          f'Timed out waiting for {len(pending)} Dynamo engine(s) to become '
          f'healthy: {[e.label for e in pending]}. ' + self.status())

    self._wait_for_frontend_models(deadline)

  def _wait_for_frontend_models(self, deadline: float) -> None:
    while time.time() < deadline:
      if not self._required_children_alive():
        raise RuntimeError(
            'Dynamo topology exited during startup. ' + self.status())
      body = _http_json(f'http://localhost:{self._frontend_port}/v1/models')
      data = (body or {}).get('data') if isinstance(body, dict) else None
      if data:
        _LOGGER.info(
            'Dynamo frontend serving %d model(s) across %d engine(s).',
            len(data),
            len(self._engines))
        return
      time.sleep(3)
    raise RuntimeError(
        'Timed out waiting for the Dynamo frontend to expose the model. ' +
        self.status())

  def is_healthy(self) -> bool:
    return self._required_children_alive()

  def status(self) -> str:
    parts = []
    if self._etcd_process is not None:
      parts.append('etcd exit code: %s' % self._etcd_process.poll())
    if self._frontend is not None:
      parts.append('frontend exit code: %s' % self._frontend.poll())
    for idx, engine in enumerate(self._engines):
      parts.append(
          'engine[%d] gpu=%s exit code: %s' %
          (idx, ','.join(engine.gpu_devices) or 'inherit', engine.poll()))
    return 'Process status: ' + (', '.join(parts) or 'no processes')

  def metrics_endpoints(self) -> dict[str, str]:
    """Local Prometheus ``/metrics`` URLs for the frontend and each engine."""
    endpoints: dict[str, str] = {}
    if self._frontend is not None:
      url = self._frontend.metrics_url()
      if url:
        endpoints['frontend'] = url
    for idx, engine in enumerate(self._engines):
      url = engine.metrics_url()
      if url:
        endpoints[f'engine-{idx}'] = url
    return endpoints

  @property
  def frontend_port(self) -> int:
    return self._frontend_port

  def stop(self) -> None:
    """Terminate every child process group, then remove managed etcd state."""
    for engine in self._engines:
      _stop_process(engine.process)
    if self._frontend is not None:
      _stop_process(self._frontend.process)
    _stop_process(self._etcd_process)
    if self._etcd_data_dir is not None:
      shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
    self._frontend = None
    self._engines = []
    self._etcd_process = None
    self._etcd_data_dir = None
    self._etcd_endpoint = None
    self._manage_etcd = False
    self._frontend_port = -1

  # ---- construction helpers ---------------------------------------------

  def _resolve_etcd(self) -> None:
    # The frontend's effective discovery backend decides whether we need etcd.
    discovery = dict(_PLANE_DEFAULTS)
    discovery.update(self._config.frontend_kwargs)
    uses_etcd = discovery.get('discovery-backend') == 'etcd'
    external = os.environ.get('ETCD_ENDPOINTS')
    if external:
      # Respect an externally supplied endpoint: never start or delete it.
      self._etcd_endpoint = external
      self._manage_etcd = False
    else:
      self._manage_etcd = uses_etcd

  def _allocate_ports(self) -> dict[str, int]:
    num_engines = len(self._config.engines)
    publish_events = self._config.kv_event_mode == 'zmq'

    # Engine system ports must fit in an i16 (<= 32767), so they are drawn from
    # the low range rather than the ephemeral range pick_port returns.
    system_ports = _pick_low_ports(num_engines)

    # Everything else is an ordinary u16 port (frontend HTTP, etcd client/peer,
    # ZMQ KV-event publishers). Allocate them in ONE pick_port call so the
    # returned ports are guaranteed distinct (pick_port holds each socket open
    # until all are chosen). These are always >= 32768, disjoint from the low
    # system ports above.
    ephemeral_slots = ['frontend_http']
    if self._manage_etcd:
      ephemeral_slots += ['etcd_client', 'etcd_peer']
    for idx in range(num_engines):
      if publish_events:
        ephemeral_slots.append(f'engine_{idx}_event')

    ephemeral = subprocess_server.pick_port(*([None] * len(ephemeral_slots)))
    ports = dict(zip(ephemeral_slots, ephemeral))
    for idx in range(num_engines):
      ports[f'engine_{idx}_system'] = system_ports[idx]

    if len(set(ports.values())) != len(ports):
      raise RuntimeError(f'Port allocation produced duplicates: {ports}')
    return ports

  def _build_frontend(self, http_port: int) -> None:
    cmd = [
        sys.executable,
        '-m',
        'dynamo.frontend',
        '--http-port',
        str(http_port),
    ]
    kwargs: dict[str, Optional[str]] = dict(_PLANE_DEFAULTS)
    kwargs['router-mode'] = self._config.router_mode
    # ``--no-router-kv-events`` is correct for round-robin and for the
    # approximate KV router; a real ZMQ event stream must NOT set it.
    if self._config.kv_event_mode != 'zmq':
      kwargs['no-router-kv-events'] = None
    # Caller extras win last (validate() already blocked runtime-owned keys).
    kwargs.update(self._config.frontend_kwargs)
    self._append_kwargs(cmd, kwargs)
    self._frontend = _ManagedProcess(
        label='dynamo.frontend', cmd=cmd, http_port=http_port)
    self._frontend_port = http_port

  def _build_engines(self, ports: dict[str, int]) -> None:
    self._engines = []
    for idx, spec in enumerate(self._config.engines):
      system_port = ports[f'engine_{idx}_system']
      event_port = ports.get(f'engine_{idx}_event')
      cmd = [
          sys.executable,
          '-m',
          'dynamo.vllm',
          '--model',
          self._model_name,
      ]
      kwargs: dict[str, Optional[str]] = dict(_PLANE_DEFAULTS)
      kwargs.update(self._common_vllm_kwargs)
      kwargs.update(spec.vllm_server_kwargs)
      kwargs['kv-events-config'] = self._kv_events_config(event_port)
      if self._config.kv_event_mode == 'zmq':
        # Prefix caching is what produces reusable KV blocks for the router to
        # exploit; without it a KV-aware decision has nothing to route on.
        kwargs.setdefault('enable-prefix-caching', None)
      self._append_kwargs(cmd, kwargs)
      self._engines.append(
          _ManagedProcess(
              label=f'dynamo.vllm[{idx}]',
              cmd=cmd,
              gpu_devices=tuple(spec.gpu_devices),
              system_port=system_port,
              event_port=event_port))

  def _kv_events_config(self, event_port: Optional[int]) -> str:
    if self._config.kv_event_mode == 'zmq':
      assert event_port is not None
      return json.dumps({
          'publisher': 'zmq',
          'topic': 'kv-events',
          'endpoint': f'tcp://*:{event_port}',
          'enable_kv_cache_events': True,
      })
    # Both 'disabled' and 'approximate' run the engines without a real event
    # stream; the difference lives entirely in the frontend's router mode.
    return json.dumps({'enable_kv_cache_events': False})

  # ---- process control ---------------------------------------------------

  def _child_env(self, engine: Optional[_ManagedProcess]) -> dict[str, str]:
    """A per-child environment, never a mutation of the SDK process env."""
    env = dict(os.environ)
    if self._etcd_endpoint is not None:
      env['ETCD_ENDPOINTS'] = self._etcd_endpoint
    # Prefix hashes must agree between the frontend and every worker for
    # KV-aware routing; the canonical launcher pins PYTHONHASHSEED for this.
    if self._config.router_mode == 'kv':
      env['PYTHONHASHSEED'] = '0'
    if engine is not None:
      if engine.gpu_devices:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(engine.gpu_devices)
      if engine.system_port is not None:
        # Distinct per engine so the two system/health servers do not collide.
        # Must be <= 32767 (Dynamo parses it as i16); see _pick_low_ports.
        env['DYN_SYSTEM_PORT'] = str(engine.system_port)
    return env

  def _launch(self, managed: _ManagedProcess, is_engine: bool) -> None:
    env = self._child_env(managed if is_engine else None)
    port = managed.http_port if managed.http_port is not None else \
        managed.system_port
    process, _ = self._start_process(
        managed.cmd, port=port, env=env, component_label=managed.label)
    managed.process = process

  def _start_etcd(self, client_port: int, peer_port: int) -> None:
    if shutil.which('etcd') is None:
      raise RuntimeError(
          'Embedded Dynamo mode requires etcd when ETCD_ENDPOINTS is not set. '
          'Install etcd in the worker container or set ETCD_ENDPOINTS to an '
          'external etcd service.')
    etcd_name = f'beam-dynamo-etcd-{uuid.uuid4().hex}'
    self._etcd_data_dir = f'/tmp/{etcd_name}'
    self._etcd_endpoint = f'http://127.0.0.1:{client_port}'
    etcd_cmd = [
        'etcd',
        '--name',
        etcd_name,
        '--listen-client-urls',
        f'http://127.0.0.1:{client_port}',
        '--advertise-client-urls',
        f'http://127.0.0.1:{client_port}',
        '--listen-peer-urls',
        f'http://127.0.0.1:{peer_port}',
        '--initial-advertise-peer-urls',
        f'http://127.0.0.1:{peer_port}',
        '--initial-cluster',
        f'{etcd_name}=http://127.0.0.1:{peer_port}',
        '--data-dir',
        self._etcd_data_dir,
        '--log-level',
        'warn',
    ]
    self._etcd_process, _ = self._start_process(
        etcd_cmd, port=client_port, env=dict(os.environ),
        component_label='etcd')

  def _wait_for_etcd(self, timeout_secs: int = 30) -> None:
    deadline = time.time() + timeout_secs
    health_url = self._etcd_endpoint.rstrip('/') + '/health'
    while time.time() < deadline and self._etcd_process.poll() is None:
      status = _http_status(health_url)
      if status is not None and status < 500:
        return
      time.sleep(1)
    status_str = self.status()
    self.stop()
    raise RuntimeError(
        'Failed to start embedded etcd for Dynamo. ' + status_str +
        '. Install etcd in the worker container or set ETCD_ENDPOINTS to an '
        'external etcd service.')

  def _required_children_alive(self) -> bool:
    if self._frontend is None or self._frontend.poll() is not None:
      return False
    if self._etcd_process is not None and self._etcd_process.poll() is not None:
      return False
    return all(engine.poll() is None for engine in self._engines)


def _stop_process(process: Optional[subprocess.Popen]) -> None:
  """Terminate a child *process group*, escalating TERM -> KILL on timeout.

  Engines are started in their own session (``start_new_session=True``) so vLLM
  and its CUDA descendants share one process group; killing the group prevents
  orphaned workers from holding GPU memory across a restart.
  """
  if process is None or process.poll() is not None:
    return
  try:
    _signal_group(process, signal.SIGTERM)
    try:
      process.wait(timeout=10)
    except subprocess.TimeoutExpired:
      _signal_group(process, signal.SIGKILL)
      process.wait()
  except OSError:
    # The process may exit between poll() and signalling; treat as stopped.
    pass


def _signal_group(process: subprocess.Popen, sig: int) -> None:
  # Prefer a process-group signal (POSIX). Fall back to the single process if
  # the platform lacks killpg or the group is already gone.
  try:
    os.killpg(os.getpgid(process.pid), sig)
    return
  except (AttributeError, ProcessLookupError, PermissionError, OSError):
    pass
  if sig == signal.SIGKILL:
    process.kill()
  else:
    process.terminate()
