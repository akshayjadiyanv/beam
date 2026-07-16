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

"""Unit tests for the embedded multi-engine Dynamo runtime (no GPU required).

These exercise topology construction (commands, per-child environments, port
allocation), validation, phased readiness, health, and cleanup entirely through
a fake process launcher so no real etcd / frontend / vLLM process is started.
"""

import json
import os
import re
import unittest
from unittest import mock

from apache_beam.ml.inference import _dynamo_runtime as dr


def _append_kwargs(cmd, kwargs):
  for k, v in kwargs.items():
    cmd.append(f'--{k}')
    if v is not None:
      cmd.append(v)


class _FakeProcess:
  def __init__(self):
    self.returncode = None
    self.pid = 424242
    self.terminated = False

  def poll(self):
    return self.returncode

  def terminate(self):
    self.terminated = True
    self.returncode = 0

  def kill(self):
    self.returncode = -9

  def wait(self, timeout=None):
    return self.returncode


class _Recorder:
  """A fake ``start_process`` that records every launch."""
  def __init__(self):
    self.calls = []

  def __call__(self, cmd, port=None, env=None, component_label=None):
    proc = _FakeProcess()
    self.calls.append({
        'cmd': list(cmd),
        'port': port,
        'env': dict(env) if env is not None else None,
        'label': component_label,
        'process': proc,
    })
    return proc, (port if port is not None else 9999)

  def by_label(self, prefix):
    return [c for c in self.calls if c['label'].startswith(prefix)]

  def frontend(self):
    return self.by_label('dynamo.frontend')[0]

  def engines(self):
    return self.by_label('dynamo.vllm')


def _build_runtime(config, common=None):
  return dr._DynamoLocalRuntime(
      'test-model', config, common or {}, _Recorder(), _append_kwargs)


def _kv_config_arg(cmd):
  idx = cmd.index('--kv-events-config')
  return json.loads(cmd[idx + 1])


def _start_with_external_etcd(config, common=None):
  """Start a topology with an external etcd endpoint (so no etcd is launched)
  and readiness stubbed to succeed. Returns (runtime, recorder)."""
  recorder = _Recorder()
  runtime = dr._DynamoLocalRuntime(
      'test-model', config, common or {}, recorder, _append_kwargs)
  with mock.patch.dict(os.environ, {'ETCD_ENDPOINTS': 'http://ext-etcd:2379'}):
    runtime.start()
  return runtime, recorder


class DynamoRuntimeConfigValidationTest(unittest.TestCase):
  def _spec(self, *devices):
    return dr.DynamoVLLMEngineSpec(gpu_devices=tuple(devices))

  def test_requires_at_least_one_engine(self):
    with self.assertRaisesRegex(ValueError, 'at least one engine'):
      dr.DynamoRuntimeConfig(engines=()).validate()

  def test_multi_engine_requires_explicit_gpus(self):
    with self.assertRaisesRegex(ValueError, 'gpu_devices'):
      dr.DynamoRuntimeConfig(
          engines=(dr.DynamoVLLMEngineSpec(), self._spec('1'))).validate()

  def test_overlapping_gpu_assignment_rejected(self):
    with self.assertRaisesRegex(ValueError, 'disjoint'):
      dr.DynamoRuntimeConfig(engines=(self._spec('0'),
                                      self._spec('0', '1'))).validate()

  def test_zmq_events_require_kv_router(self):
    with self.assertRaisesRegex(ValueError, "router_mode='kv'"):
      dr.DynamoRuntimeConfig(
          engines=(self._spec('0'), ),
          router_mode='round-robin',
          kv_event_mode='zmq').validate()

  def test_approximate_events_require_kv_router(self):
    with self.assertRaisesRegex(ValueError, "router_mode='kv'"):
      dr.DynamoRuntimeConfig(
          engines=(self._spec('0'), ),
          router_mode='round-robin',
          kv_event_mode='approximate').validate()

  def test_parallel_width_must_match_device_count(self):
    with self.assertRaisesRegex(ValueError, 'must match'):
      dr.DynamoRuntimeConfig(
          engines=(
              dr.DynamoVLLMEngineSpec(
                  gpu_devices=('0', '1'),
                  vllm_server_kwargs={'tensor-parallel-size': '1'}),
              self._spec('2'),
          )).validate()

  def test_tp_pp_product_matches_device_count(self):
    # 2 GPUs with TP2 x PP1 is valid.
    dr.DynamoRuntimeConfig(
        engines=(
            dr.DynamoVLLMEngineSpec(
                gpu_devices=('0', '1'),
                vllm_server_kwargs={'tensor-parallel-size': '2'}),
        )).validate()

  def test_frontend_owned_key_rejected(self):
    for key in ('router-mode', 'no-router-kv-events', 'http-port'):
      with self.assertRaisesRegex(ValueError, 'runtime-owned'):
        dr.DynamoRuntimeConfig(
            engines=(self._spec('0'), ), frontend_kwargs={
                key: 'x'
            }).validate()

  def test_engine_owned_key_rejected(self):
    with self.assertRaisesRegex(ValueError, 'runtime-owned'):
      dr.DynamoRuntimeConfig(
          engines=(
              dr.DynamoVLLMEngineSpec(
                  gpu_devices=('0', ),
                  vllm_server_kwargs={'kv-events-config': '{}'}), )).validate()

  def test_single_engine_config_is_legacy_shape(self):
    config = dr.single_engine_config({'router-mode': 'round-robin'})
    self.assertEqual(len(config.engines), 1)
    self.assertEqual(config.router_mode, 'round-robin')
    self.assertEqual(config.kv_event_mode, 'disabled')
    config.validate()


class DynamoRuntimeTopologyTest(unittest.TestCase):
  def _two_engine_kv_config(self):
    return dr.DynamoRuntimeConfig(
        engines=(
            dr.DynamoVLLMEngineSpec(gpu_devices=('0', )),
            dr.DynamoVLLMEngineSpec(gpu_devices=('1', )),
        ),
        router_mode='kv',
        kv_event_mode='zmq')

  def test_two_engines_produce_one_frontend_and_two_engines(self):
    _, rec = _start_with_external_etcd(self._two_engine_kv_config())
    self.assertEqual(len(rec.by_label('dynamo.frontend')), 1)
    self.assertEqual(len(rec.engines()), 2)
    self.assertNotIn('etcd', [c['label'] for c in rec.calls])

  def test_cuda_visible_devices_assigned_per_engine(self):
    _, rec = _start_with_external_etcd(self._two_engine_kv_config())
    engines = rec.engines()
    self.assertEqual(engines[0]['env']['CUDA_VISIBLE_DEVICES'], '0')
    self.assertEqual(engines[1]['env']['CUDA_VISIBLE_DEVICES'], '1')

  def test_unique_system_and_event_ports(self):
    _, rec = _start_with_external_etcd(self._two_engine_kv_config())
    engines = rec.engines()
    system_ports = [int(e['env']['DYN_SYSTEM_PORT']) for e in engines]
    event_ports = [
        _kv_config_arg(e['cmd'])['endpoint'].rsplit(':', 1)[1] for e in engines
    ]
    frontend_port = rec.frontend()['port']
    all_ports = system_ports + [int(p) for p in event_ports] + [frontend_port]
    self.assertEqual(len(all_ports), len(set(all_ports)))

  def test_system_ports_fit_in_signed_int16(self):
    # DYN_SYSTEM_PORT is parsed as an i16 by Dynamo; ephemeral ports (>= 32768)
    # overflow and crash the worker, so every engine system port must be <=
    # 32767. No engine may inherit an ephemeral DYN_SYSTEM_PORT.
    _, rec = _start_with_external_etcd(self._two_engine_kv_config())
    system_ports = [int(e['env']['DYN_SYSTEM_PORT']) for e in rec.engines()]
    self.assertEqual(len(system_ports), 2)
    for port in system_ports:
      self.assertLessEqual(port, dr._MAX_I16_PORT)
      self.assertGreater(port, 0)

  def test_deprecated_health_status_env_not_set(self):
    _, rec = _start_with_external_etcd(self._two_engine_kv_config())
    for engine in rec.engines():
      self.assertNotIn('DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS', engine['env'])

  def test_pick_low_ports_distinct_and_in_range(self):
    ports = dr._pick_low_ports(4)
    self.assertEqual(len(ports), 4)
    self.assertEqual(len(set(ports)), 4)
    for port in ports:
      self.assertLessEqual(port, dr._MAX_I16_PORT)

  def test_real_kv_mode_enables_events_and_drops_no_router_flag(self):
    _, rec = _start_with_external_etcd(self._two_engine_kv_config())
    frontend = rec.frontend()['cmd']
    self.assertIn('--router-mode', frontend)
    self.assertIn('kv', frontend)
    self.assertNotIn('--no-router-kv-events', frontend)
    for engine in rec.engines():
      cfg = _kv_config_arg(engine['cmd'])
      self.assertTrue(cfg['enable_kv_cache_events'])
      self.assertEqual(cfg['publisher'], 'zmq')
      self.assertTrue(re.match(r'tcp://\*:\d+', cfg['endpoint']))
      self.assertIn('--enable-prefix-caching', engine['cmd'])
      self.assertEqual(engine['env']['PYTHONHASHSEED'], '0')
    self.assertEqual(rec.frontend()['env']['PYTHONHASHSEED'], '0')

  def test_approximate_kv_mode_keeps_no_router_flag_events_off(self):
    config = dr.DynamoRuntimeConfig(
        engines=(
            dr.DynamoVLLMEngineSpec(gpu_devices=('0', )),
            dr.DynamoVLLMEngineSpec(gpu_devices=('1', )),
        ),
        router_mode='kv',
        kv_event_mode='approximate')
    _, rec = _start_with_external_etcd(config)
    self.assertIn('--no-router-kv-events', rec.frontend()['cmd'])
    for engine in rec.engines():
      self.assertFalse(_kv_config_arg(engine['cmd'])['enable_kv_cache_events'])

  def test_round_robin_mode_disables_events_and_hashseed(self):
    config = dr.DynamoRuntimeConfig(
        engines=(
            dr.DynamoVLLMEngineSpec(gpu_devices=('0', )),
            dr.DynamoVLLMEngineSpec(gpu_devices=('1', )),
        ),
        router_mode='round-robin',
        kv_event_mode='disabled')
    _, rec = _start_with_external_etcd(config)
    self.assertIn('--no-router-kv-events', rec.frontend()['cmd'])
    for engine in rec.engines():
      self.assertFalse(_kv_config_arg(engine['cmd'])['enable_kv_cache_events'])
      self.assertNotIn('PYTHONHASHSEED', engine['env'])
    self.assertNotIn('PYTHONHASHSEED', rec.frontend()['env'])

  def test_common_kwargs_applied_to_every_engine(self):
    _, rec = _start_with_external_etcd(
        self._two_engine_kv_config(),
        common={
            'revision': 'abc123', 'max-num-seqs': '32'
        })
    for engine in rec.engines():
      self.assertIn('--revision', engine['cmd'])
      self.assertIn('abc123', engine['cmd'])
      self.assertIn('--max-num-seqs', engine['cmd'])

  def test_per_engine_overrides_layer_on_common(self):
    config = dr.DynamoRuntimeConfig(
        engines=(
            dr.DynamoVLLMEngineSpec(
                gpu_devices=('0', ),
                vllm_server_kwargs={'gpu-memory-utilization': '0.9'}),
            dr.DynamoVLLMEngineSpec(gpu_devices=('1', )),
        ),
        router_mode='kv',
        kv_event_mode='zmq')
    _, rec = _start_with_external_etcd(
        config, common={'gpu-memory-utilization': '0.5'})
    e0 = rec.engines()[0]['cmd']
    e1 = rec.engines()[1]['cmd']
    self.assertEqual(e0[e0.index('--gpu-memory-utilization') + 1], '0.9')
    self.assertEqual(e1[e1.index('--gpu-memory-utilization') + 1], '0.5')

  def test_external_etcd_endpoint_injected_not_started(self):
    runtime, rec = _start_with_external_etcd(self._two_engine_kv_config())
    self.assertNotIn('etcd', [c['label'] for c in rec.calls])
    for call in rec.calls:
      self.assertEqual(call['env']['ETCD_ENDPOINTS'], 'http://ext-etcd:2379')
    # Stopping must not touch a caller-owned etcd endpoint in os.environ.
    with mock.patch.dict(os.environ,
                         {'ETCD_ENDPOINTS': 'http://ext-etcd:2379'}):
      with mock.patch.object(dr, '_stop_process'):
        runtime.stop()
      self.assertEqual(os.environ['ETCD_ENDPOINTS'], 'http://ext-etcd:2379')

  def test_managed_etcd_started_when_no_external_endpoint(self):
    recorder = _Recorder()
    runtime = dr._DynamoLocalRuntime(
        'test-model',
        self._two_engine_kv_config(), {},
        recorder,
        _append_kwargs)
    env_without_etcd = {
        k: v
        for k, v in os.environ.items() if k != 'ETCD_ENDPOINTS'
    }
    with mock.patch.dict(os.environ, env_without_etcd, clear=True):
      with mock.patch.object(dr.shutil, 'which', return_value='/usr/bin/etcd'):
        with mock.patch.object(dr, '_http_status', return_value=200):
          runtime.start()
    labels = [c['label'] for c in recorder.calls]
    self.assertIn('etcd', labels)
    # Every child (frontend + engines) points at the managed endpoint.
    managed = runtime._etcd_endpoint
    self.assertTrue(managed.startswith('http://127.0.0.1:'))
    for call in recorder.by_label('dynamo.vllm'):
      self.assertEqual(call['env']['ETCD_ENDPOINTS'], managed)

  def test_managed_etcd_missing_binary_raises(self):
    recorder = _Recorder()
    runtime = dr._DynamoLocalRuntime(
        'test-model',
        self._two_engine_kv_config(), {},
        recorder,
        _append_kwargs)
    env_without_etcd = {
        k: v
        for k, v in os.environ.items() if k != 'ETCD_ENDPOINTS'
    }
    with mock.patch.dict(os.environ, env_without_etcd, clear=True):
      with mock.patch.object(dr.shutil, 'which', return_value=None):
        with self.assertRaisesRegex(RuntimeError, 'requires etcd'):
          runtime.start()


class DynamoRuntimeReadinessTest(unittest.TestCase):
  def _config(self):
    return dr.DynamoRuntimeConfig(
        engines=(
            dr.DynamoVLLMEngineSpec(gpu_devices=('0', )),
            dr.DynamoVLLMEngineSpec(gpu_devices=('1', )),
        ),
        router_mode='kv',
        kv_event_mode='zmq')

  def test_ready_when_all_engines_and_frontend_healthy(self):
    runtime, _ = _start_with_external_etcd(self._config())
    with mock.patch.object(dr, '_http_status', return_value=200):
      with mock.patch.object(dr, '_http_json', return_value={'data': [{}]}):
        runtime.wait_until_ready(timeout_secs=5)  # should return promptly

  def test_readiness_waits_for_all_engines_not_first(self):
    runtime, rec = _start_with_external_etcd(self._config())
    engines = rec.engines()
    ready_port = int(engines[0]['env']['DYN_SYSTEM_PORT'])

    def only_first_ready(url, timeout=2.0):
      # Only engine[0]'s health endpoint returns ready; engine[1] stays 503.
      return 200 if f':{ready_port}/' in url else 503

    with mock.patch.object(dr, '_http_status', side_effect=only_first_ready):
      with mock.patch.object(dr, '_http_json', return_value={'data': [{}]}):
        with self.assertRaisesRegex(RuntimeError, 'engine'):
          runtime.wait_until_ready(timeout_secs=1)

  def test_readiness_fails_when_child_exits(self):
    runtime, rec = _start_with_external_etcd(self._config())
    rec.engines()[1]['process'].returncode = 1  # engine died
    with mock.patch.object(dr, '_http_status', return_value=200):
      with self.assertRaisesRegex(RuntimeError, 'exited during startup'):
        runtime.wait_until_ready(timeout_secs=5)


class DynamoRuntimeHealthAndCleanupTest(unittest.TestCase):
  def _config(self):
    return dr.DynamoRuntimeConfig(
        engines=(
            dr.DynamoVLLMEngineSpec(gpu_devices=('0', )),
            dr.DynamoVLLMEngineSpec(gpu_devices=('1', )),
        ),
        router_mode='round-robin')

  def test_is_healthy_true_when_all_alive(self):
    runtime, _ = _start_with_external_etcd(self._config())
    self.assertTrue(runtime.is_healthy())

  def test_is_healthy_false_and_status_reports_exit_code(self):
    runtime, rec = _start_with_external_etcd(self._config())
    rec.engines()[0]['process'].returncode = 137
    self.assertFalse(runtime.is_healthy())
    self.assertIn('137', runtime.status())

  def test_stop_terminates_every_child(self):
    runtime, rec = _start_with_external_etcd(self._config())
    stopped = []
    with mock.patch.object(dr, '_stop_process', side_effect=stopped.append):
      runtime.stop()
    launched = {id(c['process']) for c in rec.calls}
    # Every launched process was handed to _stop_process (frontend + engines).
    self.assertTrue({id(p) for p in stopped if p is not None} >= launched)
    # Runtime state is reset so a subsequent start does not see stale children.
    self.assertEqual(runtime._engines, [])
    self.assertEqual(runtime.frontend_port, -1)

  def test_metrics_endpoints_cover_frontend_and_each_engine(self):
    runtime, _ = _start_with_external_etcd(self._config())
    endpoints = runtime.metrics_endpoints()
    self.assertIn('frontend', endpoints)
    self.assertIn('engine-0', endpoints)
    self.assertIn('engine-1', endpoints)
    for url in endpoints.values():
      self.assertTrue(url.endswith('/metrics'))

  def test_start_cleans_up_on_launch_failure(self):
    # If a child fails to launch, the partially-started topology is stopped.
    boom = _Recorder()
    calls = {'n': 0}

    def flaky(cmd, port=None, env=None, component_label=None):
      calls['n'] += 1
      if calls['n'] == 2:  # fail launching the first engine
        raise RuntimeError('spawn failed')
      return boom(cmd, port=port, env=env, component_label=component_label)

    runtime = dr._DynamoLocalRuntime(
        'test-model', self._config(), {}, flaky, _append_kwargs)
    with mock.patch.dict(os.environ,
                         {'ETCD_ENDPOINTS': 'http://ext-etcd:2379'}):
      with mock.patch.object(dr, '_stop_process') as stop_mock:
        with self.assertRaises(RuntimeError):
          runtime.start()
        self.assertTrue(stop_mock.called)


if __name__ == '__main__':
  unittest.main()
