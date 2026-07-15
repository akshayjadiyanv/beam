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
import contextlib
import os
import sys
import types
import unittest
from unittest import mock

# Protect against environments where the OpenAI python library is not
# available. The command-construction tests below do not actually need a
# real OpenAI client; stubbing the module is enough for vllm_inference to
# import cleanly.
# pylint: disable=wrong-import-order, wrong-import-position
try:
  import openai  # pylint: disable=unused-import
except ImportError:
  openai = types.ModuleType('openai')

  class _FakeOpenAI:
    pass

  openai.AsyncOpenAI = _FakeOpenAI
  openai.OpenAI = _FakeOpenAI
  sys.modules['openai'] = openai

from apache_beam.ml.inference import _dynamo_runtime
from apache_beam.ml.inference import vllm_inference
from apache_beam.ml.inference._dynamo_runtime import DynamoRuntimeConfig
from apache_beam.ml.inference._dynamo_runtime import DynamoVLLMEngineSpec


class _FakeProcess:
  def __init__(self):
    self.returncode = None
    self.pid = 424242

  def poll(self):
    return self.returncode

  def terminate(self):
    self.returncode = 0

  def wait(self, timeout=None):
    return self.returncode

  def kill(self):
    self.returncode = -9


class _FakeModels:
  def list(self):
    return types.SimpleNamespace(data=[object()])


class _FakeClient:
  def __init__(self):
    self.models = _FakeModels()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    return False


def _record_start_process(commands):
  def start_process(cmd, port=None, env=None, component_label=None):
    commands.append({
        'cmd': list(cmd),
        'env': dict(env) if env is not None else None,
        'label': component_label,
    })
    return _FakeProcess(), (port if port is not None else 10000 + len(commands))

  return start_process


@contextlib.contextmanager
def _dynamo_topology(commands):
  """Patch the process launcher and Dynamo readiness so a dynamo ``load_model``
  runs to completion without any real subprocess or network."""
  with contextlib.ExitStack() as stack:
    stack.enter_context(
        mock.patch.object(
            vllm_inference, 'start_process', _record_start_process(commands)))
    stack.enter_context(
        mock.patch.object(_dynamo_runtime, '_http_status', return_value=200))
    stack.enter_context(
        mock.patch.object(
            _dynamo_runtime, '_http_json', return_value={'data': [{}]}))
    yield


def _labels(commands):
  return [c['label'] for c in commands]


def _cmd_with(commands, needle):
  for c in commands:
    if needle in c['cmd']:
      return c
  raise AssertionError(f'no command contained {needle!r}: {_labels(commands)}')


class VLLMInferenceTest(unittest.TestCase):
  def test_native_vllm_starts_single_server_process(self):
    commands = []
    with mock.patch.object(vllm_inference,
                           'start_process',
                           _record_start_process(commands)):
      with mock.patch.object(vllm_inference, 'getVLLMClient'):
        vllm_inference.getVLLMClient.return_value = _FakeClient()
        vllm_inference.VLLMCompletionsModelHandler(
            model_name='test-model',
            vllm_server_kwargs={
                'gpu-memory-utilization': '0.9'
            }).load_model()
    self.assertEqual(1, len(commands))
    cmd = commands[0]['cmd']
    self.assertIn('vllm.entrypoints.openai.api_server', cmd)
    self.assertIn('--model', cmd)
    self.assertIn('test-model', cmd)
    self.assertIn('--gpu-memory-utilization', cmd)
    self.assertIn('0.9', cmd)
    self.assertNotIn('dynamo.frontend', cmd)
    self.assertNotIn('dynamo.vllm', cmd)

  def test_legacy_dynamo_starts_frontend_and_single_engine(self):
    commands = []
    with mock.patch.dict(os.environ,
                         {'ETCD_ENDPOINTS': 'http://127.0.0.1:2379'}):
      with _dynamo_topology(commands):
        server = vllm_inference.VLLMCompletionsModelHandler(
            model_name='test-model',
            vllm_server_kwargs={
                'tensor-parallel-size': '1'
            },
            use_dynamo=True,
            dynamo_frontend_kwargs={
                'router-mode': 'round-robin'
            }).load_model()
        server._runtime = None  # drop fake children before gc

    # Exactly one frontend + one engine (external etcd, so no etcd process).
    self.assertEqual(2, len(commands))
    frontend_cmd = _cmd_with(commands, 'dynamo.frontend')['cmd']
    engine_cmd = _cmd_with(commands, 'dynamo.vllm')['cmd']
    self.assertIn('--http-port', frontend_cmd)
    self.assertIn('--discovery-backend', frontend_cmd)
    self.assertIn('--request-plane', frontend_cmd)
    self.assertIn('--event-plane', frontend_cmd)
    self.assertIn('--router-mode', frontend_cmd)
    self.assertIn('--no-router-kv-events', frontend_cmd)
    self.assertNotIn('--model', frontend_cmd)
    self.assertNotIn('--tensor-parallel-size', frontend_cmd)
    self.assertIn('--model', engine_cmd)
    self.assertIn('test-model', engine_cmd)
    self.assertIn('--kv-events-config', engine_cmd)
    self.assertIn('--tensor-parallel-size', engine_cmd)
    self.assertNotIn('--http-port', engine_cmd)
    self.assertNotIn('--router-mode', engine_cmd)

  def test_typed_config_two_engines_kv_routing(self):
    commands = []
    config = DynamoRuntimeConfig(
        engines=(
            DynamoVLLMEngineSpec(gpu_devices=('0', )),
            DynamoVLLMEngineSpec(gpu_devices=('1', )),
        ),
        router_mode='kv',
        kv_event_mode='zmq')
    with mock.patch.dict(os.environ,
                         {'ETCD_ENDPOINTS': 'http://127.0.0.1:2379'}):
      with _dynamo_topology(commands):
        server = vllm_inference.VLLMCompletionsModelHandler(
            model_name='test-model',
            vllm_server_kwargs={
                'max-num-seqs': '32'
            },
            use_dynamo=True,
            dynamo_runtime_config=config).load_model()
        server._runtime = None

    engines = [c for c in commands if c['label'].startswith('dynamo.vllm')]
    frontend = _cmd_with(commands, 'dynamo.frontend')
    self.assertEqual(len(engines), 2)
    self.assertNotIn('--no-router-kv-events', frontend['cmd'])
    self.assertEqual(engines[0]['env']['CUDA_VISIBLE_DEVICES'], '0')
    self.assertEqual(engines[1]['env']['CUDA_VISIBLE_DEVICES'], '1')

  def test_chat_handler_typed_config(self):
    commands = []
    config = DynamoRuntimeConfig(
        engines=(
            DynamoVLLMEngineSpec(gpu_devices=('0', )),
            DynamoVLLMEngineSpec(gpu_devices=('1', )),
        ),
        router_mode='kv',
        kv_event_mode='zmq')
    with mock.patch.dict(os.environ,
                         {'ETCD_ENDPOINTS': 'http://127.0.0.1:2379'}):
      with _dynamo_topology(commands):
        server = vllm_inference.VLLMChatModelHandler(
            model_name='test-model',
            use_dynamo=True,
            dynamo_runtime_config=config).load_model()
        server._runtime = None
    self.assertEqual(
        2, len([c for c in commands if c['label'].startswith('dynamo.vllm')]))

  def test_runtime_config_and_frontend_kwargs_are_mutually_exclusive(self):
    with self.assertRaisesRegex(ValueError, 'not\\s+both'):
      vllm_inference.VLLMCompletionsModelHandler(
          model_name='test-model',
          use_dynamo=True,
          dynamo_runtime_config=DynamoRuntimeConfig(
              engines=(DynamoVLLMEngineSpec(gpu_devices=('0', )), )),
          dynamo_frontend_kwargs={'router-mode': 'kv'})

  def test_dynamo_config_requires_use_dynamo(self):
    with self.assertRaisesRegex(ValueError, 'use_dynamo=True'):
      vllm_inference.VLLMCompletionsModelHandler(
          model_name='test-model',
          dynamo_runtime_config=DynamoRuntimeConfig(
              engines=(DynamoVLLMEngineSpec(gpu_devices=('0', )), )))

  def test_invalid_topology_fails_at_handler_construction(self):
    # Overlapping GPUs must fail before any Dataflow launch.
    with self.assertRaisesRegex(ValueError, 'disjoint'):
      vllm_inference.VLLMCompletionsModelHandler(
          model_name='test-model',
          use_dynamo=True,
          dynamo_runtime_config=DynamoRuntimeConfig(
              engines=(
                  DynamoVLLMEngineSpec(gpu_devices=('0', )),
                  DynamoVLLMEngineSpec(gpu_devices=('0', )),
              )))

  def test_validate_inference_args_accepts_openai_request_kwargs(self):
    vllm_inference.VLLMCompletionsModelHandler(
        'test-model').validate_inference_args({'max_tokens': 8})
    vllm_inference.VLLMChatModelHandler('test-model').validate_inference_args(
        {'max_tokens': 8})


if __name__ == '__main__':
  unittest.main()
