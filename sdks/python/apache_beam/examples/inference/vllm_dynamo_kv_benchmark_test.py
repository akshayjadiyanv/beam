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

"""Unit tests for the Dynamo KV-routing benchmark helpers (no GPU)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import apache_beam as beam
from apache_beam.examples.inference import vllm_dynamo_kv_benchmark as bench
from apache_beam.examples.inference import vllm_dynamo_kv_benchmark_data as data
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to


def _system_block(prompt: str) -> str:
  start = prompt.index('<|im_start|>system\n') + len('<|im_start|>system\n')
  end = prompt.index('<|im_end|>', start)
  return prompt[start:end]


class DatasetGeneratorTest(unittest.TestCase):
  def test_deterministic(self):
    a = data.generate_records(20, reuse_fraction=0.75, seed=42)
    b = data.generate_records(20, reuse_fraction=0.75, seed=42)
    self.assertEqual(a, b)

  def test_zero_reuse_has_only_unique_prefixes(self):
    records = data.generate_records(30, reuse_fraction=0.0, seed=1)
    self.assertTrue(
        all(r['prefix_family_id'].startswith('unique-') for r in records))
    # Every unique prefix differs across records -> no cross-request reuse.
    prefixes = {_system_block(r['prompt']) for r in records}
    self.assertEqual(len(prefixes), len(records))

  def test_full_reuse_draws_from_shared_families(self):
    records = data.generate_records(
        200, reuse_fraction=1.0, num_families=4, seed=7)
    self.assertTrue(
        all(r['prefix_family_id'].startswith('family-') for r in records))
    self.assertLessEqual(len({r['prefix_family_id'] for r in records}), 4)

  def test_shared_family_prefix_is_byte_identical(self):
    records = data.generate_records(
        200, reuse_fraction=1.0, num_families=3, seed=11)
    by_family: dict[str, set] = {}
    for r in records:
      by_family.setdefault(r['prefix_family_id'],
                           set()).add(_system_block(r['prompt']))
    # Each family maps to exactly one shared prefix string.
    for family, prefixes in by_family.items():
      self.assertEqual(len(prefixes), 1, f'family {family} prefix drifted')

  def test_prompt_starts_with_reusable_prefix_not_id(self):
    record = data.generate_records(1, reuse_fraction=1.0, seed=5)[0]
    self.assertTrue(record['prompt'].startswith('<|im_start|>system\n'))
    # The record id must NOT appear before the shared prefix.
    self.assertNotIn(record['id'], _system_block(record['prompt']))

  def test_write_dataset_manifest(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'reuse.jsonl')
      manifest = data.write_dataset(
          path, 12, reuse_fraction=0.5, num_families=4, seed=3)
      self.assertEqual(manifest['num_records'], 12)
      self.assertEqual(manifest['reuse_cohort'], 'reuse-050')
      self.assertEqual(len(manifest['sha256']), 64)
      with open(path, encoding='utf-8') as f:
        lines = [ln for ln in f.read().splitlines() if ln]
      self.assertEqual(len(lines), 12)

  def test_invalid_reuse_fraction(self):
    with self.assertRaises(ValueError):
      data.generate_records(4, reuse_fraction=1.5)


class ArmSelectionTest(unittest.TestCase):
  def _args(self, arm, **overrides):
    argv = [
        '--input',
        'in.jsonl',
        '--output',
        'out',
        '--arm',
        arm,
        '--run_id',
        f'{arm}-1'
    ]
    for k, v in overrides.items():
      argv += [f'--{k}', str(v)]
    known, _ = bench.parse_known_args(argv)
    return known

  def test_native_dp_arm(self):
    handler = bench.build_model_handler(self._args('native-dp', num_gpus=2))
    self.assertFalse(handler._use_dynamo)
    self.assertIsNone(handler._dynamo_config)
    self.assertEqual(handler._vllm_server_kwargs['data-parallel-size'], '2')

  def test_dynamo_rr_arm(self):
    handler = bench.build_model_handler(self._args('dynamo-rr', num_gpus=2))
    self.assertTrue(handler._use_dynamo)
    config = handler._dynamo_config
    self.assertEqual(config.router_mode, 'round-robin')
    self.assertEqual(config.kv_event_mode, 'disabled')
    self.assertEqual(len(config.engines), 2)
    self.assertEqual(config.engines[0].gpu_devices, ('0', ))
    self.assertEqual(config.engines[1].gpu_devices, ('1', ))
    self.assertNotIn('data-parallel-size', handler._vllm_server_kwargs)

  def test_dynamo_kv_arm(self):
    handler = bench.build_model_handler(self._args('dynamo-kv', num_gpus=2))
    config = handler._dynamo_config
    self.assertEqual(config.router_mode, 'kv')
    self.assertEqual(config.kv_event_mode, 'zmq')
    self.assertEqual(len(config.engines), 2)

  def test_nvext_only_for_dynamo_arms(self):
    kv = self._args('dynamo-kv')
    kv.request_nvext = True
    self.assertIn('extra_body', bench.build_inference_args(kv))
    native = self._args('native-dp')
    native.request_nvext = True
    self.assertNotIn('extra_body', bench.build_inference_args(native))

  def test_common_kwargs_include_revision_when_set(self):
    args = self._args('dynamo-kv', model_revision='deadbeef')
    kwargs = bench.common_vllm_server_kwargs(args)
    self.assertEqual(kwargs['revision'], 'deadbeef')
    self.assertEqual(kwargs['max-model-len'], str(bench._DEFAULT_MAX_MODEL_LEN))


class OutputParsingTest(unittest.TestCase):
  def test_parse_input_record_missing_field(self):
    with self.assertRaises(ValueError):
      bench.parse_input_record(json.dumps({'id': 'x'}))

  def test_extract_usage_cached_tokens(self):
    inference = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=2000,
            completion_tokens=40,
            total_tokens=2040,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1900)))
    usage = bench.extract_usage(inference)
    self.assertEqual(usage['cached_tokens'], 1900)
    self.assertEqual(usage['prompt_tokens'], 2000)

  def test_extract_worker_id(self):
    self.assertEqual(bench.extract_worker_id({'worker_id': 7}), '7')
    self.assertIsNone(bench.extract_worker_id({'timing': {}}))
    self.assertIsNone(bench.extract_worker_id(None))

  def test_format_output_record(self):
    inference = SimpleNamespace(
        choices=[
            SimpleNamespace(
                text='{"category":"access","priority":"p2",'
                '"sentiment":"neutral"}',
                finish_reason='stop')
        ],
        usage=SimpleNamespace(
            prompt_tokens=2000,
            completion_tokens=5,
            total_tokens=2005,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1800)),
        model_extra={'nvext': {
            'worker_id': 1, 'timing': {
                'ttft': 0.2
            }
        }})
    out = bench.format_output_record({
        'id': 'req-0',
        'prefix_family_id': 'family-001',
        'reuse_cohort': 'reuse-095'
    },
                                     PredictionResult('prompt', inference),
                                     arm='dynamo-kv',
                                     run_id='dynamo-kv-1')
    self.assertTrue(out['valid_json'])
    self.assertEqual(out['cached_tokens'], 1800)
    self.assertEqual(out['worker_id'], '1')
    self.assertTrue(out['has_nvext'])
    self.assertEqual(out['arm'], 'dynamo-kv')

  def test_pipeline_graph_under_direct_runner(self):
    records = data.generate_records(3, reuse_fraction=1.0, num_families=2)
    lines = [json.dumps(r, sort_keys=True) for r in records]

    def _fake_infer(element):
      meta, prompt = element
      inference = SimpleNamespace(
          choices=[
              SimpleNamespace(
                  text='{"category":"billing","priority":"p3",'
                  '"sentiment":"polite"}',
                  finish_reason='stop')
          ],
          usage=SimpleNamespace(
              prompt_tokens=3, completion_tokens=2, total_tokens=5),
          model_extra=None)
      return (meta, PredictionResult(prompt, inference))

    with TestPipeline() as p:
      out = (
          p
          | beam.Create(lines)
          | beam.ParDo(bench.ParseInputDoFn())
          | beam.Map(_fake_infer)
          | beam.ParDo(bench.FormatOutputDoFn(arm='dynamo-kv', run_id='t1')))
      assert_that(
          out | beam.Map(lambda s: json.loads(s)['id']),
          equal_to(['req-000000', 'req-000001', 'req-000002']))


if __name__ == '__main__':
  unittest.main()
