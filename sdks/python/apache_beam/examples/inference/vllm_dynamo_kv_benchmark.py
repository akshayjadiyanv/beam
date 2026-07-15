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

"""Multi-engine Dynamo KV-routing A/B/C benchmark pipeline.

Runs one of three same-hardware arms over a prefix-reuse dataset so the
KV-aware router can be isolated (see ``DYNAMO_DATAFLOW_NEXT_PR_POC_PLAN.md``
section 10.4):

  * ``native-dp``  -- native vLLM with internal data parallelism (fair,
                      non-Dynamo, same-hardware baseline);
  * ``dynamo-rr``  -- two aggregated one-GPU engines, round-robin router,
                      KV events off (isolates generic Dynamo overhead);
  * ``dynamo-kv``  -- the same two engines with the KV-aware router and real
                      KV-cache events (the treatment).

Only the arm differs between runs; model, dataset, batching, and generation
settings are held fixed. The most important comparison is ``dynamo-kv`` vs
``dynamo-rr``; ``native-dp`` answers the broader product question.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable
from typing import Any
from typing import Optional

import apache_beam as beam
from apache_beam.metrics import Metrics
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.vllm_inference import DynamoRuntimeConfig
from apache_beam.ml.inference.vllm_inference import DynamoVLLMEngineSpec
from apache_beam.ml.inference.vllm_inference import VLLMCompletionsModelHandler
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult

_ARMS = ('native-dp', 'dynamo-rr', 'dynamo-kv')

_DEFAULT_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_TOKENS = 64
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
_DEFAULT_MAX_NUM_SEQS = 32
_DEFAULT_MAX_MODEL_LEN = 4096
_DEFAULT_NUM_GPUS = 2

_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


def parse_known_args(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, help='Input JSONL URI.')
  parser.add_argument('--output', required=True, help='Output JSONL prefix.')
  parser.add_argument(
      '--arm', required=True, choices=_ARMS, help='Experimental arm.')
  parser.add_argument('--run_id', required=True)
  parser.add_argument('--model', default=_DEFAULT_MODEL)
  parser.add_argument(
      '--model_revision',
      default=None,
      help='Immutable Hugging Face revision (commit SHA). Strongly '
      'recommended so every arm loads identical weights.')
  parser.add_argument(
      '--num_gpus',
      type=int,
      default=_DEFAULT_NUM_GPUS,
      help='GPUs on the worker: engine count for Dynamo arms / data-parallel '
      'size for native-dp.')
  parser.add_argument('--max_tokens', type=int, default=_DEFAULT_MAX_TOKENS)
  parser.add_argument('--batch_size', type=int, default=_DEFAULT_BATCH_SIZE)
  parser.add_argument(
      '--gpu_memory_utilization',
      type=float,
      default=_DEFAULT_GPU_MEMORY_UTILIZATION)
  parser.add_argument('--max_num_seqs', type=int, default=_DEFAULT_MAX_NUM_SEQS)
  parser.add_argument(
      '--max_model_len', type=int, default=_DEFAULT_MAX_MODEL_LEN)
  parser.add_argument('--temperature', type=float, default=0.0)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument(
      '--request_nvext',
      action='store_true',
      help='Diagnostic only: request nvext worker_id/timing fields. Keep off '
      'for primary measured requests (native vLLM may reject the extra body).')
  return parser.parse_known_args(argv)


def common_vllm_server_kwargs(known_args) -> dict[str, str]:
  kwargs = {
      'max-num-seqs': str(known_args.max_num_seqs),
      'gpu-memory-utilization': str(known_args.gpu_memory_utilization),
      'max-model-len': str(known_args.max_model_len),
  }
  if known_args.model_revision:
    kwargs['revision'] = known_args.model_revision
  return kwargs


def build_model_handler(known_args) -> VLLMCompletionsModelHandler:
  """Construct the handler for the selected arm; only the arm varies."""
  vllm_kwargs = common_vllm_server_kwargs(known_args)
  arm = known_args.arm

  if arm == 'native-dp':
    # Fair same-hardware control: one native vLLM server with internal data
    # parallelism across all GPUs (each rank keeps an independent KV cache).
    vllm_kwargs['data-parallel-size'] = str(known_args.num_gpus)
    return VLLMCompletionsModelHandler(
        model_name=known_args.model,
        vllm_server_kwargs=vllm_kwargs,
        min_batch_size=known_args.batch_size,
        max_batch_size=known_args.batch_size)

  engines = tuple(
      DynamoVLLMEngineSpec(gpu_devices=(str(i), ))
      for i in range(known_args.num_gpus))
  if arm == 'dynamo-rr':
    config = DynamoRuntimeConfig(
        engines=engines, router_mode='round-robin', kv_event_mode='disabled')
  else:  # dynamo-kv
    config = DynamoRuntimeConfig(
        engines=engines, router_mode='kv', kv_event_mode='zmq')

  return VLLMCompletionsModelHandler(
      model_name=known_args.model,
      vllm_server_kwargs=vllm_kwargs,
      use_dynamo=True,
      dynamo_runtime_config=config,
      min_batch_size=known_args.batch_size,
      max_batch_size=known_args.batch_size)


def build_inference_args(known_args) -> dict[str, Any]:
  args: dict[str, Any] = {
      'max_tokens': known_args.max_tokens,
      'temperature': known_args.temperature,
      'seed': known_args.seed,
  }
  # nvext is Dynamo-only and native vLLM may reject an unknown extra body, so
  # it is opt-in and never mixed into the native-dp arm.
  if known_args.request_nvext and known_args.arm != 'native-dp':
    args['extra_body'] = {'nvext': {'extra_fields': ['worker_id', 'timing']}}
  return args


def parse_input_record(line: str) -> dict[str, Any]:
  record = json.loads(line)
  for key in ('id', 'prefix_family_id', 'reuse_cohort', 'prompt'):
    if key not in record:
      raise ValueError(f'Input record missing required field {key!r}: {record}')
  return record


def extract_completion_text(inference: Any) -> str:
  if inference is None:
    return ''
  if hasattr(inference, 'choices') and inference.choices:
    return getattr(inference.choices[0], 'text', None) or ''
  if hasattr(inference, 'text'):
    return inference.text or ''
  return str(inference)


def extract_finish_reason(inference: Any) -> Optional[str]:
  if (inference is None or not hasattr(inference, 'choices') or
      not inference.choices):
    return None
  return getattr(inference.choices[0], 'finish_reason', None)


def extract_usage(inference: Any) -> dict[str, Optional[int]]:
  usage = getattr(inference, 'usage', None)
  if usage is None:
    return {
        'prompt_tokens': None,
        'completion_tokens': None,
        'total_tokens': None,
        'cached_tokens': None,
    }
  cached = None
  details = getattr(usage, 'prompt_tokens_details', None)
  if details is not None:
    cached = getattr(details, 'cached_tokens', None)
    if cached is None and isinstance(details, dict):
      cached = details.get('cached_tokens')
  return {
      'prompt_tokens': getattr(usage, 'prompt_tokens', None),
      'completion_tokens': getattr(usage, 'completion_tokens', None),
      'total_tokens': getattr(usage, 'total_tokens', None),
      'cached_tokens': cached,
  }


def extract_nvext(inference: Any) -> Any:
  """Best-effort extraction of Dynamo ``nvext`` response metadata."""
  if inference is None:
    return None
  model_extra = getattr(inference, 'model_extra', None)
  if isinstance(model_extra, dict) and 'nvext' in model_extra:
    return model_extra.get('nvext')
  if hasattr(inference, 'nvext'):
    return getattr(inference, 'nvext')
  if hasattr(inference, 'choices') and inference.choices:
    choice_extra = getattr(inference.choices[0], 'model_extra', None)
    if isinstance(choice_extra, dict) and 'nvext' in choice_extra:
      return choice_extra.get('nvext')
  return None


def extract_worker_id(nvext: Any) -> Optional[str]:
  if isinstance(nvext, dict):
    worker = nvext.get('worker_id')
    return str(worker) if worker is not None else None
  worker = getattr(nvext, 'worker_id', None)
  return str(worker) if worker is not None else None


def parse_classification(
    raw_text: str) -> tuple[Optional[dict[str, Any]], bool]:
  if not raw_text:
    return None, False
  match = _JSON_OBJECT_RE.search(raw_text)
  if not match:
    return None, False
  try:
    parsed = json.loads(match.group(0))
  except json.JSONDecodeError:
    return None, False
  if not isinstance(parsed, dict):
    return None, False
  return parsed, True


def format_output_record(
    record_meta: dict[str, Any],
    prediction: PredictionResult,
    *,
    arm: str,
    run_id: str) -> dict[str, Any]:
  raw_text = extract_completion_text(prediction.inference)
  parsed, valid = parse_classification(raw_text)
  usage = extract_usage(prediction.inference)
  nvext = extract_nvext(prediction.inference)
  return {
      'id': record_meta['id'],
      'prefix_family_id': record_meta['prefix_family_id'],
      'reuse_cohort': record_meta['reuse_cohort'],
      'arm': arm,
      'run_id': run_id,
      'raw_text': raw_text,
      'parsed': parsed,
      'valid_json': valid,
      'finish_reason': extract_finish_reason(prediction.inference),
      'prompt_tokens': usage['prompt_tokens'],
      'completion_tokens': usage['completion_tokens'],
      'total_tokens': usage['total_tokens'],
      'cached_tokens': usage['cached_tokens'],
      'worker_id': extract_worker_id(nvext),
      'nvext': nvext,
      'has_nvext': nvext is not None,
  }


class LogEnvironmentDoFn(beam.DoFn):
  """Logs once-per-worker environment details for the run manifest."""
  def __init__(self, *, run_id, arm, model, model_revision, num_gpus):
    self._run_id = run_id
    self._arm = arm
    self._model = model
    self._model_revision = model_revision
    self._num_gpus = num_gpus
    self._logged = False

  def setup(self):
    if self._logged:
      return
    self._logged = True
    info = {
        'run_id': self._run_id,
        'arm': self._arm,
        'model': self._model,
        'model_revision': self._model_revision,
        'num_gpus': self._num_gpus,
        'python': sys.version,
        'apache_beam': getattr(beam, '__version__', 'unknown'),
    }
    for mod_name in ('vllm', 'openai', 'dynamo'):
      try:
        mod = __import__(mod_name)
        info[mod_name] = getattr(mod, '__version__', 'imported-no-version')
      except Exception as exc:  # pylint: disable=broad-except
        info[mod_name] = f'unavailable: {exc}'
    logging.info('KV_POC_ENV_MANIFEST %s', json.dumps(info, sort_keys=True))

  def process(self, element):
    yield element


class ParseInputDoFn(beam.DoFn):
  def __init__(self):
    self._processed = Metrics.counter(self.__class__, 'processed_records')

  def process(self, line: str):
    record = parse_input_record(line)
    self._processed.inc()
    meta = {
        'id': record['id'],
        'prefix_family_id': record['prefix_family_id'],
        'reuse_cohort': record['reuse_cohort'],
    }
    yield (meta, record['prompt'])


class FormatOutputDoFn(beam.DoFn):
  def __init__(self, *, arm: str, run_id: str):
    self._arm = arm
    self._run_id = run_id
    self._valid = Metrics.counter(self.__class__, 'valid_json_outputs')
    self._invalid = Metrics.counter(self.__class__, 'invalid_json_outputs')
    self._failed = Metrics.counter(self.__class__, 'failed_requests')
    self._prompt_tokens = Metrics.counter(self.__class__, 'prompt_tokens')
    self._completion_tokens = Metrics.counter(
        self.__class__, 'completion_tokens')
    self._cached_tokens = Metrics.counter(self.__class__, 'cached_tokens')
    self._nvext = Metrics.counter(self.__class__, 'responses_with_nvext')
    self._with_worker_id = Metrics.counter(
        self.__class__, 'responses_with_worker_id')

  def process(
      self, element: tuple[dict[str, Any], PredictionResult]) -> Iterable[str]:
    meta, prediction = element
    try:
      out = format_output_record(
          meta, prediction, arm=self._arm, run_id=self._run_id)
    except Exception as exc:  # pylint: disable=broad-except
      self._failed.inc()
      out = {
          'id': meta.get('id'),
          'prefix_family_id': meta.get('prefix_family_id'),
          'reuse_cohort': meta.get('reuse_cohort'),
          'arm': self._arm,
          'run_id': self._run_id,
          'raw_text': '',
          'parsed': None,
          'valid_json': False,
          'error': str(exc),
          'has_nvext': False,
      }
      yield json.dumps(out, sort_keys=True)
      return

    if out['valid_json']:
      self._valid.inc()
    else:
      self._invalid.inc()
    if out.get('prompt_tokens'):
      self._prompt_tokens.inc(int(out['prompt_tokens']))
    if out.get('completion_tokens'):
      self._completion_tokens.inc(int(out['completion_tokens']))
    if out.get('cached_tokens'):
      self._cached_tokens.inc(int(out['cached_tokens']))
    if out.get('has_nvext'):
      self._nvext.inc()
    if out.get('worker_id'):
      self._with_worker_id.inc()
    yield json.dumps(out, sort_keys=True)


def run(
    argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
  known_args, pipeline_args = parse_known_args(argv)
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

  model_handler = KeyedModelHandler(build_model_handler(known_args))
  inference_args = build_inference_args(known_args)

  pipeline = test_pipeline or beam.Pipeline(options=pipeline_options)
  _ = (
      pipeline
      | 'ReadInput' >> beam.io.ReadFromText(known_args.input)
      | 'LogEnv' >> beam.ParDo(
          LogEnvironmentDoFn(
              run_id=known_args.run_id,
              arm=known_args.arm,
              model=known_args.model,
              model_revision=known_args.model_revision,
              num_gpus=known_args.num_gpus))
      | 'ParseInput' >> beam.ParDo(ParseInputDoFn())
      | 'RunInference' >> RunInference(
          model_handler, inference_args=inference_args)
      | 'FormatOutput' >> beam.ParDo(
          FormatOutputDoFn(arm=known_args.arm, run_id=known_args.run_id))
      | 'WriteOutput' >> beam.io.WriteToText(
          known_args.output,
          file_name_suffix='.jsonl',
          shard_name_template='',
          append_trailing_newlines=True))

  result = pipeline.run()
  result.wait_until_finish()
  return result


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
