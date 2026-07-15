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

"""Prefix-reuse dataset for the multi-engine KV-routing Dynamo POC.

Each record is a document-grounded classification request. The rendered prompt
begins with a **byte-identical reusable prefix** (system instructions + one
long policy document + an output schema) shared by every record in a prefix
"family"; the short variable ticket follows. That layout is what lets vLLM
prefix caching and Dynamo's KV-aware router observe the intended overlap:
nothing unique (id, timestamp) is placed before the shared prefix.

A dataset is generated for one reuse cohort at a time (``--reuse_fraction``).
With fraction ``r``, a fraction ``r`` of records draw their long prefix from a
small pool of shared families (high cross-request reuse); the remaining records
get a per-record unique prefix (no reuse). Families are interleaved across the
stream so a naive round-robin router tends to split a family across both KV
caches -- the exact condition a KV-aware router is meant to exploit.

The generator is deterministic given ``--seed`` so every arm consumes the same
immutable input. See ``DYNAMO_DATAFLOW_NEXT_PR_POC_PLAN.md`` section 10.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Any

_CATEGORIES = (
    'billing', 'access', 'performance', 'data_loss', 'feature_request')
_PRIORITIES = ('p0', 'p1', 'p2', 'p3')
_SENTIMENTS = ('frustrated', 'neutral', 'polite', 'urgent')

_SYSTEM_INSTRUCTIONS = (
    'You are a support-ticket triage assistant operating under the policy '
    'document below. Read the policy, then classify the ticket that follows. '
    'Reply with ONLY a compact JSON object with keys category, priority, and '
    'sentiment. No markdown, no explanation.')

_OUTPUT_SCHEMA = (
    'OUTPUT SCHEMA: {"category": one of '
    '["billing","access","performance","data_loss","feature_request"], '
    '"priority": one of ["p0","p1","p2","p3"], '
    '"sentiment": one of ["frustrated","neutral","polite","urgent"]}')

# Sentence fragments assembled into deterministic, realistic-looking policy
# documents. The exact prose does not matter; what matters is that a given
# family's document is stable (reusable prefix) and long enough that its prefill
# cost is worth caching.
_POLICY_FRAGMENTS = (
    'Section {n}: Tickets referencing {topic} must be routed to the {team} '
    'team within {sla} hours and tagged with the {tag} label.',
    'Section {n}: When a customer reports {topic}, escalate to priority {prio} '
    'if the affected account is on the {plan} plan.',
    'Section {n}: The {team} team owns remediation for {topic}; sentiment is '
    'considered {mood} when the customer references contractual penalties.',
    'Section {n}: For {topic} incidents, confirm the {plan} entitlement before '
    'promising a {sla}-hour resolution window under the {tag} runbook.',
)
_TOPICS = (
    'authentication outages',
    'billing discrepancies',
    'query latency regressions',
    'data synchronization gaps',
    'export pipeline failures',
    'quota exhaustion',
    'permission propagation delays',
    'webhook delivery loss')
_TEAMS = ('platform', 'billing', 'reliability', 'data', 'access')
_PLANS = ('starter', 'business', 'enterprise', 'trial')
_TAGS = ('sev1', 'sev2', 'watch', 'audit')
_MOODS = ('frustrated', 'neutral', 'urgent')


def build_policy_document(rng: random.Random, target_tokens: int) -> str:
  """Assemble a deterministic policy document of roughly ``target_tokens``."""
  # ~4 characters per token is a rough but stable proxy; the exact HF tokenizer
  # count is validated during the pilot, not here.
  target_chars = target_tokens * 4
  lines = []
  section = 1
  size = 0
  while size < target_chars:
    template = _POLICY_FRAGMENTS[section % len(_POLICY_FRAGMENTS)]
    line = template.format(
        n=section,
        topic=rng.choice(_TOPICS),
        team=rng.choice(_TEAMS),
        plan=rng.choice(_PLANS),
        tag=rng.choice(_TAGS),
        mood=rng.choice(_MOODS),
        prio=rng.choice(_PRIORITIES),
        sla=rng.choice((2, 4, 8, 24)))
    lines.append(line)
    size += len(line) + 1
    section += 1
  return '\n'.join(lines)


def build_shared_prefix(family_key: str, prefix_tokens: int, seed: int) -> str:
  """The reusable system block for a family: instructions + policy + schema.

  Deterministic in ``(family_key, seed)`` so every record in a family -- and
  every arm -- sees the byte-identical leading text.
  """
  rng = random.Random(f'{seed}:{family_key}')
  policy = build_policy_document(rng, prefix_tokens)
  return (
      f'{_SYSTEM_INSTRUCTIONS}\n\n'
      f'POLICY DOCUMENT ({family_key}):\n{policy}\n\n'
      f'{_OUTPUT_SCHEMA}')


def _ticket_text(rng: random.Random, idx: int) -> str:
  product = rng.choice(
      ('Acme Cloud', 'Acme Analytics', 'Acme Mobile', 'Acme API'))
  symptom = rng.choice((
      'login failures after SSO rotation',
      'elevated p95 latency on the query path',
      'unexpected invoice line items',
      'missing rows after a nightly sync',
      'UI freezes when exporting CSV',
  ))
  return (
      f'Ticket {idx:06d} for {product}: {symptom}. '
      'Classify this ticket per the policy above.')


def render_chatml(shared_prefix: str, ticket: str) -> str:
  """ChatML with the reusable system block first and the ticket as the suffix.

  ChatML matches the Qwen instruct family used in the POC. Align this template
  with the chosen ``--model`` before a measured run.
  """
  return (
      f'<|im_start|>system\n{shared_prefix}<|im_end|>\n'
      f'<|im_start|>user\n{ticket}<|im_end|>\n'
      f'<|im_start|>assistant\n')


def generate_records(
    num_records: int,
    *,
    reuse_fraction: float,
    num_families: int = 32,
    prefix_tokens: int = 2048,
    seed: int = 42) -> list[dict[str, Any]]:
  """Generate ``num_records`` interleaved prefix-reuse classification records.

  Args:
    num_records: total records to emit.
    reuse_fraction: fraction (0..1) of records that draw a shared family prefix;
      the rest get a unique, non-reused prefix.
    num_families: size of the shared-family pool (tens of families).
    prefix_tokens: approximate token length of each reusable prefix.
    seed: dataset seed; identical output for identical inputs.
  """
  if not 0.0 <= reuse_fraction <= 1.0:
    raise ValueError('reuse_fraction must be in [0, 1].')
  rng = random.Random(f'{seed}:records:{reuse_fraction}')
  cohort = f'reuse-{int(round(reuse_fraction * 100)):03d}'
  # Cache shared-family prefixes so repeated draws reuse the identical string.
  shared_cache: dict[str, str] = {}
  records = []
  for i in range(num_records):
    if num_families > 0 and rng.random() < reuse_fraction:
      family_key = f'family-{rng.randrange(num_families):03d}'
      shared_prefix = shared_cache.get(family_key)
      if shared_prefix is None:
        shared_prefix = build_shared_prefix(family_key, prefix_tokens, seed)
        shared_cache[family_key] = shared_prefix
    else:
      # A unique prefix that no other record shares -> zero reuse opportunity.
      family_key = f'unique-{i:06d}'
      shared_prefix = build_shared_prefix(family_key, prefix_tokens, seed)

    ticket = _ticket_text(rng, i)
    records.append({
        'id': f'req-{i:06d}',
        'prefix_family_id': family_key,
        'reuse_cohort': cohort,
        'reuse_fraction': reuse_fraction,
        'dataset_seed': seed,
        'target_input_tokens': prefix_tokens,
        'prompt': render_chatml(shared_prefix, ticket),
    })
  return records


def write_dataset(
    path: str,
    num_records: int,
    *,
    reuse_fraction: float,
    num_families: int = 32,
    prefix_tokens: int = 2048,
    seed: int = 42) -> dict[str, Any]:
  records = generate_records(
      num_records,
      reuse_fraction=reuse_fraction,
      num_families=num_families,
      prefix_tokens=prefix_tokens,
      seed=seed)
  os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8') as f:
    for rec in records:
      f.write(json.dumps(rec, sort_keys=True) + '\n')
  with open(path, 'rb') as f:
    raw = f.read()
  distinct_families = len({
      r['prefix_family_id']
      for r in records if r['prefix_family_id'].startswith('family-')
  })
  return {
      'path': path,
      'num_records': num_records,
      'reuse_fraction': reuse_fraction,
      'reuse_cohort': f'reuse-{int(round(reuse_fraction * 100)):03d}',
      'num_families': num_families,
      'distinct_shared_families_used': distinct_families,
      'prefix_tokens': prefix_tokens,
      'seed': seed,
      'bytes': len(raw),
      'sha256': hashlib.sha256(raw).hexdigest(),
  }


def main(argv=None):
  parser = argparse.ArgumentParser(
      description='Generate prefix-reuse Dynamo KV-routing POC datasets.')
  parser.add_argument('--output', required=True, help='Output JSONL path.')
  parser.add_argument('--num_records', type=int, required=True)
  parser.add_argument(
      '--reuse_fraction',
      type=float,
      required=True,
      help='Fraction of records drawing a shared prefix family (e.g. 0.95).')
  parser.add_argument('--num_families', type=int, default=32)
  parser.add_argument('--prefix_tokens', type=int, default=2048)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--manifest', default=None)
  args = parser.parse_args(argv)
  manifest = write_dataset(
      args.output,
      args.num_records,
      reuse_fraction=args.reuse_fraction,
      num_families=args.num_families,
      prefix_tokens=args.prefix_tokens,
      seed=args.seed)
  manifest_path = args.manifest or (args.output + '.manifest.json')
  with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write('\n')
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
