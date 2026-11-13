# ACT and SmolVLA diagnostic procedure

Use the installed implementation as the operational source of truth. The source
anchors below were inspected at `e6ffde603` (LeRobot fork, November 2026); names
and flags must be rechecked for other revisions. This workflow does not require
the separate dataset-audit skill to be installed.

## Artifact identity and coverage

Start with explicit local paths. Discover package/version without constructing a
policy, dataset, robot, or processor. Record the imported package's resolved
location, distribution version, checkout HEAD and dirty status. Distinguish an
editable checkout from a different installed copy. Do not collect environment
variables, credential files, or an unfiltered dependency/environment dump.

Inspect `common/train_utils.py:save_checkpoint`, policy `pretrained.py`, and
`utils/constants.py`. A saved checkpoint normally contains `pretrained_model/`
with `config.json`, `train_config.json`, model weights, and
`policy_preprocessor.json` / `policy_postprocessor.json`. EMA checkpoints,
adapters, and sharded weights need their actual loading path identified. Record
all evaluated weight shard digests and adapter/base identities if applicable.
Do not identify weights using the configuration hash. Resolve `last`/symlinks
once and distinguish the intended checkpoint from the one the runtime loaded.

Inspect processor `steps`, `state_file`, and local tokenizer `artifacts` references
in `processor/pipeline.py`. Check file existence and compare normalization tensor
keys, feature association, dimensions, modes, and values against the training
artifacts. Hashes establish byte identity only: differing serialization can encode
equal values, and matching files do not prove the runtime used them. Missing local
state is incomplete evidence, not proof the Hub version is broken. Never load a
pickle checkpoint from untrusted evidence merely to inspect metadata.

Include relevant training settings (dataset revision/split, seed, optimizer steps,
checkpoint-selection rule) and selected loss/validation measurements. Loss values
from different normalization, masks, objectives, or splits are not comparable.
Record missing artifacts and limit conclusions accordingly.

## Observations and actions

Trace the real caller through `common/control_utils.py:predict_action` and the
active rollout engine, then through `policies/factory.py:make_pre_post_processors`.
Inspect runtime overrides, renaming, relative-action processors, clipping and robot
action conversion rather than inferring semantics from a policy family.

- Compare camera keys **after renaming**, recording any generated empty cameras.
  Compare per-sample shapes at the same stage; dataset HWC storage and batched
  BCHW tensors are not directly comparable. Check RGB/BGR, channel order, range,
  resizing/cropping, camera roles and freshness separately. Shape equality cannot
  prove that a camera shows the intended workspace.
- Trace each state/action index to dataset feature names and robot joint names.
  Compare physical units and gripper open/closed direction/range. Joint names may
  require an explicit mapping; a permutation is not automatically a fault if the
  processor intentionally inverts it. Do not assume radians, degrees, or normalized
  motor positions from numerical ranges alone.
- Follow normalization into the policy and unnormalization out. Compare selected
  before/after values, finite statistics, per-feature association and round trips
  with the saved state. Never silently regenerate statistics to make a check pass.
- Record predicted horizon (`chunk_size`), actions consumed before replanning
  (`n_action_steps`), temporal aggregation, requested control Hz, measured dispatch
  Hz, observation age, and latency. Dataset FPS is a timing reference, not a proof
  of the physical capture rate. A frequency difference is a hypothesis requiring
  the execution contract and trace, not an automatic format violation.

## ACT branch

Inspect `policies/act/{configuration_act,modeling_act,processor_act}.py` and the
default processor factory. ACT's `select_action` consumes a queue unless temporal
ensembling is enabled; `reset()` clears queue/ensemble history. Compare resets at
episode boundaries, including aborted attempts. A missing reset log is not proof
that reset was omitted. A nonempty queue before a new observation can be normal
within an episode. A reset should be inspected before the first new episode's
action selection; a newly populated queue after selection is also normal.

With `temporal_ensemble_coeff` set, current ACT requires `n_action_steps == 1` and
predicts each step. Otherwise `n_action_steps <= chunk_size`. A different chunk
size changes the predicted horizon; it does not by itself identify replanning Hz.
ACT's VAE training objective includes KL; its inference path does not use the
training action-conditioned latent. Use `select_action` / `predict_action_chunk`
for inference evidence rather than interpreting a training forward pass as an
equivalent rollout. Mask padded action targets when comparing validation loss.

## SmolVLA branch

Inspect `policies/smolvla/{configuration_smolvla,modeling_smolvla,processor_smolvla}.py`.
The saved path includes task newline handling and tokenization; inspect task
presence, batch cardinality, truncation, tokenizer identity/local artifacts and
runtime overrides. ACT need not consume language, so missing language is not a
universal fault. Do not put private task text or token contents into a bundle.

Compare image padding/resizing, `empty_cameras`, state/action padding dimensions,
`adapt_to_pi_aloha`, and any relative-action processor. In this revision
`use_delta_joint_actions_aloha=True` is rejected; do not recommend enabling an
unimplemented option. Inspect queue/reset behavior and RTC separately; a trace
field for ACT's queue cannot validate an asynchronous RTC implementation.
SmolVLA sampling is stochastic: record seed, device, dtype and sampling steps and
use appropriate tolerances rather than requiring bitwise equality.

## Small offline prediction experiment

Inspect loaders for network/initialization effects before running. Use only local
artifacts and explicit offline settings (`HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`); those settings supplement inspection and are not proof
of isolation. Stop if a required backbone, tokenizer or dependency is missing.
Do not initialize an unrelated default model or install packages automatically.

Choose a few fixed episode/frame indices from held-out data with seed and exact
coverage recorded. Respect the current reader's delta indices and padding masks.
Load the actual saved policy and both saved processors using inspected current
APIs and `local_files_only=True` where supported. Disable compilation unless it
is part of the comparison and within budget. Call `eval()`, use inference mode,
reset the policy and any stateful processors between episodes, preprocess the
batch, call inference, and postprocess selected actions. Record shapes, finite
values, per-joint errors in physical units, and bounded selected numeric traces.
Compare actions and targets in the same coordinate/normalization space. Do not
construct a robot/env, dispatch commands, or run a real-world evaluation CLI.

Separate: raw model prediction → policy postprocessing → robot action processing
→ dispatch → measured response. Clipping, stale observations, command drops,
resets, clock domains and latency can distinguish a plausible model prediction
from an execution fault. Without downstream traces, those links are not checked.

## Ranking and experiments

Prioritize an observed contract discrepancy before speculative capacity or data
volume explanations, while keeping severity and confidence separate. A verified
contract violation is still only a possible cause of the physical task failure.
For each hypothesis retain contradictory and missing evidence; absence of an
artifact is not supporting evidence that it was wrong at runtime.

Examples of smallest experiments:

| Evidence                                          | Distinguishing experiment                                                             | Interpretation                                                                              |
| ------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Runtime key differs from trained camera key       | Replay one recorded observation through the saved renamer without a robot             | Correct downstream key weakens mismatch; persistent mismatch supports it                    |
| Normalization hashes differ                       | Compare decoded tensor keys/values and a fixed offline round trip                     | Equal tensors weaken mismatch; different feature/scales support it                          |
| Ordered joints or action convention differs       | Trace one offline synthetic vector through both documented mappings                   | An intentional inverse mapping explains discrepancy; wrong physical destination supports it |
| Checkpoint digests differ                         | Recompute complete evaluated/intended weight identities                               | A true difference confirms identity mismatch, not which checkpoint would succeed            |
| Queue contains old entries after an episode reset | Replay two episodes through the actual offline reset/select path                      | Clearing before first selection weakens leakage; old actions surviving supports it          |
| Dispatch is slower than demonstration timing      | Compare existing timestamped actions and observation age against the runtime contract | Deadline misses support timing hypothesis; matching Hz does not rule out latency            |
| Contracts match but contact failure is ambiguous  | Inspect the shortest existing failed grasp clip against a successful demonstration    | Placement/occlusion/contact differences narrow hypotheses; no clip leaves them unresolved   |

Local inspection is CPU-small; bounded inference is CPU/GPU-small only if measured
memory/runtime fit the agreed budget. Hardware experiments, uploads, and training
require their own scope/authorization. Estimate wall time only from measured
throughput. Do not prescribe extra demonstrations or training until a controlled
experiment would distinguish that explanation.

## Behavioral acceptance tasks

Use the [paired agent evaluation harness](../../../docs/agent-skill-evaluation.md)
to prepare matching tasks, capture actual agent runs and elapsed time, and load
independent adjudications. The harness includes camera-mismatch and
insufficient-evidence cases; its unit tests are bookkeeping checks, not measured
agent behavior.

Use the scenarios above plus missing-all-evidence and clean-contract physical
failure requests to evaluate the **agent**, not just the helper. Supply raw
synthetic configs/trace snippets, hide the intended answer from the evaluator,
and record routing, findings, unsupported claims, executed commands, interventions
and elapsed time with/without the skill. Include a valid camera rename, a normal
within-episode queue, and ACT without language as false-alarm cases. Reject any
run that launches training/motion, treats missing data as a verified fault, or
claims closed-loop success from offline agreement. The automated helper tests
exercise comparisons and export boundaries; they do not measure agent behavior.
