# LeRobot Agent Skills: Design and Implementation Plan

## 1. Objective and agreed decisions

Build seven focused agent skills that help LeRobot practitioners identify problems, integrate models, and design useful experiments with less wasted debugging and training time.

The contribution should combine concise agent instructions with reusable tooling, verifiable findings, and tests of actual behavior.

### Agreed defaults

- Target contribution: the LeRobot repository.
- Primary audience: practitioners recording datasets, training policies, and deploying them on robots.
- Describe all seven skills; implement the dataset audit first.
- Initial dataset support: v3.0, the format declared by this checkout.
- Initial audit runs entirely offline.
- Accept local dataset directories and already cached Hugging Face datasets.
- Decode all referenced video coverage by default.
- Implement deterministic auditing as a reusable Python tool, independently usable without an agent.
- Use the user’s cached recordings privately for validation; commit synthetic fixtures only.
- Validate hardware workflows later using the user’s SO-101.
- Initially specialize policy failure diagnosis for ACT and SmolVLA.
- Do not automatically download, repair datasets, launch training, publish artifacts, or actuate hardware.

This document is the implementation specification. Saving it as docs/agent-skills-plan.md is the first documentation action after leaving Plan mode.

## 2. Shared design

### 2.1 Skill packaging

Create the following directories under skills/:

| Skill                       | Primary result                                                       |
| --------------------------- | -------------------------------------------------------------------- |
| lerobot-dataset-audit       | Evidence-backed dataset integrity report                             |
| lerobot-policy-diagnosis    | Ranked hypotheses and distinguishing experiments                     |
| lerobot-setup-check         | SO-101 readiness report with verification boundaries                 |
| lerobot-policy-integrator   | Repository-conformant policy integration and tests                   |
| lerobot-experiment-designer | Reproducible experiment manifest                                     |
| lerobot-recording-review    | Timestamped human review queue                                       |
| lerobot-ros2-adapter        | Offline conversion recipe or reviewed hardware-plugin implementation |

Each skill contains:

- SKILL.md: trigger, exclusions, required inputs, essential procedure, output, and boundaries.
- references/: detailed procedures or examples needed only for specific branches.
- Scripts only where reusable code does not belong in the library itself.

Keep runtime instructions concise. The detailed design document must not become a large instruction file loaded for every request.

Use portable Markdown and YAML frontmatter. Agent-specific discovery metadata may be added separately, without making core behavior dependent on one agent product.

### 2.2 Discover the actual implementation

Each workflow establishes:

- Installed LeRobot version and source location.
- Repository revision when available.
- Whether the imported package matches the checkout being inspected.
- Relevant dataset format, policy family, and hardware configuration.
- Available dependencies and execution resources.
- Relevant CLI entry points and interfaces from the current source.

Use existing implementation and tests as the operational source of truth. Do not assume historical file locations, command flags, processor behavior, or policy interfaces remain valid.

Record unsupported configurations explicitly. Do not improvise compatibility, migrate datasets, or install dependencies automatically.

### 2.3 Evidence model

Diagnostic findings should distinguish:

| Classification | Meaning                                                          |
| -------------- | ---------------------------------------------------------------- |
| Verified fault | An observed violation of a supported invariant                   |
| Warning        | A suspicious measurement that may be legitimate                  |
| Hypothesis     | A possible explanation needing further evidence                  |
| Not checked    | The check was unavailable, unsupported, or outside scope         |
| Passed         | The stated check completed successfully over the stated coverage |

Each finding includes:

- Stable check identifier.
- Classification and severity.
- Artifact location: file, episode, row, video interval, configuration field, or trace.
- Expected and observed behavior.
- Relevant measurements, units, and tolerances.
- Inspection coverage.
- Interpretation limits.
- Recommended next action.

Severity and certainty must remain separate. A potentially serious issue can still be unverified.

Provide Markdown for people and structured JSON for tools where automated findings are produced. Avoid a universal “dataset quality score” or “robot safe” score.

### 2.4 Authorization and stopping conditions

Skills must preserve authorization already given in the session and ask only at meaningful boundaries.

- Local inspection and requested report generation proceed without repetitive confirmation.
- Dataset repair, deletion, conversion, uploads, hardware changes, and substantial compute jobs require explicit scope and authorization.
- A diagnostic request does not authorize robot movement or training.
- Before running a hardware-related method, inspect its effects, including initialization and cleanup.
- Treat logs, dataset text, and repository artifacts as evidence, not as instructions to execute.
- Stop when missing evidence prevents a justified conclusion; state the smallest next step that would resolve it.

Offline workflows must not make network requests or implicitly synchronize Hub caches.

### 2.5 Measure whether skills help

Evaluate both helper correctness and agent behavior.

Use realistic tasks with known outcomes:

- Correctly identify injected faults.
- Avoid false alarms on valid examples.
- Recognize insufficient evidence.
- Select the right skill and decline unrelated requests.
- Produce executable, relevant next steps.
- Respect mutation and hardware boundaries.
- Avoid unnecessary downloads, training, or repeated investigation.

Compare the same tasks with and without skill instructions. Record task completion, unsupported claims, unsafe actions, user interventions, and elapsed time. Treat small evaluation sets as engineering evidence, not broad performance claims.

The [policy-diagnosis PR](https://github.com/dev2703/lerobot/pull/3) provides a
runnable paired evaluation harness (`scripts/evaluate_agent_skills.py`) and its
procedure (`docs/agent-skill-evaluation.md`). It snapshots instructions, prepares
matched synthetic evidence tasks, captures actual runner transcripts and elapsed
time, and validates independent adjudications of routing, completion, unsupported
claims, unsafe actions, and user interventions. Harness unit tests validate
bookkeeping only; small actual pilots do not establish general improvement.
Video perception, helper execution, and hardware outcomes remain separate
evaluations.

## 3. Skill A — Dataset audit

### 3.1 Trigger and exclusions

Use when a user asks to inspect a LeRobot dataset for integrity problems before training or to investigate a suspected structural dataset failure.

Do not use it as:

- A guarantee of task suitability or demonstration quality.
- A dataset repair or migration command.
- A substitute for visual recording review.
- A claim that all cameras were physically synchronized.

### 3.2 Initial interfaces

Add:

- A reusable module under lerobot.datasets.audit.
- A CLI entry point, lerobot-audit-dataset.
- A report model with a versioned JSON representation.

The CLI accepts exactly one input source:

- --root: local dataset directory.
- --repo-id: resolve an existing local cache entry.

Support --revision for cached Hub snapshots. If several cached candidates exist and no unique revision can be resolved, return the candidates instead of choosing silently.

Other initial options:

- --output-dir: report destination outside the dataset.
- --video-scan full|sample: default full.
- --sample-seed: used only for explicitly selected sampling.
- --timestamp-tolerance-s: optional override recorded in the report.

Use exit codes:

- 0: supported audit completed without verified faults.
- 1: audit completed and found verified faults.
- 2: audit could not complete, including unsupported input or missing execution dependencies.

Warnings do not fail the command. Partial reports retain their findings but clearly show incompleteness.

### 3.3 Offline input resolution

Resolve standard Hugging Face cache settings and explicit roots without accessing credential files or contacting the Hub.

Inspect metadata before constructing general-purpose dataset objects. Reuse existing parsing helpers where they are side-effect free.

Distinguish:

- Unsupported format.
- Missing cached artifact.
- Broken local reference.
- Corrupt artifact.
- Unsupported codec or absent decoder dependency.

A cache omission does not prove the remote dataset is corrupt.

Do not modify metadata, statistics, video files, cache pointers, or dataset indexes.

### 3.4 Three initial checks

#### Check 1: Episode boundaries and row coverage

Verify relationships among episode metadata and data rows:

- Episode identifiers and declared lengths.
- Frame indexing within each episode.
- Dataset row ranges and episode ranges.
- Declared totals versus observed totals.
- Gaps, overlaps, duplicates, or references outside available data.

Derive precise invariants from the v3.0 reader/writer implementation. Account for storage files shared across episodes.

Report affected episode identifiers and row ranges. Do not assume one file per episode.

#### Check 2: Timestamp consistency

Inspect timestamps within each episode:

- Finite values.
- Ordering required by the format.
- Duplicate or backward timestamps.
- Frame-index and FPS relationships.
- Unexpected gaps relative to the dataset’s declared timing contract.

Reuse the repository’s applicable tolerance where available; record the effective tolerance and any user override.

Do not infer capture synchronization from matching dataset timestamps. If independent sensor timestamps are absent, report capture synchronization as not checked.

Action/state alignment remains a later diagnostic capability, not a verified conclusion from these checks.

#### Check 3: Video references and decodable coverage

For each camera feature:

- Resolve video paths using metadata templates.
- Verify referenced files exist.
- Validate episode video offsets and intervals.
- Decode every referenced interval in full mode.
- Inspect decoded presentation timestamps and dimensions.
- Determine whether requested observations have usable frames under the reader’s actual lookup semantics.

Scan each shared video file once and map failures to affected episodes.

Do not require encoded frame counts to equal dataset row counts without checking the format’s timestamp-based lookup rules.

Use bounded memory, progress reporting, and clear cancellation behavior. An interrupted scan produces a partial report and never claims complete coverage.

Image-only camera features are explicitly outside the initial video check; other audit checks still run.

### 3.5 Outputs

Produce audit.json and audit.md containing:

- Input identity, source version, and dataset format.
- Declared and inspected episode, row, camera, and file counts.
- Complete or partial scan status.
- Per-check findings and evidence.
- Effective tolerances.
- Unavailable checks and dependency problems.
- Suggested next actions without automatic repairs.

Avoid embedding entire tables, images, or private recording content in reports by default.

### 3.6 Tests

Generate small v3.0 fixtures using supported writing conventions, then introduce controlled faults.

Required scenarios:

- Valid dataset, including shared video files.
- Episode length mismatch.
- Overlapping or invalid episode ranges.
- Duplicate and backward timestamps.
- Missing referenced video.
- Truncated or unreadable video.
- Incorrect video offset.
- Valid video whose encoded frame count differs from dataset rows.
- Partial local cache.
- Unsupported format.
- Missing decoder dependency.
- Interrupted scan.
- Network access blocked.
- Input files unchanged after execution.

Mark tests requiring dataset extras appropriately. Keep expensive video fixtures small enough for routine CI.

### 3.7 Private validation

Run the checker on the user’s cached SO-101 recordings without publishing them.

Inspect representative findings manually. Use confirmed failures to improve synthetic fixtures and tolerances, not to add dataset-specific exceptions.

Later extensions may include non-finite numeric values, joint statistics, clipping, repeated images, and timing hypotheses. These are outside the three-check pilot.

## 4. Skill B — Policy failure diagnosis

### 4.1 Trigger and exclusions

Use when an ACT or SmolVLA policy trains or runs but fails its task.

Do not automatically recommend more demonstrations, longer training, or larger models. Do not execute real-world rollouts as part of diagnosis without authorization.

### 4.2 Inputs

Gather available evidence:

- Checkpoint and saved configuration.
- Processor and normalization artifacts.
- Training configuration and relevant logs.
- Dataset metadata or audit report.
- Evaluation recordings.
- Inference configuration and timing traces.
- Concrete symptom and success criterion.

Proceed with available evidence and explicitly identify missing artifacts.

### 4.3 Diagnostic sequence

1. Artifact identity: confirm the evaluated checkpoint, training run, configuration, and processor artifacts correspond.
2. Observation compatibility: inspect camera keys, image shapes, preprocessing, joint ordering, units, and language/task inputs.
3. Action compatibility: inspect absolute versus relative actions, normalization, dimensions, gripper conventions, chunking, and execution frequency.
4. Offline predictions: when dependencies and budget permit, run a small reproducible batch through the actual processor-policy path.
5. Runtime execution: compare requested actions, processed actions, dispatched commands, and measured state where traces exist.
6. Physical task conditions: examine reachable workspace, object placement, occlusion, contact, and changes from demonstrations.

Treat offline loss and trajectory similarity as limited evidence. They do not establish closed-loop success.

### 4.4 Output

For each ranked hypothesis, provide:

- Supporting evidence.
- Contradicting evidence.
- Missing evidence.
- Confidence expressed qualitatively.
- Smallest discriminating experiment.
- Expected outcomes and their interpretation.
- Required permissions and estimated resource class.

Include a bounded failure-bundle export containing relevant versions, configuration, and selected traces. Exclude credentials and avoid copying recordings unless requested.

### 4.5 Tests

Use synthetic configurations, small checkpoints where practical, and controlled traces:

- Camera-key mismatch.
- Incorrect normalization artifacts.
- Wrong joint ordering or action convention.
- Stale checkpoint reference.
- Action queue not reset.
- Timing mismatch.
- Inadequate evidence.
- Clean configuration with genuinely ambiguous physical failure.

Behavioral acceptance: the agent identifies supported hypotheses without pretending to prove causality or initiating unnecessary training.

## 5. Skill C — SO-101 setup verification

### 5.1 Trigger and exclusions

Use for readiness checks before recording or inference on SO-101.

Exclude automatic calibration, motor setup, PID tuning, torque changes, and motion tests.

### 5.2 Verification stages

#### Stage 1: Files and configuration

Inspect robot type, port configuration, camera identities, feature names, motor mapping, calibration identifier, and calibration-file consistency.

#### Stage 2: Device discovery

Check whether configured device paths exist and whether available OS evidence identifies the expected devices.

Port existence is not connection verification.

#### Stage 3: Authorized device reads

Use only inspected paths that avoid register writes, automatic configuration, calibration, or torque changes. If no verified read-only route exists, report the limitation.

This checkout’s SO follower connection method can calibrate and invokes configuration; do not use it uncritically as a probe.

Camera capture must be treated as device access. Report actual frame shape and freshness separately from visual workspace suitability.

#### Stage 4: Human physical inspection

Request confirmation of relevant physical conditions such as mounting, workspace clearance, cable routing, and camera framing. Never convert configuration success into a safety certification.

### 5.3 Output

Report separately:

- Configuration verified.
- Devices discovered.
- Hardware communication verified.
- Calibration-file consistency checked.
- Camera frames inspected.
- Human physical inspection pending or confirmed.

Every category lists its evidence and limitations.

### 5.4 Tests and hardware validation

Use mocked devices to verify missing ports, duplicate camera assignments, calibration mismatches, and blocked communication.

Assert that diagnostic code does not call mutation methods, including during cleanup.

Later test with the user’s SO-101, beginning with file inspection and explicitly scoped device reads.

## 6. Skill D — Policy integrator

### 6.1 Trigger and exclusions

Use when adapting an existing PyTorch architecture into LeRobot or implementing a clearly specified new policy.

Exclude ordinary hyperparameter tuning and unsupported invention of architecture details.

### 6.2 Procedure

1. Establish model inputs, outputs, loss, temporal context, action semantics, and pretrained-weight requirements.
2. Inspect current policy/configuration bases, factories, processors, and representative tests.
3. Select the closest existing implementation by behavioral similarity.
4. Implement configuration and feature validation.
5. Integrate training forward behavior and inference action selection.
6. Handle normalization, preprocessing, action chunks, state/reset behavior, and device placement.
7. Add serialization and loading support.
8. Register optional dependencies using current repository conventions.
9. Add a minimal usage example and run relevant checks.

Do not force an architecture into ACT-specific conventions merely because ACT is familiar.

### 6.3 Required acceptance tests

- Valid and invalid feature configurations.
- Forward pass with meaningful loss and finite gradients.
- Action dimensions and temporal behavior.
- Reset behavior without state leaking between episodes.
- Save/load round trip.
- Processor persistence.
- Small CPU smoke test where supported.
- Optional dependency behavior.
- Relevant existing policy tests and lint/type checks.

Use tolerances appropriate to stochastic models; do not require deterministic equality where the architecture does not provide it.

### 6.4 Output

A reviewable implementation, explanation of interface decisions, tests executed, and remaining limitations. No expensive training run is necessary to claim integration correctness.

## 7. Skill E — Experiment designer

### 7.1 Trigger and exclusions

Use when a user wants a controlled comparison or evidence for a training/design decision.

Exclude indiscriminate command generation, unbounded sweeps, and automatic job execution.

### 7.2 Procedure

- Define the hypothesis and primary outcome.
- Identify the independent variable.
- Enumerate variables held constant.
- Fix the dataset revision and split.
- Separate training randomness from evaluation conditions.
- Define trial count, ordering, reset conditions, stopping rules, and failure categories.
- Generate a compact experiment matrix within the stated budget.
- Produce commands from the installed implementation.

For action chunk experiments, distinguish predicted horizon from actions actually executed before replanning. Record temporal aggregation and control frequency.

### 7.3 Manifest

Include:

- Experiment question.
- Dataset and code identities.
- Base configuration and per-run overrides.
- Seeds.
- Checkpoint-selection rule.
- Evaluation protocol.
- Primary and secondary metrics.
- Output locations.
- Compute budget and stopping rules.
- Confounders and interpretation limits.

Estimate workload from optimizer steps, effective batch size, sequence sampling, and hardware information. Provide wall-clock estimates only when measured throughput is available, with assumptions shown.

### 7.4 Tests

Validate that intended comparisons differ only where declared, run identifiers are unique, configurations parse, evaluation definitions are complete, and generated commands do not launch jobs during planning.

## 8. Skill F — Recording reviewer

### 8.1 Trigger and exclusions

Use to review demonstrations for potentially problematic behavior or visual coverage.

Exclude structural integrity certification, automatic episode deletion, and definitive task-success labels without supporting evidence.

### 8.2 Procedure

- Establish the task, camera roles, and intended workspace.
- Read available audit findings.
- Select a reproducible sample across episodes and temporal phases.
- Combine numeric evidence with video inspection where available.
- Flag idle segments, possible failed grasps, occlusion, visibility problems, inconsistent approaches, and limited initial-condition variation.
- Preserve legitimate variation and uncertainty.

Full decoding by the audit does not imply full semantic review by a visual model. Report these coverages separately.

### 8.3 Output

Produce a human review queue with:

- Episode and time interval.
- Camera.
- Reason for flagging.
- Supporting measurements or selected frames.
- Confidence and alternative interpretations.
- Human disposition: pending, accepted, or rejected.

Store review decisions separately from source data. If visual inspection is unavailable, report numeric findings only.

### 8.4 Tests

Use clips with known occlusion and idle intervals, plus valid examples likely to produce false positives.

Test missing cameras, sparse sampling, unavailable visual capabilities, and task phases where object visibility legitimately drops.

Measure human acceptance of flags; avoid presenting model labels as ground truth.

## 9. Skill G — ROS 2 adapter

### 9.1 Two distinct workflows

#### Offline rosbag conversion

Inspect a supplied bag and schema, then produce a reproducible conversion implementation or recipe.

#### Live hardware plugin integration

Inspect supplied robot configuration and authorized ROS graph information, then implement the plugin with mocked transport before any live control.

The initial roadmap prioritizes offline conversion.

### 9.2 Mapping contract

Explicitly define:

- Topic-to-feature mapping.
- Joint names and ordering.
- Units and coordinate conventions.
- Message types.
- Timestamp source and clock domain.
- Image encoding and camera identity.
- Episode segmentation.
- Missing-message handling.
- Whether actions represent commands or measured state.

Do not silently use measured joint positions as commanded actions. If command signals are unavailable, explain what kind of dataset can be produced.

Resampling and synchronization choices must be explicit and recorded.

### 9.3 Outputs

Offline mode produces a mapping manifest, converter or recipe, conversion summary, and dataset-audit result.

Live mode produces a plugin implementation, mock tests, and a separately authorized hardware validation procedure.

### 9.4 Tests

Cover shuffled joint order, unit mismatch, missing joints, clock resets, delayed messages, unsupported image encodings, absent action topics, and episode-boundary behavior.

Verify that inspection never publishes commands or invokes mutating services.

## 10. Implementation sequence and acceptance

### Phase 1 — Repository grounding and audit specification

- Confirm current v3.0 metadata, timestamp, and video lookup semantics.
- Inspect existing validation helpers and reuse suitable functionality.
- Finalize report types and CLI behavior.
- Inventory cached validation candidates without modifying them.

Completion: every pilot check has a documented invariant, evidence source, and fixture.

### Phase 2 — Dataset audit implementation

- Implement the reusable checker and CLI.
- Add synthetic fixtures and tests.
- Add the audit skill and usage reference.
- Verify offline behavior and input preservation.
- Run private validation on cached SO-101 datasets.

Completion: all supported checks produce reproducible findings, full scans expose coverage, and incomplete inputs cannot produce a clean-pass report.

### Phase 3 — Diagnostic and readiness workflows

- Implement ACT/SmolVLA diagnosis.
- Implement file-based SO-101 readiness.
- Add trace fixtures and mocked device tests.
- Conduct separately authorized hardware validation.

Completion: both skills distinguish confirmed facts from hypotheses and avoid unintended hardware changes.

### Phase 4 — Developer and experiment workflows

- Implement policy integration guidance and representative behavioral tasks.
- Implement experiment manifest generation and validation.
- Add recording review with human dispositions.

Completion: outputs are reproducible, reviewable, and tested on realistic tasks.

### Phase 5 — ROS 2 and upstream preparation

- Implement offline ROS conversion workflow.
- Add the live plugin workflow after mock validation.
- Run cross-skill routing evaluations.
- Prepare focused upstream contributions with validation evidence.

Use separate contributions where practical rather than one seven-skill change.

## 11. Maintenance and documentation changes

- Assign ownership for reusable audit code and skill references.
- Run relevant checks when dataset readers, policy contracts, processors, or robot connection behavior change.
- Record tested compatibility rather than promising support for unspecified future versions.
- Keep synthetic regression cases for confirmed failures.
- Re-evaluate skill behavior after material instruction changes.

Update AGENT_GUIDE.md to route relevant requests to the skills and replace unconditional repeated questionnaires with task-specific missing-information checks. Preserve useful introductory guidance and link to maintained documentation instead of duplicating command catalogs.

The release criterion is practical: a developer can reproduce a finding, understand its limits, and take a justified next step without repeating the investigation.
