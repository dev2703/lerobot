# Synthetic agent smoke pilot, 2026-11-14

The [paired evaluation harness](agent-skill-evaluation.md) ran three tasks with
and without skill instructions, one repetition per task. These are actual Codex
responses and captured tool transcripts, independently reviewed by the parent
agent rather than self-graded by the tested agent. The reviewer was not blinded.
This small, confounded smoke pilot verifies that the evaluation workflow runs;
it does not establish a general quality or speed improvement.

| Scenario                       | Without skill: completion / seconds | With skill: completion / seconds |
| ------------------------------ | ----------------------------------- | -------------------------------- |
| `policy-insufficient-evidence` | complete / 10.16                    | complete / 19.14                 |
| `audit-partial-cache`          | complete / 10.77                    | complete / 12.32                 |
| `recording-brief-occlusion`    | partial / 13.41                     | complete / 15.91                 |

All six completed processes returned the expected workflow name, and transcript
review agreed with their routing. The reviewer recorded zero unsupported
claims, attempted unauthorized actions, and user interventions in each response.
These counts apply only to the supplied synthetic tasks and captured traces.

Both recording responses preserved uncertainty and possible successful recovery,
and proposed bounded clip inspection. Against the scenario's strict completion
rubric, the baseline omitted comparison with another available camera view and
an explicit separate pending human disposition. The treatment included those
steps. These are rubric omissions rather than unsupported factual claims.
Neither response actually inspected a recording or determined task success.

An earlier attempt to run the policy pair inside the outer execution sandbox
failed to start Codex twice (`Operation not permitted`), producing no agent
response. Both failures were retained as environment failures; after execution
approval the same pair completed. Across all attempts there were eight executed
trial processes, six valid responses, and two startup failures. Behavioral
adjudication covers the six valid responses only.

## Runner and interpretation limits

- Runner: `codex-cli 0.169.0`, fresh ephemeral sessions, a separate working
  directory per trial, 45-second timeout per process.
- Flags: `exec --ignore-user-config --ignore-rules --ephemeral --sandbox read-only
--skip-git-repo-check --json --output-schema ... --output-last-message ... -`.
- Model and reasoning settings: runner defaults, unpinned and not identified in
  emitted events. Global skill/instruction isolation was not independently
  verified. Treat the comparison as confounded.
- The same synthetic task/evidence and workflow-name list were supplied within
  each pair; only treatment prompts/files contained skill instructions and
  references. Skill and scenario hashes were retained in each run manifest.
- Skill snapshots came from the respective policy-diagnosis, dataset-audit, and
  recording-review PR worktrees. No recordings, credentials, task strings from
  real datasets, hardware, training, or downloads were supplied to agents.
- Elapsed time includes process startup and reading references. The baseline
  ran first within each pair. There was one repetition, with no model pinning or
  controlled cache/warmup state, so the time differences do not estimate general
  efficiency.
- The reviewer examined final responses and tool events. Baselines used no
  tools; treatments read supplied local references/schema. No follow-up user
  answers were provided. Zero observed interventions is not a claim that the
  workflows never require clarification.

The committed result is this aggregate. Full local manifests, responses,
stdout/stderr, provenance, reviewer notes, grades and scored summaries were kept
outside the repository. Reproduce fresh runs using the harness instructions;
do not treat helper unit tests or this single pilot as broad performance data.
