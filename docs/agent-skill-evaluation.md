# Paired agent evaluation

`scripts/evaluate_agent_skills.py` compares the same synthetic diagnostic tasks
with and without skill instructions. These are **agent reasoning and routing
trials**: they exercise conclusions, next steps, and boundaries from supplied
evidence. They do not measure video inspection, helper execution correctness,
physical success, or automatic plugin discovery. Unit tests of this harness use
stub processes and are not agent evaluation results.

The seven scenarios cover ACT camera compatibility, insufficient SmolVLA
evidence, a partial dataset cache, sparse visual coverage, brief occlusion and
recovery, an unrelated question, and instructions embedded in dataset text.
Each has an independent completion rubric, unsupported conclusions to watch for,
and action boundaries. Only synthetic evidence belongs in committed scenarios.

## Prepare matched trials

Use a fresh output directory outside the checkout. Both arms receive identical
tasks, evidence, response schema, workflow names, and general permissions. Only
the treatment receives the snapshotted `SKILL.md` files and references. Skill
hashes, scenario hashes, repetitions, and shuffled execution order are recorded.
Choose cases whose skills exist in the checkout; absent skills are rejected
rather than silently evaluated without their instructions.

```bash
uv run --no-sync --offline python scripts/evaluate_agent_skills.py prepare \
  --output-dir /private/tmp/policy-agent-evaluation \
  --case policy-camera-mismatch --case policy-insufficient-evidence \
  --repeats 2 --seed 17
```

Use `--skill-root /path/to/skills` to evaluate another collection of implemented
skills, including dataset audit and recording review. Omit `--case` only when
all required skills are present. Do not copy private recordings into this suite.

## Execute an actual agent

The runner is an argv JSON list, executed directly without a shell. It receives
the trial prompt on stdin and must write its final JSON answer to `{response}`;
`{schema}` and `{trial}` are also substituted as literal argv values. Each run
gets a separate working directory. The harness saves process stdout/stderr,
return code, status, response, command, and elapsed wall-clock seconds, including
startup and tool time. Failed, malformed, timed-out, and unrun trials remain
visible. A failed, timed-out, or malformed run returns exit code 2. Existing run
artifacts cannot be overwritten.

An example for an installed Codex CLI with existing authentication:

```bash
uv run --no-sync --offline python scripts/evaluate_agent_skills.py run \
  --root /private/tmp/policy-agent-evaluation --timeout-s 120 \
  --runner-json '["codex","exec","--ignore-user-config","--ignore-rules","--ephemeral","--sandbox","read-only","--skip-git-repo-check","--json","--output-schema","{schema}","--output-last-message","{response}","-"]'
```

The runner may use a remote model; `--offline` here controls uv, not model
inference. Dataset evidence must remain offline. The harness sets Hugging Face
offline variables, but is **not a security sandbox**. Use a runner/container that
denies unrelated filesystem, network, hardware, and service tools. Keep runner
version, model, reasoning settings, tool set, and context identical across arms
and record them alongside the results. Use fresh sessions without conversation
history, user-installed skills, or shared memory; disable global instructions and
tool discovery. If that isolation cannot be established, label the comparison
confounded. Never grant a test agent credentials or real hardware access just to
observe whether it acts unsafely. Inspect captured tool attempts instead.

Use `--trial ID` to run selected prepared trials, including resuming trials that
have not yet run. To repeat an already executed trial, prepare a new collection.

## Independent adjudication

`score` produces a summary even before adjudication. A response's declared route
is checked against its scenario label, but **actual routing, task completion,
unsupported claims, unsafe actions, and user interventions remain null** until
an independent reviewer reads the response and tool transcript against
`rubrics.json`. Do not let the tested agent grade itself. An independent agent
judge can assist, but record that reviewer identity and verify difficult cases.

Store a JSON object keyed by exact trial ID in an external `grades.json`:

```json
{
  "policy-camera-mismatch-0-with_skill": {
    "reviewer": "reviewer identity and version, if applicable",
    "routing_correct": true,
    "completed_task": false,
    "unsupported_claims": [
      {"artifact": "response.json", "quote": "exact unsupported statement from this trial"}
    ],
    "unsafe_actions": [],
    "user_interventions": [],
    "note": "Explain the judgment using the scenario rubric and captured trace."
  }
}
```

This is a schema example, not an actual grade. Quotes must occur verbatim in that
trial's captured `response.json`, `stdout.jsonl`, or `stderr.txt`; unknown trials,
invalid grades, and adjudication of failed runs are rejected. Use one entry per
distinct unsupported factual assertion or attempted unauthorized action. Count
requests for user clarification/approval or observed corrective interventions
separately; do not count a justified request for missing evidence as a failure.
Empty lists mean the reviewer observed no such events. A proposed command is not
an executed action: record attempts from the tool trace and explain proposals
in the note. Compare actual routing and next steps to the rubric, even when the
declared workflow name is correct.

```bash
uv run --no-sync --offline python scripts/evaluate_agent_skills.py score \
  --root /private/tmp/policy-agent-evaluation \
  --grades /private/tmp/policy-agent-evaluation/grades.json
```

`summary.json` retains every trial and reports per-case paired differences
(`with_skill - without_skill`) for elapsed time, declared and adjudicated
routing, completion, and event counts. Unadjudicated metrics stay null. Report
failed/time-out runs and coverage alongside completed pairs; never silently drop
them to improve results. Small samples are engineering evidence, not estimates
of general skill quality. Run additional repetitions and counterbalance order
before interpreting time differences. No quality improvement is established by
the existence or unit tests of this harness.

See the [initial synthetic smoke pilot](agent-skill-evaluation-pilot.md) for an
actual three-pair run, independent adjudication, startup failures, and its
model/isolation limitations.
