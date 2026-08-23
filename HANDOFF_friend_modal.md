# Handoff: launch 3 Track A concepts on a second Modal account

## Task
Launch 3 concepts in parallel on **your own Modal account** (separate from the
primary one already running Homo Sapiens/Cannabis): **Republic of Ireland**,
**Patriarchy**, **Ancient Rome**. This is Track A of a larger pipeline
(entanglement-predicts-unlearning-difficulty project) — a paper deadline is
Aug 29, real time pressure, so get these launched and left running rather
than over-investigating.

## Repo
```
git clone --recurse-submodules https://github.com/antrip03/steering.git
cd steering
git checkout kaggle
```
Submodule `pisces_ref` (a PISCES fork, branch `steering-fixes`) comes along
with `--recurse-submodules`. Nothing needs to be installed locally except the
`modal` CLI (`pip install modal`) and `gcloud`/local Python are NOT needed for
this task — the actual run happens inside Modal's own container, defined by
`track_a_feature_discovery/modal_app.py`.

## One-time setup
1. `modal token new` (or `modal setup`) — authenticate the `modal` CLI to
   your own Modal account/workspace.
2. Create the HF secret Modal needs to download the model and push results:
   ```
   modal secret create huggingface HF_TOKEN="$(cat ~/.cache/huggingface/token)"
   ```
   Run this yourself, not through an agent — it's credential material.
   **Important**: results need to land in the shared HF dataset repo
   `antrip03/pisces-track-a-runs` (private) so they merge with everything
   else already done. Your HF token needs write access there — either you've
   already been added as a collaborator on that repo, or you need to ask for
   that before running (otherwise `--push-to-hub` will fail with a 403, and
   results will only exist on your Modal container's ephemeral disk).
3. Deploy the app once (re-run this if `modal_app.py` changes):
   ```
   cd track_a_feature_discovery
   modal deploy modal_app.py
   ```

## Launch the 3 concepts

**Use `modal_spawn.py`, not `modal run --detach`.** This session hit a real,
repeated failure: a local network DNS blip (`[Errno 11001] getaddrinfo
failed`) killed two `--detach` runs outright, one 66% through a ~2h stage,
despite `--detach` supposedly protecting against local disconnection.
`modal_spawn.py` uses Modal's `.spawn()` API against the deployed app instead
— it only needs the local connection for the initial submit call, then runs
entirely independent of this machine. Verified working this session.

```
python modal_spawn.py --concept "Republic of Ireland" --layers 3,4,5,6,7,8,9,10,11,12 --reduced
python modal_spawn.py --concept "Patriarchy" --layers 3,4,5,6,7,8,9,10,11,12 --reduced
python modal_spawn.py --concept "Ancient Rome" --layers 3,4,5,6,7,8,9,10,11,12 --reduced
```

Each prints a `call_id` — save them. Check status anytime with:
```
python modal_spawn.py --status <call_id>
```
(prints "Still running." or the final exit code). `--push-to-hub` is on by
default in `modal_spawn.py` (pass `--no-push-to-hub` to disable, but don't —
that's how results get back to the shared repo).

## Cost expectations (calibrated from real runs this session, ~$1.20/hr A10G)
| Concept | Predicted effect-measurement time |
|---|---|
| Republic of Ireland | ~49 min |
| Patriarchy | ~60 min |
| Ancient Rome | ~81 min |

Plus an MMLU filtering stage on top of each (now has progress logging —
prints per-candidate, not silent). Check your own Modal billing
(`modal billing report --for today`) against your account's credit budget
before/while these run.

## What "done" looks like
Each run pushes a parquet to the hub named
`<concept_slug>__layers_3_4_5_6_7_8_9_10_11_12__reduced.parquet` (e.g.
`republic_of_ireland__layers_3_4_5_6_7_8_9_10_11_12__reduced.parquet`) at
`https://huggingface.co/datasets/antrip03/pisces-track-a-runs`. Nothing
further needs to happen on your end after that — the primary session pulls
completed results down via `track_a_feature_discovery/pull_production_results.py`.

## Known gotchas (don't rediscover these)
- If your shell's cwd is already inside `track_a_feature_discovery/`, don't
  also prefix commands with that path — causes a doubled-path
  `FileNotFoundError`.
- `discover.py` and `modal_app.py`'s heavy imports (`sae_lens`,
  `transformer_lens`) hang if run directly on a machine with no GPU — that's
  expected/known, not a bug; everything actually executes inside the Modal
  container, not locally.
- Checkpointing is automatic per concept+layers — if a run needs restarting
  for any reason, just re-run the same `modal_spawn.py` command; it resumes
  instead of starting over.

## Background context (optional reading, not required to complete the task)
`track_a_feature_discovery/README.md` has the full investigation history —
bugs found/fixed, methodology findings (PISCES's own selection criterion has
no effect-magnitude floor — documented, not something to fix, `neg_effect_score`
exists as raw data for downstream judgment), reduction validations. Not
needed to just launch these 3 runs, but useful if anything looks surprising.
