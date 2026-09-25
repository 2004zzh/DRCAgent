# DRC Agent

DRC Agent is a multi-region, physically verified DRC repair system for ASAP7 layouts. Each Region Agent combines the current rule witness, executable degrees of freedom, neighborhood relations, an experience blueprint, and the initial layer-specific EvoDRC skill to select a repair strategy. A deterministic Repair Kernel converts that strategy into legal geometry edits. Selected candidates are committed only after fresh KLayout DRC, sanity, and connectivity checks pass.

This release provides three reproducible experiment variants:

- `complete`: the complete method, with the Experience Graph and all three types of inter-region relations and message passing enabled.
- `without-experience`: disables only the Experience Graph while preserving the rest of the execution path.
- `without-region-messaging`: preserves the Experience Graph, Repair Kernel, candidate graph, and CP-SAT joint selection, but disables geometry, shared-net, and resource edges as well as inter-region message passing.

## Platform requirements

- Linux x86_64.
- CPython 3.12. The core modules are distributed as CPython 3.12 native `.so` extensions and cannot be loaded by another Python version or platform.
- Docker, with permission for the current user to run `docker` commands.
- At least 16 CPU cores, 32 GB RAM, and 60 GB of available disk space are recommended. Reserve additional resources for concurrent experiments.
- Network access to the SiliconFlow API.

The included experiment profiles support SiliconFlow only. The API key is read from the `SILICONFLOW_API_KEY` environment variable. Never store a key in YAML files, scripts, logs, or Git.

The default production model is `deepseek-ai/DeepSeek-V4-Flash`, with thinking disabled and no model fallback.

KLayout does not need to be installed on the host. Physical verification runs inside the `drc-benchmark-repair:latest` Docker image. The image is built from the DAC26 benchmark's `Dockerfile.repair` and contains KLayout 0.30.1.

## 1. Clone the repository and benchmark

Use a recursive clone so that EvoDRC and its nested DAC26 benchmark submodule are checked out together:

```bash
git clone --recursive <YOUR_DRCAgent_REPOSITORY_URL> DRCAgent
cd DRCAgent
git submodule update --init --recursive
```

If `benchmarks/EvoDRC/DAC26_DRC_Benchmark` is empty after a non-recursive clone, run the final command above. The benchmark does not need to be copied manually.

## 2. Create the Python environment

```bash
conda create -n drcagent python=3.12 -y
conda activate drcagent
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the binary runtime and the three experiment profiles. The commands below use Block1 as an example:

```bash
python -c "import drc_agent; from drc_agent.workflow.runtime import ResearchRuntime; print('binary runtime import: OK')"

PYTHONPATH=src:. python -m drc_agent.cli inspect-config \
  --config configs/base.yaml \
  --experiment-config configs/experiments/release/complete/Block1.yaml

PYTHONPATH=src:. python -m drc_agent.cli inspect-config \
  --config configs/base.yaml \
  --experiment-config configs/experiments/p5_ablations_20260917/block1_without_experience.yaml

PYTHONPATH=src:. python -m drc_agent.cli inspect-config \
  --config configs/base.yaml \
  --experiment-config configs/experiments/p5_ablations_20260917/block1_without_region_messaging.yaml
```

## 3. Build the KLayout repair image

Run the following commands from the repository root:

```bash
cd benchmarks/EvoDRC/DAC26_DRC_Benchmark
docker build -f Dockerfile.repair -t drc-benchmark-repair:latest .

cd ..
docker build -t drc-benchmark-repair:latest - < Dockerfile.evodrc

cd ../..
docker image inspect drc-benchmark-repair:latest >/dev/null
```

The second build extends the image with dependencies required by EvoDRC. DRC Agent uses only the KLayout and evaluator components in the container. Claude, Cursor, the Codex CLI, and the EvoDRC experiment entry point are not required.

## 4. Configure SiliconFlow

Set the key without writing it to the repository or shell history:

```bash
read -rsp "SiliconFlow API key: " SILICONFLOW_API_KEY
echo
export SILICONFLOW_API_KEY
```

If the server can reach SiliconFlow directly, remove any proxy variables inherited from a temporary SSH tunnel:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

## 5. Run the complete method

The following command starts Block1 from the official baseline with seed 1101 and a maximum of five iterations:

```bash
PYTHONPATH=src:. python tools/run_experiment.py \
  --variant complete \
  --case Block1 \
  --run-id complete-block1-s1101-r1
```

Replace `Block1` with `Block2` through `Block6` to run another benchmark case. Concurrent processes must use distinct run IDs. To avoid provider account-level rate limits, use a separate SiliconFlow account for each concurrent process.

Example background launch:

```bash
nohup env PYTHONPATH=src:. SILICONFLOW_API_KEY="$SILICONFLOW_API_KEY" \
  python tools/run_experiment.py \
  --variant complete --case Block1 --run-id complete-block1-s1101-r1 \
  > /tmp/complete-block1-s1101-r1.log 2>&1 &
```

The API key is passed only through the child process environment and is not written to project files.

## 6. Run the ablation experiments

Disable the Experience Graph:

```bash
PYTHONPATH=src:. python tools/run_experiment.py \
  --variant without-experience \
  --case Block1 \
  --run-id without-experience-block1-s1101-r1
```

Disable inter-region edges and message passing:

```bash
PYTHONPATH=src:. python tools/run_experiment.py \
  --variant without-region-messaging \
  --case Block1 \
  --run-id without-region-messaging-block1-s1101-r1
```

Both ablations still use the real LLM, Repair Kernel, KLayout, sanity and connectivity checks, candidate graph, CP-SAT, and master acceptance. They do not use a mock or simplified evaluator.

## 7. Resume and inspect a run

After a process is interrupted, resume the original run ID from its latest safe checkpoint:

```bash
PYTHONPATH=src:. python tools/run_experiment.py \
  --variant complete \
  --case Block1 \
  --run-id complete-block1-s1101-r1 \
  --resume
```

Before resuming an interrupted run that has not reached its configured iteration limit, verify its checkpoint, inputs, source identity, and configuration without invoking the LLM or EDA:

```bash
PYTHONPATH=src:. python tools/run_experiment.py \
  --variant complete \
  --case Block1 \
  --run-id complete-block1-s1101-r1 \
  --resume --verify-only
```

To continue a completed bounded run under the same run ID, explicitly raise its total iteration limit. The current continuation protocol permits at most five total iterations. For example, verify and then extend a one-iteration run to five total iterations:

```bash
PYTHONPATH=src:. python tools/run_experiment.py \
  --variant complete \
  --case Block1 \
  --run-id complete-block1-s1101-r1 \
  --resume --verify-only --extend-max-iterations 5

# Remove --verify-only only after verification succeeds.
PYTHONPATH=src:. python tools/run_experiment.py \
  --variant complete \
  --case Block1 \
  --run-id complete-block1-s1101-r1 \
  --resume --extend-max-iterations 5
```

Read the current run status without modifying it:

```bash
PYTHONPATH=src:. python -m drc_agent.cli health-summary \
  --project-root . --run-id complete-block1-s1101-r1

PYTHONPATH=src:. python -m drc_agent.cli summarize-run \
  --project-root . --run-id complete-block1-s1101-r1 --json
```

## 8. Additional synthetic cases

`benchmarks/extra/all_cases/` contains the layout Python sources and connectivity descriptions for additional synthetic cases. These files are not presented as ready-to-run official benchmark profiles.

Before connecting an additional case to the production workflow, generate its GDS and use the same `drc-benchmark-repair:latest` image and ASAP7 rule deck to create its baseline DRC JSON. All four inputs must be available:

- layout script;
- GDS file;
- DRC report;
- connectivity JSON.

Do not modify the benchmark evaluator or rule deck to accommodate a result.

The local release copy retains `gcd.json`, `sasc.json`, `simple_spi.json`, `usb_phy.json`, and `all_cases.tar.gz`, but these large files are excluded from GitHub by default and must be distributed separately when needed. The smaller Synth1–Synth5 source files and descriptions can be committed normally.

## 9. Outputs and reproducibility boundaries

- Each experiment writes to `runs/<run-id>/`, including the resolved configuration, manifest, checkpoints, LLM and EDA evidence, and final snapshot.
- `runs/`, `tmp/`, logs, temporary EDA workspaces, and credential files are excluded by `.gitignore`.
- Each Experience Graph online branch is run-local and cannot contaminate another run.
- The EvoDRC submodule supplies both the official benchmark and the initial layer-specific skills enabled by default. Keep the pinned submodule commit unchanged for reproducible inputs.
- The bundled native extensions target Linux x86_64 and CPython 3.12. A maintainer must build new binaries to support another operating system, CPU architecture, or Python ABI.
