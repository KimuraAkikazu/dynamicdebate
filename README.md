# dynamicdebate

This repository contains the code and documentation for the paper:

**Interruptible Multi-Agent Debate: Sentence-Level Disclosure and Urgency-Based Turn-Taking for Early Error Correction**

## Overview

This repository accompanies our paper on interruptible multi-agent debate (MAD) for early error correction under a shared public-token budget.

The paper compares the following three discussion settings:

1. **Fixed order**
2. **Dynamic order**
3. **Proposed framework**

The experiments were conducted on MMLU with **3 agents** and a **public utterance token budget of 500**.

## Important note about branches

This repository uses different branches for the experimental conditions reported in the paper.

- **main**  
  Landing page for the paper repository.  
  This branch is intentionally kept lightweight and serves as the entry point for readers and reviewers.

- **round-robin**  
  Code used for the **Fixed order** condition.

- **ablation**  
  Code used for both:
  - **Dynamic order**
  - **Proposed framework**

In the `ablation` branch, the setting is controlled by `config.yaml`:

- `enable_interruption: false`  
  → **Dynamic order**
- `enable_interruption: true`  
  → **Proposed framework**

## Mapping from paper conditions to repository branches

| Paper condition | Branch | Main switch |
|---|---|---|
| Fixed order | `round-robin` | fixed speaker order |
| Dynamic order | `ablation` | `enable_interruption: false` |
| Proposed framework | `ablation` | `enable_interruption: true` |

## Initial answer settings

To control the initial debate state, we use pre-generated initial answer pools.

The experiments in the paper evaluate two controlled settings:

- **two incorrect, one correct**
- **one incorrect, two correct**

Please configure the number and identity of incorrect agents in the branch-specific `config.yaml` used for each run.

## Repository contents relevant to the paper

The following components are relevant for reproducing the experiments:

- `run_mmlu.py`  
  Main script for running the MMLU evaluation.

- `config.yaml`  
  Experiment configuration file.

- `src/`  
  Core implementation of the debate framework.

- `Initial_answer/.../initial_pool.jsonl`  
  Pre-generated initial answer pool used to align initial conditions across settings.

- analysis scripts  
  Scripts used to summarize results and generate paper-relevant analyses.

## Recommended reading order

If you are visiting this repository for the first time:

1. Read this `README` on `main`.
2. Go to the **`round-robin`** branch for the Fixed order implementation.
3. Go to the **`ablation`** branch for Dynamic order and Proposed framework.
4. Check the branch-specific `config.yaml` before running experiments.

## How to reproduce the paper conditions

### A. Fixed order
Use the `round-robin` branch.

Example workflow:
1. Checkout `round-robin`
2. Inspect `config.yaml`
3. Set the incorrect-agent condition
4. Run `run_mmlu.py`

### B. Dynamic order
Use the `ablation` branch with:

```yaml
enable_interruption: false
```

### C. Proposed framework
Use the `ablation` branch with:

```yaml
enable_interruption: true
```

## Notes on reproducibility

- The paper compares conditions under aligned initial answers.
- The public utterance budget is fixed.
- The debate behavior depends on configuration values such as:
  - interruption setting
  - token budget
  - initial incorrect-agent setting
  - model configuration

## What is intentionally not kept on `main`

To make the repository easier to inspect for readers and reviewers, the `main` branch does **not** contain the full experiment code or generated outputs.

Please use the experiment branches instead:

- `round-robin`
- `ablation`
