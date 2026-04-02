# ablation branch

This branch contains the implementation used for the following two conditions in the paper:

**Interruptible Multi-Agent Debate: Sentence-Level Disclosure and Urgency-Based Turn-Taking for Early Error Correction**

- **Dynamic order**
- **Proposed framework**

## What this branch corresponds to

This branch is used for the experiments where speaker selection is dynamic.

The exact condition is controlled by `config.yaml`.

## Condition switch

In this branch, the key switch is:

```yaml
enable_interruption: false
```

or

```yaml
enable_interruption: true
```

Interpretation:

- `enable_interruption: false`  
  → **Dynamic order**

- `enable_interruption: true`  
  → **Proposed framework**

## Main files

The following files are the main entry points for the paper experiments in this branch:

- `run_mmlu.py`  
  Main script for running the MMLU-based evaluation.

- `config.yaml`  
  Configuration file controlling the debate behavior.

- `src/`  
  Core implementation for dynamic speaker selection and related debate logic.

- `Initial_answer/.../initial_pool.jsonl`  
  Pre-generated initial answer pool used to align initial conditions across compared settings.

- analysis scripts  
  Scripts used to summarize outputs and inspect paper-relevant results.

## Experimental role of this branch

Use this branch to reproduce:

1. **Dynamic order**  
   Dynamic speaker selection without interruption.

2. **Proposed framework**  
   Dynamic speaker selection with interruption enabled.

## Initial condition settings

The paper evaluates controlled initial-answer settings such as:

- **two incorrect, one correct**
- **one incorrect, two correct**

Please configure the initial incorrect-agent condition through the branch-specific settings used in your experiments.

## Typical workflow

1. Checkout this branch:
   ```bash
   git checkout ablation
   ```

2. Review the configuration:
   ```bash
   cat config.yaml
   ```

3. Choose the condition:

   - For **Dynamic order**:
     ```yaml
     enable_interruption: false
     ```

   - For **Proposed framework**:
     ```yaml
     enable_interruption: true
     ```

4. Adjust the remaining settings as needed:
   - initial incorrect-agent condition
   - model-related settings
   - token budget or runtime parameters if needed

5. Run the experiment:
   ```bash
   python run_mmlu.py
   ```

## Recommended interpretation of this branch

This branch is the main experiment branch for the paper's dynamic-turn conditions.

Use this branch when your goal is to understand or reproduce the difference between:

- message-level dynamic speaker selection without interruption
- interruption-enabled debate with finer-grained control

## Suggested explanation for readers

A simple way to explain this branch is:

> `ablation` contains the code for both Dynamic order and the Proposed framework. The difference between the two is controlled by `enable_interruption` in `config.yaml`.

## Notes for readers

This branch is intentionally focused on the experiment implementation.

For the repository landing page and overall mapping of paper conditions to branches, please see the `main` branch.

## Suggested cleanup policy for this branch

For reader-facing release, it is recommended to keep the following clearly separated:

### Keep
- experiment code required to run Dynamic order / Proposed framework
- configuration files
- prompt-related files used in the paper
- initial answer pool files required for the controlled setup
- analysis scripts that are directly relevant to the paper results

### Move or remove
- temporary outputs
- backups
- exploratory or abandoned scripts
- caches, logs, and large generated artifacts

If multiple analysis scripts are kept, organizing them under a dedicated `analysis/` directory is recommended.

## Recommended branch message

If you reference this branch in the paper or repository documentation, you can describe it as:

> `ablation` contains the implementation for Dynamic order and the Proposed framework, controlled by the interruption setting in `config.yaml`.

