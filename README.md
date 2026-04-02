# round-robin branch

This branch contains the implementation used for the **Fixed order** condition in the paper:

**Interruptible Multi-Agent Debate: Sentence-Level Disclosure and Urgency-Based Turn-Taking for Early Error Correction**

## What this branch corresponds to

This branch is used for the **Fixed order** setting described in the paper.

In this condition:

- speakers alternate in a predetermined order
- utterances are disclosed at the message level
- interruption is not used

If you are looking for:

- **Dynamic order**
- **Proposed framework**

please use the `ablation` branch instead.

## Main files

The following files are the main entry points for the paper experiments in this branch:

- `run_mmlu.py`  
  Main script for running the MMLU-based evaluation.

- `config.yaml`  
  Configuration file for the experiment.

- `src/`  
  Core implementation for the debate framework used in this branch.

- `Initial_answer/.../initial_pool.jsonl`  
  Pre-generated initial answer pool used to align initial conditions across compared settings.

## Experimental role of this branch

Use this branch to reproduce the **Fixed order** baseline in the paper.

This branch should be used when you want to run the condition where:

- the speaking order is fixed
- agents do not interrupt one another
- the number of initially incorrect agents is controlled through configuration / assigned initial answers

## Initial condition settings

The paper evaluates controlled initial-answer settings such as:

- **two incorrect, one correct**
- **one incorrect, two correct**

Please check `config.yaml` and related initial-answer settings to specify the intended condition.

## Typical workflow

1. Checkout this branch:
   ```bash
   git checkout round-robin
   ```

2. Review the configuration:
   ```bash
   cat config.yaml
   ```

3. Adjust the settings for the intended experiment:
   - initial incorrect-agent condition
   - model-related settings
   - token budget or other runtime parameters if needed

4. Run the experiment:
   ```bash
   python run_mmlu.py
   ```

## Expected use in the paper

This branch is intended for reproducing the **Fixed order** results only.

If you want to compare the following two conditions:

- **Dynamic order**
- **Proposed framework**

use the `ablation` branch, where those two settings are switched by configuration.

## Notes for readers

This branch is kept as an experiment branch rather than a repository landing page.

For the overall repository guide, paper-to-branch mapping, and high-level explanation, please see the `main` branch.

## Suggested cleanup policy for this branch

For reader-facing release, it is recommended to keep the following clearly separated:

### Keep
- experiment code required to run the Fixed order condition
- configuration files
- prompt-related files used in the paper
- initial answer pool files required for the controlled setup
- analysis scripts that are directly relevant to the paper

### Move or remove
- temporary outputs
- backup files
- exploratory scripts not required to understand or reproduce the paper
- caches, logs, and large generated artifacts

If analysis scripts are retained, it is recommended to organize them under a dedicated directory such as `analysis/`.

## Recommended branch message

If you reference this branch in the paper or repository documentation, you can describe it as:

> `round-robin` contains the implementation of the Fixed order baseline used in the paper.

