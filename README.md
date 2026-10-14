# IIRA: Injury-Inspired Residual Assistance

机械臂辅助卒中上肢康复的六自由度仿真算法，包含损伤模型、经典与神经控制器、IIRA 与消融、闭环患者模型和外部运动参考评测。

This repository provides research simulation code for **Injury-inspired adaptive residual learning for robot-assisted upper-limb rehabilitation after stroke: a 6-DOF simulation benchmark study**. Clinical effects have not been validated; the FMA proxy and patient adaptation parameters are synthetic assumptions. See `SPEC.md` for the model and protocol.

## Installation

Use Python 3.11 or newer in a virtual environment, from this repository root:

```bash
python -m pip install -r requirements.txt
```

## Generate synthetic data and validate

```bash
python src/arm6dof.py
python src/dataset.py
python src/check_coadapt.py
python src/smoke_test.py
```

The dataset command generates 60 virtual subjects and reference trajectories locally. It contains no real patient data. The smoke test performs a small training and closed-loop evaluation and writes generated outputs under ignored directories.

## Main experiment

```bash
python src/run_experiment.py
python src/compute_rho_realized.py
python src/tune_deployment.py
python src/run_coadapt.py --workers 4 --output results/coadapt/new_run
python src/make_coadapt_outputs.py --run results/coadapt/new_run
```

Run the main experiment first to generate fold-specific weights required by subsequent experiments. Training and evaluation are CPU-intensive. Scripts under `src/transfer/` additionally require separately obtained real datasets; those datasets and learned weights are not distributed here.

## External motion references

```bash
python src/prepare_cmu_motion.py --workers 8
python src/run_cmu_generalization.py --workers 4 --output results/external_motion_cmu/new_run
```

The preparation script downloads motion references from the [CMU Motion Capture Database](https://mocap.cs.cmu.edu/). Review the source dataset's terms before obtaining or redistributing data. These motion references are combined with simulated patients and robots; they do not constitute clinical control validation.

## Code map

- `src/arm6dof.py`: kinematics, inverse dynamics, and numerical checks.
- `src/injury.py`, `src/tasks.py`, `src/dataset.py`: synthetic patients and tasks.
- `src/controllers/`: classical controllers, neural models, IIRA and ablations.
- `src/simulate.py`: shared closed-loop simulation engine.
- `src/patient_closed_loop.py`: delayed patient feedback, effort, fatigue, and adaptation.
- `src/run_experiment.py`, `src/run_coadapt.py`: training and evaluation.
- `src/transfer/`: optional external-data analysis.

## Interpretation limits

The experiments do not establish universal IIRA superiority. The tuned deployment law was selected on fold-1 test patients, so its reported comparison is exploratory. Simulated participation and synthetic FMA changes should not be interpreted as measured clinical recovery. External motion evaluation did not satisfy the prespecified mean tracking error requirement.

## Public release contents

Only Python source, technical specifications and setup documentation are included. Personal contact details, machine-specific paths, real participant data, original manuscripts, patent documents, trained weights, experiment outputs and prior Git history are excluded. Generated data and outputs are ignored by Git. No software license is granted by this initial publication.
