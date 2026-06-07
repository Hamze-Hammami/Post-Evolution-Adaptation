# Synaptic Hebbian Evaluation

*Copyright (c) 2024 Hamze Hammami (ORCID: 0009-0004-5754-5842)*

This repository contains evaluation code that applies Hebbian learning to pre-trained genotypes using evolutionary robotics to study and research the effects of synaptic and stimuli reactions in a robotic system.

See the [root README](../README.md) for full method details and parameters.

## Files

```
controllers/
├── epuck_python-SynapticER/
│   ├── epuck_python-SynapticER.py   # robot controller with Hebbian adaptation
│   └── hebmlp.py                    # MLP extended with Hebbian plasticity (trace decay, weight clipping)
└── Supervisor-SynapticER/
    ├── Supervisor-SynapticER.py     # supervisor: loads gene, runs GA-only and Hebbian evaluation trials
    ├── hebbian_visualizer.py        # plots weight changes vs fitness level and light sensor response
    ├── Best.npy                     # pre-evolved gene used for evaluation
    └── robot_evaluation/            # written to at runtime, not tracked in version control
worlds/                              # Webots world file
```

## Run

To fully visualise the effect of Hebbian learning on the gene, follow these steps:

1. Open the world file in `worlds/` in Webots. Set luminosity on both background lights to `1.0`.
2. Run the robot with the original gene by pressing `R`. The robot navigates correctly, this is the baseline.
3. Change luminosity on both background lights to `0.1`.
4. Press `R` to run the robot with the original gene again. The robot will fail to navigate correctly under the unfamiliar light conditions.
5. Set the Hebbian rate to `0.002` in the controller if not already set, then run the robot with Hebbian adaptation active. The robot will adapt to the new light environment.

## Notes

`Best.npy` is a pre-evolved gene. To retrain it from scratch, use the [`ER/`](../ER/) experiment.
