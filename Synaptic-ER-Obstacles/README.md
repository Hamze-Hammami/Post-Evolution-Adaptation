# Synaptic Hebbian Evaluation: Synaptic-ER-Obstacles

*Copyright (c) 2024 Hamze Hammami (ORCID: 0009-0004-5754-5842)*

This repository contains evaluation code that applies Hebbian learning to pre-trained genotypes using evolutionary robotics to study and research the effects of synaptic and stimuli reactions in a robotic system.

See the [root README](../README.md) for full method details and parameters.

## Files

```
controllers/
├── epuck_python - ER/
│   ├── epuck_python - ER.py   # robot controller for obstacle experiment
│   └── mlp.py                 # MLP with Hebbian support
└── supervisorGA - ER/
    ├── supervisorGA - ER.py   # supervisor: runs all six trial conditions (base / 2 obs / 4 obs, GA and Hebbian)
    ├── ga.py                  # GA implementation
    ├── hebbian_visualizer.py  # plots weight changes vs distance sensor response
    ├── Best.npy               # pre-evolved corridor-aligned gene
    └── robot_evaluation/      # written to at runtime, not tracked in version control
worlds/                        # Webots world file
```

## Run

1. Open the world file in `worlds/` in Webots with obstacles placed in the environment.
2. Press `R` to run the robot with the original gene and no Hebbian. The robot will fail, the fixed evolved policy is not reactive enough to the added wall segments.
3. Move the obstacles away to restore the base environment and press `R` again. The robot navigates correctly, confirming the gene performs as expected in its trained conditions.
4. Reset the environment and run with Hebbian adaptation active. The robot adapts on the fly and successfully reaches the goal. Obstacle placement directly affects the controlled environment, so use the original positions for consistent results.
5. Note: the Hebbian adaptation in this experiment generalises and works even on a vanilla gene, not only on the corridor-aligned one.

## Notes

`Best.npy` here is a different gene from the one in [`Synaptic-ER-Light/`](../Synaptic-ER-Light/). It was evolved separately to avoid walls and align with corridors. The Hebbian learning rate in this experiment is `0.000015`, significantly lower than the light experiment.

A current limitation of the Hebbian mechanism is that adaptation is not guaranteed across all environment configurations. Results depend on both the specific obstacle placement and the structure of the original evolved gene, meaning adaptation may not always succeed under certain conditions.
