# ER

GA training phase. A Genetic Algorithm optimises the MLP controller weights over 30 generations in a fixed T-maze environment. The best gene produced here (`Best.npy`) is the baseline genotype carried into the Synaptic-ER experiments.

See the [root README](../README.md) for full method details and parameters.

## Files

```
controllers/
├── epuck_python - ER/
│   ├── epuck_python - ER.py   # robot controller: reads sensors, runs MLP, drives motors
│   └── mlp.py                 # MLP implementation
└── supervisorGA - ER/
    ├── supervisorGA - ER.py   # supervisor: manages the GA loop, evaluates fitness, saves best gene
    ├── ga.py                  # GA: population init, selection, crossover, mutation
    └── Best.npy               # saved best-performing genotype (output of training)
worlds/                        # Webots world file
```
