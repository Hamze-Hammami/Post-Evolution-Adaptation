# Why Evolve When You Can Adapt?

**Post-Evolution Adaptation of Genetic Memory for On-the-Fly Control**

A zero-shot adaptation mechanism for evolutionary robotics that merges a standard Genetic Algorithm (GA) controller with online Hebbian plasticity. The genotype acts as fixed "genetic memory" while Hebbian updates provide live, temporary adaptation during runtime. The robot's own fitness function is reused as a neuromodulation signal that scales the learning rate, so the controller can rewire its synaptic weights on the fly with no extra training. After each trial the robot reverts to its original weights, preserving the evolved core behaviour.

Presented at **ALIFE 2025** (Kyoto / Online), MIT Press.
Paper: https://direct.mit.edu/isal/proceedings/isal2025/37/15/134065

---

## The Idea

GA controllers perform well in static environments, but a fixed evolved policy can fail when the environment changes, often requiring full and costly retraining. This work keeps the evolved genome fixed and adds a plastic layer on top of it, an instance of the Baldwin effect where lifetime learning influences performance without altering the inherited memory.

The system runs in two phases:

1. **Evolution (memory).** A GA optimises the baseline MLP weights in a fixed environment. These weights become the genetic memory.
2. **Post-evolution adaptation (learning).** During runtime the Hebbian rule adjusts the weights in response to unfamiliar conditions, scaled by live fitness. When the trial ends, the temporary changes are discarded and the robot returns to its original genome.

---

## Repository Structure
 
| Folder | Stage | Description |
|---|---|---|
| [`ER/`](./ER/) | Evolution | Evolutionary robotics training and evaluation. Produces the original gene (baseline genotype) for the T-maze task. |
| [`Synaptic-ER-Light/`](./Synaptic-ER-Light/) | Experiment 1 | Light-sensitive experiment. The GA gene is trained in high luminosity, then evaluated under low luminosity with and without Hebbian adaptation. |
| [`Synaptic-ER-Obstacles/`](./Synaptic-ER-Obstacles/) | Experiment 2 | Obstacle-avoidance experiment. A corridor-aligned gene is evaluated against added obstacles, GA alone versus GA plus Hebbian. |
 

---

## Method

### Platform
- Simulator: Webots
- Robot: e-puck
- Environment: T-shaped maze
- Task: at the junction, turn right if light is present, turn left if light is absent. 


### Genetic Algorithm
Genotype is vector of MLP weights

Fitness combines behaviour terms with a goal reward, with collision avoidance weighted highest:

```
forwardFitness        = (vLeft + vRight) / 1.5
avoidCollisionFitness = 1 - (max(proximity) / 3)
spinningFitness       = 1 - |vRight - vLeft| / 2
junctionFitness       = 1 if correct turn, else 0

combinedFitness = (forward + 2*avoid + spin + junction) / 5

dist         = euclidean distance to goal
reward       = 1 - min(1, (1.7 * dist)^3)
finalFitness = (combinedFitness + reward) / 2
```

### Post-Evolution Hebbian Adaptation
The fitness value F is reused as a neuromodulation signal that scales the Hebbian learning rate, so a strong performer reinforces active pathways faster while a weak performer adapts slowly but never fully stops (the floor is 20% of the base rate). A decaying trace and weight clipping keep the otherwise unstable Hebbian rule bounded:

```
Ne     = N * max(0.2, F)                        # effective rate, N = base rate
C      = outer(pre_activation, post_activation)  # correlation matrix
T(i+1) = (Tdecay * T(i)) + (Tupdate * C)         # Tdecay = 0.95, Tupdate = 0.05
dW     = Ne * T(i+1)
W(t+1) = clip(W(t) + dW, -Wmax, Wmax)            # Wmax = 2.0
```

---

## Experiments

### Experiment 1: Light (`Synaptic-ER-Light/`)
The gene is trained in high luminosity (`TexturedBackgroundLight` = 1.0), then luminosity is dropped to 0.1 at test time. The GA-only controller drifts because the light sensors receive unfamiliar inputs, while the Hebbian-adapted controller nearly reproduces the original trajectory in the much darker environment.

| Condition | Reaches goal | Notes |
|---|---|---|
| High light, GA | yes | |
| Low light, GA | no | |
| Low light, GA + Hebbian | yes | |
| High light, GA + Hebbian | no | limitation |

The genotype is identical across the first three conditions, only the lighting differs. The base rate here is 0.002, which is high. Applying adaptation in the already-familiar bright environment causes instability, a current limitation: the rate should ideally scale with the degree of environmental novelty and switch off when nothing has changed.

### Experiment 2: Obstacles (`Synaptic-ER-Obstacles/`)
A different gene, evolved to avoid walls and align with corridors, is tested with 2 and 4 added wall segments tightening the junction. GA alone crashes or stalls because the fixed policy does not react quickly enough. With a much lower learning rate of 0.000015, Hebbian adaptation reaches the goal in every case, with no extra training. Trajectory overlays use CoTracker (Meta).

| Environment | GA | GA + Hebbian |
|---|---|---|
| Base | success | success |
| +2 obstacles | fail | success |
| +4 obstacles | fail | success |

---

## Requirements
- Webots (e-puck model)
- Controllers run inside Webots (Python 3.x with NumPy; adjust to your setup)

Open the world file inside each folder in Webots and run the simulation.

---

## Citation

```bibtex
@inproceedings{hammami2025adapt,
  title     = {Why Evolve When You Can Adapt? Post-Evolution Adaptation of Genetic Memory for On-the-Fly Control},
  author    = {Hammami, Hamze and Barbulescu, Eva Denisa and Shaikh, Talal and Aldada, Mouayad and Munawar, Muhammad Saad},
  booktitle = {Artificial Life Conference Proceedings (ALIFE 2025)},
  year      = {2025},
  publisher = {MIT Press}
}
```
