# GENREG: Evolutionary Neural Network Training

Training neural networks through trust-based evolutionary selection - no gradients, no backpropagation.

## Overview

GENREG (Genetic Regulatory Networks) is an evolutionary learning system that optimizes neural networks through population-based selection rather than gradient descent. Networks accumulate "trust" based on task performance, and high-trust genomes reproduce with mutations to create the next generation.

This repository contains benchmark results and trained models demonstrating GENREG's capabilities on visual recognition tasks.

## Results

### MNIST Digit Recognition

Architecture: 784 → 64 → 10 (50,890 parameters)
- Test Accuracy: 81.47%
- Training: 600 generations (~40 minutes)
- Method: Pure evolutionary selection, no gradients

Per-Digit Performance:
```
Digit 0: 87.1%  |  Digit 5: 70.9%
Digit 1: 94.5%  |  Digit 6: 86.3%
Digit 2: 80.7%  |  Digit 7: 82.2%
Digit 3: 73.4%  |  Digit 8: 76.7%
Digit 4: 78.2%  |  Digit 9: 79.3%
```

### Alphabet Recognition

Architecture: 10,000 → 128 → 26
- Test Accuracy: 100% (all letters mastered)
- Training: ~1,800 generations
- Task: Recognize rendered letters A-Z from visual display

## What Makes This Different

Traditional neural network training uses gradient descent to minimize a loss function. GENREG instead:

1. Maintains a population of competing neural networks (genomes)
2. Evaluates each genome on the task, accumulating "trust" for correct predictions
3. High-trust genomes reproduce, passing weights to offspring with mutations
4. Process repeats, with evolutionary pressure driving optimization

Key differences from gradient-based training:
- No loss function derivatives or backpropagation
- No learning rate, optimizer, or gradient clipping
- Population-based search explores weight space differently
- Can modify hyperparameters mid-training without restarting
- Discovers remarkably compact solutions under capacity constraints



## Quick Start

### Evaluate MNIST Checkpoint

```python
from genreg_genome import GENREGGenome
from genreg_controller import GENREGController
import pickle

# Load checkpoint
with open('checkpoints/mnist/best_genome_mnist_gen00600.pkl', 'rb') as f:
    genome = pickle.load(f)

# Test on MNIST
from torchvision import datasets, transforms
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                               transform=transforms.ToTensor())

# Run evaluation (see evaluation/mnist_eval_official_10k.py for full script)
```

### Evaluate Alphabet Checkpoint

```python
# Load checkpoint
with open('checkpoints/alphabet/best_genome_alphabet_gen01862.pkl', 'rb') as f:
    genome = pickle.load(f)

# Test on rendered letters (see evaluation/alphabet_eval.py for full script)
```

## Key Findings

### 1. Fitness Signal Stability is Critical

Early MNIST training plateaued at 65% accuracy. The breakthrough came from increasing samples per class from 1 to 20 images per digit per generation. This dramatically reduced fitness variance and enabled stable evolutionary selection.

Lesson: Evolutionary learning requires stable, consistent fitness signals. With high-variance real-world data (handwriting, photos), you must average performance over multiple samples.

### 2. Child Mutation Rate Drives Exploration

Mutation during reproduction (child mutation) is far more important than mutation of existing population (base mutation). Disabling child mutation completely flatlined learning.

Mechanism:
- Child mutation: Explores around high-trust genomes (guided search)
- Base mutation: Maintains population diversity (prevents convergence)
- Trust-based reproduction: Focuses search on promising regions

### 3. Capacity Constraints Force Efficiency

Training a 32-neuron MNIST model (784 → 32 → 10, 25,450 params) achieved 72.52% accuracy - competitive performance with half the parameters.

The 32-neuron model exhibits fascinating dynamics: it initially masters easy digits (0, 1 both >90%) while struggling on hard digits (5, 8 at ~55%). Over time, evolutionary pressure forces it to redistribute capacity, sacrificing some performance on easy digits to improve hard digits. Overall accuracy climbs even as individual digit performance shifts.

This suggests most neural networks are significantly overparameterized. Evolutionary pressure with capacity constraints reveals minimal viable architectures.

### 4. Embedding Space Reveals Learning Strategy

UMAP projections of hidden layer activations show:

32-neuron model: Cannot create sufficient separation for all 10 classes. Forced to collapse similar features, resulting in overlapping clusters for confusable digits (3/5/8).

64-neuron model: Clean 10-cluster topology with distinct regions for each digit. Errors occur primarily at decision boundaries between visually similar digits.

Visual proof that capacity directly determines representation quality and that GENREG learns genuine geometric structure through evolutionary pressure.

## Training Configuration

### MNIST

```python
# Network Architecture
INPUT_SIZE = 784              # 28×28 pixels
HIDDEN_SIZE = 64              # Hidden layer neurons
OUTPUT_SIZE = 10              # 10 digit classes

# Population & Evolution
POPULATION_SIZE = 200
MUTATION_RATE = 0.08          # Base mutation
CHILD_MUTATION_RATE = 0.07    # Child mutation (critical!)
SURVIVAL_CUTOFF = 0.2         # Top 20% survive
TRUST_INHERITANCE = 0.03      # Children inherit 3% parent trust

# Trust Dynamics
TRUST_DECAY = 0.2             # Trust *= 0.8 per generation
CORRECT_REWARD = 10.0
WRONG_PENALTY = -1.5

# Training Loop
DIGITS_PER_EPISODE = 10       # All 10 digits
IMAGES_PER_DIGIT = 20         # Multi-image sampling (critical!)
```

### Alphabet

```python
# Network Architecture
INPUT_SIZE = 10000            # 100×100 pixels
HIDDEN_SIZE = 128
OUTPUT_SIZE = 26              # 26 letter classes

# Population & Evolution
POPULATION_SIZE = 200
MUTATION_RATE = 0.08
CHILD_MUTATION_RATE = 0.07
SURVIVAL_CUTOFF = 0.2

# Training Loop
LETTERS_PER_EPISODE = 26      # All 26 letters
VARIATIONS_PER_LETTER = 1     # Single clean rendering
```

## Comparison to Gradient Descent

| Aspect | GENREG | Gradient Descent |
|--------|--------|------------------|
| Search Method | Population-based exploration | Local gradient following |
| Update Rule | Selection + mutation | Weight -= lr × gradient |
| Convergence | Slower, explores broadly | Faster, can get stuck |
| Hyperparameters | Mutation rates, population size | Learning rate, optimizer params |
| Mid-training changes | Can modify config and resume | Must restart training |
| Parameter efficiency | Discovers compact solutions | Often overparameterized |

Performance Trade-offs:
- GENREG: 81% on MNIST with 50K params in ~40 minutes
- Typical MLP: 97-98% with 200K params in ~5-10 minutes
- CNNs: 99%+ with 60K-500K params in ~10-20 minutes

GENREG achieves ~83% of typical MLP performance with ~25% of the parameters.

## Evolutionary Dynamics

Training exhibits interesting population dynamics:

Early Phase (gen 0-100):
- Rapid initial learning (10% → 60%)
- Population discovers basic features
- High variance as genomes explore

Mid Phase (gen 100-400):
- Steady climb (60% → 75%)
- Population converges on effective strategies
- Variance decreases as trust accumulates

Late Phase (gen 400+):
- Refinement (75% → 81%+)
- Strategic trade-offs emerge
- Capacity redistribution visible
- Still improving at gen 600+

For capacity-constrained models (32 neurons):
- Early specialization on easy classes
- Forced redistribution as fitness pressure increases
- Individual class performance shifts while overall accuracy climbs
- Demonstrates evolutionary optimization under constraints

## Ongoing Research

Current experiments in progress:
- Architecture sweep: 16/32/64/128/256 hidden neurons
- Mutation rate ablation: Testing different child vs base mutation ratios
- Augmented alphabet: Testing generalization across 30 font variations
- Curriculum learning: Does natural curriculum emerge from evolutionary pressure?

Open questions:
- Can evolutionary learning achieve 90%+ on MNIST?
- What is minimum viable capacity for digit recognition?
- How do evolutionary embeddings compare to gradient-trained CNNs?
- Does this approach scale to CIFAR-10, ImageNet?

## Technical Details

### Trust Mechanics

Trust is the core fitness metric driving selection:

```python
# Each generation
for genome in population:
    trust = 0
    for sample in evaluation_samples:
        prediction = genome.forward(sample)
        if correct(prediction):
            trust += CORRECT_REWARD    # +10
        else:
            trust += WRONG_PENALTY      # -1.5
    
    # Apply decay from previous generation
    genome.trust = genome.trust * (1 - TRUST_DECAY) + trust
```

Trust accumulation creates selection pressure:
- High performers accumulate more trust over time
- Trust decay prevents stale champions from dominating
- Equilibrium trust = current_performance / decay_rate

### Selection & Reproduction

```python
# Selection
population.sort(key=lambda g: g.trust, reverse=True)
survivors = population[:int(POPULATION_SIZE * SURVIVAL_CUTOFF)]

# Reproduction
new_population = []
for _ in range(POPULATION_SIZE):
    parent = weighted_random_choice(survivors, weights=trust_scores)
    child = parent.copy()
    child.mutate(rate=CHILD_MUTATION_RATE)
    child.trust *= TRUST_INHERITANCE
    new_population.append(child)
```

### Mutation

Gaussian noise applied to network weights:

```python
def mutate(self, rate):
    for layer in self.network.parameters():
        mask = torch.rand_like(layer) < rate
        noise = torch.randn_like(layer) * MUTATION_SCALE
        layer.data += mask * noise
```

Child mutation rate (applied during reproduction) is more impactful than base mutation for driving exploration.

## Limitations

Current limitations and areas for improvement:

1. Speed: 4-8x slower than gradient descent for similar accuracy
2. Accuracy ceiling: Haven't matched state-of-the-art gradient methods yet
3. Scalability: Unclear how this scales to high-resolution images
4. Population size: Larger populations might improve results but increase compute
5. Theory: Lacking formal convergence guarantees or sample complexity bounds

## Citation

If you use this work, please cite:

```
@misc{genreg2024,
  author = {Payton Miller},
  title = {GENREG: Evolutionary Neural Network Training Through Trust-Based Selection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/[your-username]/genreg}
}
```

## Related Work

GENREG builds on ideas from:
- Evolution Strategies (Rechenberg 1973, Schwefel 1977)
- Neuroevolution (Montana & Davis 1989)
- NEAT (Stanley & Miikkulainen 2002)
- Natural Evolution Strategies (Wierstra et al. 2014)
- CMA-ES (Hansen & Ostermeier 2001)

Key differences:
- Trust-based fitness (not loss-based)
- Explicit population diversity maintenance
- Strategic capacity allocation under constraints
- Focus on parameter efficiency

## Future Directions

Planned experiments:
1. Convolutional architectures (can GENREG evolve conv filters?)
2. Multi-task learning (MNIST + Alphabet simultaneously)
3. Continual learning (catastrophic forgetting resistance?)
4. Benchmark on CIFAR-10, Fashion-MNIST, SVHN
5. Evolutionary architecture search (evolve network structure itself)
6. Theoretical analysis (convergence proofs, sample complexity)

## License

MIT License - see LICENSE file for details

## Acknowledgments

Thanks to the ML community for feedback and the evolutionary computation researchers whose work inspired this project.

## Contact

Questions, suggestions, or collaboration inquiries: [your contact info]

Project Link: https://github.com/[your-username]/genreg

---

Last updated: December 29, 2024
