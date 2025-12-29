# ================================================================
# GENREG MNIST Evaluation Script (Standalone)
# ================================================================
# Self-contained script to evaluate a trained MNIST genome.
# No external dependencies on config.py or genreg_controller.py.
#
# Usage:
#   python mnist_eval.py
#
# Output:
#   - Console report with accuracy metrics
#   - JSON file with detailed results
# ================================================================

import os
import sys
import pickle
import json
import time
import math
import random
from datetime import datetime

import numpy as np


# ================================================================
# SIMPLE NEURAL NETWORK (replaces GENREGController)
# ================================================================
class SimpleNetwork:
    """
    Minimal forward-pass-only neural network for inference.
    Supports both PyTorch (GPU) and pure Python (CPU) modes.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Try to use PyTorch if available
        try:
            import torch
            self._use_torch = True
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        except ImportError:
            self._use_torch = False
            self.device = None

        # Weights will be loaded from checkpoint
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def load_weights(self, ctrl_data):
        """Load weights from checkpoint data."""
        if self._use_torch:
            import torch
            self.w1 = torch.tensor(ctrl_data['w1'], dtype=torch.float32, device=self.device)
            self.b1 = torch.tensor(ctrl_data['b1'], dtype=torch.float32, device=self.device)
            self.w2 = torch.tensor(ctrl_data['w2'], dtype=torch.float32, device=self.device)
            self.b2 = torch.tensor(ctrl_data['b2'], dtype=torch.float32, device=self.device)
        else:
            self.w1 = ctrl_data['w1']
            self.b1 = ctrl_data['b1']
            self.w2 = ctrl_data['w2']
            self.b2 = ctrl_data['b2']

    def forward_visual(self, visual_input):
        """Forward pass returning digit probabilities."""
        if self._use_torch:
            return self._forward_torch(visual_input)
        else:
            return self._forward_python(visual_input)

    def _forward_torch(self, visual_input):
        """GPU-accelerated forward pass."""
        import torch
        if not isinstance(visual_input, torch.Tensor):
            x = torch.tensor(visual_input, dtype=torch.float32, device=self.device)
        else:
            x = visual_input.to(self.device)

        hidden = torch.tanh(self.w1 @ x + self.b1)
        outputs = self.w2 @ hidden + self.b2

        # For MNIST, first 10 outputs are digit probabilities
        digit_logits = outputs[:10] if len(outputs) >= 10 else outputs
        digit_probs = torch.softmax(digit_logits, dim=0)

        return digit_probs.cpu().tolist()

    def _forward_python(self, visual_input):
        """Pure Python forward pass."""
        # Hidden layer
        hidden = []
        for i in range(self.hidden_size):
            s = self.b1[i]
            for j in range(min(self.input_size, len(visual_input))):
                s += self.w1[i][j] * visual_input[j]
            hidden.append(math.tanh(s))

        # Output layer
        outputs = []
        for i in range(min(self.output_size, 10)):
            s = self.b2[i]
            for j in range(self.hidden_size):
                s += self.w2[i][j] * hidden[j]
            outputs.append(s)

        # Softmax
        max_logit = max(outputs) if outputs else 0
        exp_logits = [math.exp(x - max_logit) for x in outputs]
        sum_exp = sum(exp_logits)
        digit_probs = [e / sum_exp for e in exp_logits]

        return digit_probs


# ================================================================
# GENOME LOADING
# ================================================================
def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def list_mnist_genomes():
    """List all MNIST genomes in best_genomes folder."""
    genomes = []
    genome_dir = "best_genomes"

    if not os.path.exists(genome_dir):
        return []

    for filename in os.listdir(genome_dir):
        if filename.endswith('.pkl') and 'mnist' in filename.lower():
            filepath = os.path.join(genome_dir, filename)
            file_size = os.path.getsize(filepath)

            # Extract generation from filename
            gen_num = "?"
            try:
                if "_gen" in filename:
                    gen_part = filename.split("_gen")[1]
                    gen_num = gen_part.split("_")[0]
            except:
                pass

            genomes.append({
                'path': filepath,
                'filename': filename,
                'size': file_size,
                'generation': gen_num
            })

    # Sort by filename (newest first typically)
    genomes.sort(key=lambda x: x['filename'], reverse=True)
    return genomes


def select_genome():
    """Interactive genome selection."""
    genomes = list_mnist_genomes()

    if not genomes:
        print("No MNIST genomes found in best_genomes/")
        print("Extract one first with: python extract_best_genome.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SELECT MNIST GENOME FOR EVALUATION")
    print("=" * 60)
    print("\nAvailable MNIST genomes:\n")

    for i, g in enumerate(genomes[:10]):
        size_str = format_size(g['size'])
        print(f"  [{i+1}] {g['filename']}")
        print(f"       Gen {g['generation']}, {size_str}")

    if len(genomes) == 1:
        print(f"\n  [Enter] Use {genomes[0]['filename']}")
    else:
        print(f"\n  [Enter] Use most recent")

    choice = input("\nYour choice: ").strip()

    if choice == "":
        return genomes[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(genomes):
        return genomes[int(choice) - 1]
    else:
        print("Invalid choice, using most recent...")
        return genomes[0]


def load_genome(genome_info):
    """Load a genome from pickle file."""
    print(f"\nLoading: {genome_info['filename']}")

    with open(genome_info['path'], 'rb') as f:
        data = pickle.load(f)

    # Extract controller data
    ctrl_data = data['controller']
    input_size = ctrl_data['input_size']
    hidden_size = ctrl_data['hidden_size']
    output_size = ctrl_data['output_size']

    print(f"  Network: {input_size} -> {hidden_size} -> {output_size}")

    # Verify it's a MNIST model
    if input_size != 784:
        print(f"  WARNING: Expected 784 input (MNIST), got {input_size}")
        print("  This may not be a MNIST model!")

    if output_size != 10:
        print(f"  WARNING: Expected 10 output (digits), got {output_size}")

    # Create network and load weights
    network = SimpleNetwork(input_size, hidden_size, output_size)
    network.load_weights(ctrl_data)

    # Get metadata
    trust = data.get('genome', {}).get('trust', 0)
    print(f"  Trust: {trust:.2f}")

    return network, data


# ================================================================
# MNIST DATASET LOADING
# ================================================================
def load_mnist_test_set():
    """Load the official MNIST 10K test set using torchvision."""
    print("\nLoading MNIST test set...")

    try:
        from torchvision import datasets, transforms
        import torch
    except ImportError:
        print("ERROR: torchvision is required. Install with: pip install torchvision")
        sys.exit(1)

    # Standard MNIST normalization (convert to tensor, normalize to [0,1])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download if necessary, load test set
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,  # Test set (10,000 images)
        download=True,
        transform=transform
    )

    print(f"  Loaded {len(test_dataset)} test images")
    print(f"  Image shape: 28x28 (784 pixels)")

    return test_dataset


# ================================================================
# EVALUATION
# ================================================================
def evaluate_genome(network, test_dataset):
    """
    Evaluate genome on the entire MNIST test set.

    Returns detailed results dict.
    """
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION ON 10,000 TEST IMAGES")
    print("=" * 60)

    total = len(test_dataset)
    correct = 0

    # Per-digit stats
    digit_correct = [0] * 10
    digit_total = [0] * 10

    # Confusion matrix
    confusion = [[0] * 10 for _ in range(10)]

    # Store all predictions for detailed analysis
    predictions = []

    # Timing
    start_time = time.perf_counter()
    inference_times = []

    print(f"\nEvaluating {total} images...")

    for idx, (image, label) in enumerate(test_dataset):
        # Convert image to flat list (784 floats)
        # image shape: (1, 28, 28) -> flatten to 784
        obs = image.view(-1).tolist()

        # Time the inference
        inf_start = time.perf_counter()

        # Get prediction
        digit_probs = network.forward_visual(obs)
        predicted = digit_probs[:10].index(max(digit_probs[:10]))

        inf_end = time.perf_counter()
        inference_times.append((inf_end - inf_start) * 1000)

        # Record results
        is_correct = (predicted == label)
        if is_correct:
            correct += 1
            digit_correct[label] += 1

        digit_total[label] += 1
        confusion[label][predicted] += 1

        predictions.append({
            'index': idx,
            'true_label': label,
            'predicted': predicted,
            'correct': is_correct,
            'confidence': digit_probs[predicted],
            'probabilities': digit_probs[:10]
        })

        # Progress update
        if (idx + 1) % 1000 == 0:
            current_acc = correct / (idx + 1) * 100
            print(f"  Progress: {idx + 1}/{total} ({current_acc:.2f}% accuracy so far)")

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate per-digit accuracy
    digit_accuracy = []
    for d in range(10):
        if digit_total[d] > 0:
            acc = digit_correct[d] / digit_total[d] * 100
        else:
            acc = 0.0
        digit_accuracy.append(acc)

    # Overall accuracy
    overall_accuracy = correct / total * 100

    # Timing stats
    avg_inference_time = sum(inference_times) / len(inference_times)

    results = {
        'overall': {
            'total': total,
            'correct': correct,
            'accuracy': overall_accuracy,
            'accuracy_str': f"{overall_accuracy:.2f}%"
        },
        'per_digit': {
            'correct': digit_correct,
            'total': digit_total,
            'accuracy': digit_accuracy
        },
        'confusion_matrix': confusion,
        'timing': {
            'total_seconds': total_time,
            'avg_inference_ms': avg_inference_time,
            'throughput_per_sec': total / total_time
        },
        'predictions': predictions  # Full detail (can be large)
    }

    return results


# ================================================================
# INFERENCE BENCHMARK
# ================================================================
def run_inference_benchmark(network, test_dataset, iterations=1000):
    """
    Run dedicated inference benchmark for accurate timing.

    Uses a single image repeated many times to measure pure inference speed.
    """
    print("\n" + "=" * 60)
    print(f"INFERENCE BENCHMARK ({iterations} iterations)")
    print("=" * 60)

    # Get a sample image
    image, label = test_dataset[0]
    obs = image.view(-1).tolist()

    # Warmup runs (not counted)
    print("  Warmup (10 runs)...")
    for _ in range(10):
        network.forward_visual(obs)

    # Benchmark runs
    print(f"  Benchmarking ({iterations} runs)...")
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        network.forward_visual(obs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{iterations}...")

    # Calculate statistics
    times.sort()
    benchmark_results = {
        'iterations': iterations,
        'min_ms': min(times),
        'max_ms': max(times),
        'mean_ms': sum(times) / len(times),
        'median_ms': times[len(times) // 2],
        'p95_ms': times[int(len(times) * 0.95)],
        'p99_ms': times[int(len(times) * 0.99)],
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'throughput_per_sec': 1000 / (sum(times) / len(times))
    }

    # Print results
    print("\n  Results:")
    print(f"    Min:        {benchmark_results['min_ms']:.4f} ms")
    print(f"    Max:        {benchmark_results['max_ms']:.4f} ms")
    print(f"    Mean:       {benchmark_results['mean_ms']:.4f} ms")
    print(f"    Median:     {benchmark_results['median_ms']:.4f} ms")
    print(f"    Std Dev:    {benchmark_results['std_ms']:.4f} ms")
    print(f"    P95:        {benchmark_results['p95_ms']:.4f} ms")
    print(f"    P99:        {benchmark_results['p99_ms']:.4f} ms")
    print(f"    Throughput: {benchmark_results['throughput_per_sec']:.0f} inferences/sec")

    return benchmark_results


def print_results(results, genome_info):
    """Print evaluation results in a nice format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nGenome: {genome_info['filename']}")
    print(f"Generation: {genome_info['generation']}")

    print(f"\n{'='*40}")
    print(f"OVERALL ACCURACY: {results['overall']['accuracy']:.2f}%")
    print(f"  Correct: {results['overall']['correct']} / {results['overall']['total']}")
    print(f"{'='*40}")

    print("\nPer-Digit Accuracy:")
    print("-" * 40)
    for digit in range(10):
        acc = results['per_digit']['accuracy'][digit]
        correct = results['per_digit']['correct'][digit]
        total = results['per_digit']['total'][digit]
        bar = "#" * int(acc / 5) + "." * (20 - int(acc / 5))
        print(f"  Digit {digit}: {acc:5.1f}% [{bar}] ({correct}/{total})")

    print("\nConfusion Matrix:")
    print("-" * 40)
    print("         Predicted")
    print("         " + " ".join(f"{d:4}" for d in range(10)))
    print("        " + "-" * 45)
    for true_digit in range(10):
        row = results['confusion_matrix'][true_digit]
        row_str = " ".join(f"{c:4}" for c in row)
        print(f"True {true_digit} | {row_str}")

    print(f"\nTiming:")
    print(f"  Total time: {results['timing']['total_seconds']:.2f} seconds")
    print(f"  Avg inference: {results['timing']['avg_inference_ms']:.3f} ms")
    print(f"  Throughput: {results['timing']['throughput_per_sec']:.0f} images/sec")

    # Random guess baseline
    print(f"\nBaseline (random guess): 10.00%")
    improvement = results['overall']['accuracy'] - 10.0
    print(f"Improvement over random: +{improvement:.2f}%")


def save_results(results, genome_info, benchmark_results=None):
    """Save results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen = genome_info['generation']

    # Create output filename
    output_filename = f"mnist_eval_gen{gen}_{timestamp}.json"
    output_path = os.path.join("best_genomes", output_filename)

    # Prepare JSON-serializable results
    # Remove full predictions list to keep file size reasonable
    save_data = {
        'genome': {
            'filename': genome_info['filename'],
            'generation': genome_info['generation'],
            'path': genome_info['path']
        },
        'evaluation': {
            'timestamp': timestamp,
            'test_set_size': results['overall']['total'],
        },
        'results': {
            'overall': results['overall'],
            'per_digit': results['per_digit'],
            'confusion_matrix': results['confusion_matrix'],
            'timing': results['timing']
        },
        # Include a sample of wrong predictions for analysis
        'sample_errors': [
            p for p in results['predictions'] if not p['correct']
        ][:100]  # First 100 errors
    }

    # Add benchmark results if available
    if benchmark_results:
        save_data['inference_benchmark'] = benchmark_results

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "=" * 60)
    print("GENREG MNIST EVALUATION")
    print("Official 10K Test Set Benchmark")
    print("=" * 60)

    # Select genome
    genome_info = select_genome()

    # Load genome
    network, data = load_genome(genome_info)

    # Load MNIST test set
    test_dataset = load_mnist_test_set()

    # Run evaluation
    results = evaluate_genome(network, test_dataset)

    # Print results
    print_results(results, genome_info)

    # Run inference benchmark
    benchmark_results = run_inference_benchmark(network, test_dataset)

    # Save results (including benchmark)
    output_path = save_results(results, genome_info, benchmark_results)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
