# ================================================================
# GENREG Alphabet Inference Tool (Standalone)
# ================================================================
# Self-contained script to test a trained alphabet model.
# No external dependencies on config.py or genreg_controller.py.
#
# Features:
# - Select a genome from best_genomes/ folder
# - Keyboard layout to click letters
# - Text input to type letters
# - Shows model's prediction vs actual letter
# - 5-Cycle randomized test with JSON output
# ================================================================

import os
import sys
import json
import pickle
import random
import math
from datetime import datetime
import pygame
import numpy as np

# ================================================================
# HARDCODED CONFIGURATION (from config.py)
# ================================================================
ALPHABET_FIELD_WIDTH = 100
ALPHABET_FIELD_HEIGHT = 100
ALPHABET_FONT_SIZE = 64


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
        """Forward pass returning character probabilities."""
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

        char_logits = outputs[:26] if len(outputs) >= 26 else outputs
        char_probs = torch.softmax(char_logits, dim=0)

        return char_probs.cpu().tolist()

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
        for i in range(min(self.output_size, 26)):
            s = self.b2[i]
            for j in range(self.hidden_size):
                s += self.w2[i][j] * hidden[j]
            outputs.append(s)

        # Softmax
        max_logit = max(outputs) if outputs else 0
        exp_logits = [math.exp(x - max_logit) for x in outputs]
        sum_exp = sum(exp_logits)
        char_probs = [e / sum_exp for e in exp_logits]

        return char_probs

    def generate_char(self, visual_input, temperature=1.0):
        """Generate a single character from visual input."""
        char_probs = self.forward_visual(visual_input)

        # Apply temperature
        if temperature != 1.0:
            scaled = [p ** (1.0 / temperature) for p in char_probs]
            total = sum(scaled)
            char_probs = [p / total for p in scaled]

        # Sample from distribution
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(char_probs):
            cumsum += p
            if r <= cumsum:
                if i < 26:
                    return chr(ord('a') + i)
                break

        # Fallback: return highest probability letter
        max_idx = char_probs.index(max(char_probs[:26]))
        return chr(ord('a') + max_idx)


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


def list_alphabet_genomes():
    """List all alphabet genomes in best_genomes folder."""
    genomes = []
    genome_dir = "best_genomes"

    if not os.path.exists(genome_dir):
        return []

    for filename in os.listdir(genome_dir):
        if filename.endswith('.pkl') and 'alphabet' in filename.lower():
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
    genomes = list_alphabet_genomes()

    if not genomes:
        print("No alphabet genomes found in best_genomes/")
        print("Extract one first with: python extract_best_genome.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SELECT ALPHABET GENOME")
    print("=" * 60)
    print("\nAvailable alphabet genomes:\n")

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

    # Verify it's an alphabet model
    expected_input = ALPHABET_FIELD_WIDTH * ALPHABET_FIELD_HEIGHT
    if input_size != expected_input:
        print(f"  WARNING: Expected {expected_input} input (alphabet), got {input_size}")
        print("  This may not be an alphabet model!")

    if output_size != 26:
        print(f"  WARNING: Expected 26 output (letters), got {output_size}")

    # Create network and load weights
    network = SimpleNetwork(input_size, hidden_size, output_size)
    network.load_weights(ctrl_data)

    # Get metadata
    trust = data.get('genome', {}).get('trust', 0)
    print(f"  Trust: {trust:.2f}")

    return network, data


# ================================================================
# LETTER RENDERER (matches training)
# ================================================================
class LetterRenderer:
    """Render letters for inference (matches training rendering)."""

    def __init__(self):
        self.width = ALPHABET_FIELD_WIDTH
        self.height = ALPHABET_FIELD_HEIGHT
        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.Font(None, ALPHABET_FONT_SIZE)

    def render(self, letter):
        """Render a letter and return observation."""
        self.surface.fill((0, 0, 0))
        text = self.font.render(letter.upper(), True, (255, 255, 255))
        rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.surface.blit(text, rect)

        # Convert to observation (grayscale normalized)
        pixels = pygame.surfarray.array3d(self.surface)
        grayscale = np.mean(pixels, axis=2)
        normalized = grayscale / 255.0
        return normalized.flatten().tolist()

    def get_surface(self):
        """Get the rendered surface for display."""
        return self.surface


# ================================================================
# INFERENCE UI
# ================================================================
class AlphabetInferenceUI:
    """Pygame UI for alphabet inference testing."""

    def __init__(self, network):
        self.network = network

        # Window dimensions
        self.width = 800
        self.height = 600

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("GENREG Alphabet Inference")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Letter renderer
        self.renderer = LetterRenderer()

        # State
        self.current_letter = None
        self.predicted_letter = None
        self.is_correct = None
        self.input_text = ""
        self.input_active = False

        # Stats
        self.total_tests = 0
        self.correct_tests = 0

        # Keyboard layout
        self.keyboard_rows = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM")
        ]
        self.key_buttons = []
        self._create_keyboard()

        # Input box
        self.input_rect = pygame.Rect(50, 500, 200, 40)

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (100, 100, 100)
        self.GREEN = (50, 200, 50)
        self.RED = (200, 50, 50)
        self.BLUE = (50, 100, 200)
        self.YELLOW = (200, 200, 50)

    def _create_keyboard(self):
        """Create keyboard button rectangles."""
        key_width = 50
        key_height = 50
        key_margin = 5
        start_y = 300

        self.key_buttons = []

        for row_idx, row in enumerate(self.keyboard_rows):
            row_width = len(row) * (key_width + key_margin)
            start_x = (self.width - row_width) // 2

            for col_idx, letter in enumerate(row):
                x = start_x + col_idx * (key_width + key_margin)
                y = start_y + row_idx * (key_height + key_margin)
                rect = pygame.Rect(x, y, key_width, key_height)
                self.key_buttons.append((letter, rect))

    def test_letter(self, letter):
        """Test the model on a specific letter."""
        letter = letter.upper()
        if letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return

        self.current_letter = letter

        # Render letter and get observation
        obs = self.renderer.render(letter)

        # Get model prediction
        char = self.network.generate_char(obs)
        self.predicted_letter = char.upper()

        # Check if correct
        self.is_correct = (self.predicted_letter == self.current_letter)

        # Update stats
        self.total_tests += 1
        if self.is_correct:
            self.correct_tests += 1

    def draw(self):
        """Draw the UI."""
        self.screen.fill((30, 30, 40))

        # Title
        title = self.font_medium.render("GENREG Alphabet Inference", True, self.WHITE)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 20))

        # Instructions
        instr = self.font_small.render("Click a letter or type in the box below", True, self.GRAY)
        self.screen.blit(instr, (self.width // 2 - instr.get_width() // 2, 60))

        # Letter display area
        display_rect = pygame.Rect(self.width // 2 - 150, 100, 300, 150)
        pygame.draw.rect(self.screen, self.GRAY, display_rect, 2)

        if self.current_letter:
            # Show the rendered letter (what model sees)
            letter_surface = self.renderer.get_surface()
            scaled = pygame.transform.scale(letter_surface, (100, 100))
            self.screen.blit(scaled, (self.width // 2 - 130, 125))

            # Arrow
            arrow = self.font_medium.render("→", True, self.WHITE)
            self.screen.blit(arrow, (self.width // 2 - 20, 155))

            # Model's prediction
            if self.is_correct:
                pred_color = self.GREEN
            else:
                pred_color = self.RED

            pred_text = self.font_large.render(self.predicted_letter, True, pred_color)
            self.screen.blit(pred_text, (self.width // 2 + 50, 140))

            # Result text
            if self.is_correct:
                result = self.font_small.render("CORRECT!", True, self.GREEN)
            else:
                result = self.font_small.render(f"WRONG (expected {self.current_letter})", True, self.RED)
            self.screen.blit(result, (self.width // 2 - result.get_width() // 2, 260))
        else:
            hint = self.font_small.render("Select a letter to test", True, self.GRAY)
            self.screen.blit(hint, (self.width // 2 - hint.get_width() // 2, 165))

        # Draw keyboard
        for letter, rect in self.key_buttons:
            # Highlight if this is the current letter
            if letter == self.current_letter:
                if self.is_correct:
                    color = self.GREEN
                else:
                    color = self.RED
            else:
                color = self.BLUE

            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.WHITE, rect, 2)

            text = self.font_medium.render(letter, True, self.WHITE)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

        # Input box
        pygame.draw.rect(self.screen, self.WHITE if self.input_active else self.GRAY,
                         self.input_rect, 2)

        input_label = self.font_small.render("Type letter:", True, self.WHITE)
        self.screen.blit(input_label, (self.input_rect.x, self.input_rect.y - 25))

        input_text = self.font_medium.render(self.input_text or "_", True, self.WHITE)
        self.screen.blit(input_text, (self.input_rect.x + 10, self.input_rect.y + 8))

        # Stats
        if self.total_tests > 0:
            accuracy = self.correct_tests / self.total_tests * 100
            stats = self.font_small.render(
                f"Accuracy: {self.correct_tests}/{self.total_tests} ({accuracy:.1f}%)",
                True, self.YELLOW
            )
            self.screen.blit(stats, (self.width - stats.get_width() - 20, 550))

        # Test all button
        test_all_rect = pygame.Rect(self.width - 300, 500, 130, 40)
        pygame.draw.rect(self.screen, self.BLUE, test_all_rect)
        pygame.draw.rect(self.screen, self.WHITE, test_all_rect, 2)
        test_all_text = self.font_small.render("Test All (A-Z)", True, self.WHITE)
        self.screen.blit(test_all_text, (test_all_rect.x + 10, test_all_rect.y + 10))

        # 5-Cycle Test button
        cycle_test_rect = pygame.Rect(self.width - 150, 500, 130, 40)
        pygame.draw.rect(self.screen, (150, 50, 150), cycle_test_rect)  # Purple
        pygame.draw.rect(self.screen, self.WHITE, cycle_test_rect, 2)
        cycle_test_text = self.font_small.render("5-Cycle Test", True, self.WHITE)
        self.screen.blit(cycle_test_text, (cycle_test_rect.x + 15, cycle_test_rect.y + 10))

        # Quit hint
        quit_hint = self.font_small.render("ESC to quit", True, self.GRAY)
        self.screen.blit(quit_hint, (20, 560))

        pygame.display.flip()

        return test_all_rect, cycle_test_rect

    def run(self):
        """Main UI loop."""
        running = True

        print("\n[INFERENCE] Ready! Click letters or type to test.")
        print("[INFERENCE] Press ESC to quit.\n")

        while running:
            test_all_rect, cycle_test_rect = self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_RETURN:
                        if self.input_text:
                            self.test_letter(self.input_text)
                            self.input_text = ""

                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]

                    elif event.unicode.isalpha():
                        self.input_text = event.unicode.upper()
                        self.test_letter(self.input_text)
                        self.input_text = ""

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()

                    # Check keyboard clicks
                    for letter, rect in self.key_buttons:
                        if rect.collidepoint(pos):
                            self.test_letter(letter)
                            break

                    # Check input box click
                    if self.input_rect.collidepoint(pos):
                        self.input_active = True
                    else:
                        self.input_active = False

                    # Check test all button
                    if test_all_rect.collidepoint(pos):
                        self.test_all_letters()

                    # Check 5-cycle test button
                    if cycle_test_rect.collidepoint(pos):
                        self.run_5cycle_test()

            self.clock.tick(30)

        pygame.quit()

        # Print final stats
        if self.total_tests > 0:
            accuracy = self.correct_tests / self.total_tests * 100
            print(f"\n[FINAL] Accuracy: {self.correct_tests}/{self.total_tests} ({accuracy:.1f}%)")

    def test_all_letters(self):
        """Test all 26 letters in sequence."""
        print("\n[TEST ALL] Testing A-Z...")

        results = {}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.test_letter(letter)
            results[letter] = (self.predicted_letter, self.is_correct)
            self.draw()
            pygame.display.flip()
            pygame.time.wait(100)  # Brief pause to see each result

        # Print summary
        correct = sum(1 for _, (_, is_correct) in results.items() if is_correct)
        print(f"[TEST ALL] Results: {correct}/26 ({correct/26*100:.1f}%)")

        # Show which letters were wrong
        wrong = [f"{l}→{p}" for l, (p, c) in results.items() if not c]
        if wrong:
            print(f"[TEST ALL] Mistakes: {', '.join(wrong)}")

    def run_5cycle_test(self):
        """Run 5 randomized cycles of all 26 letters and save detailed JSON results."""
        print("\n" + "=" * 60)
        print("[5-CYCLE TEST] Starting randomized 5-cycle test...")
        print("=" * 60)

        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference", "output")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize tracking
        all_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        cycle_results = []
        per_letter_stats = {letter: {"correct": 0, "wrong": 0, "predictions": []} for letter in all_letters}
        confusion_matrix = {letter: {pred: 0 for pred in all_letters} for letter in all_letters}

        total_correct = 0
        total_tests = 0

        # Run 5 cycles
        for cycle_num in range(1, 6):
            print(f"\n[CYCLE {cycle_num}/5] Testing randomized alphabet...")

            # Shuffle the alphabet for this cycle
            shuffled = all_letters.copy()
            random.shuffle(shuffled)

            cycle_data = {
                "cycle": cycle_num,
                "order": shuffled.copy(),
                "results": []
            }

            cycle_correct = 0

            for letter in shuffled:
                # Test the letter
                obs = self.renderer.render(letter)
                predicted = self.network.generate_char(obs).upper()
                is_correct = (predicted == letter)

                # Update tracking
                total_tests += 1
                if is_correct:
                    total_correct += 1
                    cycle_correct += 1
                    per_letter_stats[letter]["correct"] += 1
                else:
                    per_letter_stats[letter]["wrong"] += 1

                per_letter_stats[letter]["predictions"].append(predicted)
                confusion_matrix[letter][predicted] += 1

                cycle_data["results"].append({
                    "letter": letter,
                    "predicted": predicted,
                    "correct": is_correct
                })

                # Update display
                self.current_letter = letter
                self.predicted_letter = predicted
                self.is_correct = is_correct
                self.draw()
                pygame.time.wait(50)  # Quick visual feedback

                # Process events to keep UI responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

            cycle_data["accuracy"] = cycle_correct / 26 * 100
            cycle_results.append(cycle_data)
            print(f"[CYCLE {cycle_num}/5] Accuracy: {cycle_correct}/26 ({cycle_data['accuracy']:.1f}%)")

        # Calculate final statistics
        overall_accuracy = total_correct / total_tests * 100

        # Find best and worst performing letters
        letter_accuracies = []
        for letter in all_letters:
            stats = per_letter_stats[letter]
            acc = stats["correct"] / 5 * 100  # 5 tests per letter
            letter_accuracies.append((letter, acc))

        letter_accuracies.sort(key=lambda x: x[1], reverse=True)
        best_letters = [l for l, a in letter_accuracies if a == 100]
        worst_letters = [(l, a) for l, a in letter_accuracies if a < 100]
        worst_letters.sort(key=lambda x: x[1])

        # Build detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_data = {
            "test_info": {
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "total_cycles": 5,
                "letters_per_cycle": 26,
                "total_tests": total_tests
            },
            "summary": {
                "overall_accuracy": round(overall_accuracy, 2),
                "total_correct": total_correct,
                "total_wrong": total_tests - total_correct,
                "perfect_letters": best_letters,
                "perfect_letter_count": len(best_letters),
                "problem_letters": [{"letter": l, "accuracy": round(a, 1)} for l, a in worst_letters[:5]]
            },
            "per_letter_stats": {
                letter: {
                    "correct": stats["correct"],
                    "wrong": stats["wrong"],
                    "accuracy": round(stats["correct"] / 5 * 100, 1),
                    "all_predictions": stats["predictions"]
                }
                for letter, stats in per_letter_stats.items()
            },
            "confusion_matrix": confusion_matrix,
            "cycle_details": cycle_results
        }

        # Save to JSON
        output_filename = f"5cycle_test_{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("[5-CYCLE TEST] COMPLETE")
        print("=" * 60)
        print(f"\nOverall Accuracy: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
        print(f"Perfect Letters ({len(best_letters)}): {', '.join(best_letters) if best_letters else 'None'}")

        if worst_letters:
            print(f"\nWorst Performing:")
            for letter, acc in worst_letters[:5]:
                preds = per_letter_stats[letter]["predictions"]
                pred_summary = ', '.join(preds)
                print(f"  {letter}: {acc:.0f}% (predicted: {pred_summary})")

        print(f"\nResults saved to: {output_path}")
        print("=" * 60)

        # Update UI stats
        self.total_tests += total_tests
        self.correct_tests += total_correct


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "=" * 60)
    print("GENREG ALPHABET INFERENCE TOOL")
    print("=" * 60)

    # Select genome from best_genomes folder
    genome_info = select_genome()

    # Load genome
    network, data = load_genome(genome_info)

    print("\n" + "=" * 60)
    print("Starting inference UI...")
    print("=" * 60)

    # Run UI
    ui = AlphabetInferenceUI(network)
    ui.run()


if __name__ == "__main__":
    main()
