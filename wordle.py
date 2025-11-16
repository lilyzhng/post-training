"""
Wordle Game with LLM Player

This script implements a Wordle game where an LLM (Qwen 2.5 7B) attempts to guess
a 5-letter word based on feedback. It demonstrates:
- System prompt design with detailed instructions
- Assistant prefilling for structured output
- Multi-turn conversation with state tracking
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================
load_dotenv()

# Initialize OpenAI client to query Qwen 2.5 7B Instruct on Predibase
client = OpenAI(
    base_url=os.environ["PREDIBASE_MODEL_QWEN_URL"],
    api_key=os.environ["PREDIBASE_API_KEY"],
)

# Load tokenizer for chat template formatting
base_model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)


# ============================================================================
# CONSTANTS
# ============================================================================
SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close 
your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:
Secret Word: BRISK

Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

### Response Format:
Think through the problem and feedback step by step. Make sure to 
first add your step by step thought process within <think> </think> 
tags. Then, return your guessed word in the following format: 
<guess> guessed-word </guess>.
"""


# ============================================================================
# DATA STRUCTURES
# ============================================================================
class LetterFeedback(Enum):
    """Represents feedback for a single letter in a Wordle guess."""
    CORRECT = "✓"       # Letter is correct and in correct position
    WRONG_POS = "-"     # Letter exists but in wrong position
    WRONG_LETTER = "x"  # Letter doesn't exist in the word


@dataclass
class GuessWithFeedback:
    """Stores a guess and its corresponding letter-by-letter feedback."""
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        """Returns a readable string showing the guess alongside its feedback."""
        feedback_str = " ".join(
            f"{letter}({fb.value})" 
            for letter, fb in zip(self.guess, self.feedback)
        )
        return f"{self.guess} → Feedback: {feedback_str}"


# ============================================================================
# CORE GAME LOGIC
# ============================================================================
def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    """
    Evaluates a guess against the secret word and returns feedback.
    
    Args:
        guess: The guessed word (should be 5 letters)
        secret_word: The target word (should be 5 letters)
    
    Returns:
        List of LetterFeedback for each letter in the guess
    """
    valid_letters = set(secret_word)
    feedback = []
    
    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            feedback.append(LetterFeedback.CORRECT)
        elif letter in valid_letters:
            feedback.append(LetterFeedback.WRONG_POS)
        else:
            feedback.append(LetterFeedback.WRONG_LETTER)
    
    return feedback


# ============================================================================
# PROMPT RENDERING
# ============================================================================
def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    """
    Creates the user-facing prompt that includes past guesses and feedback.
    
    Args:
        past_guesses: List of previous guesses with their feedback
    
    Returns:
        Formatted user prompt string
    """
    prompt = "Make a new 5-letter word guess."
    
    if past_guesses:
        prompt += "\n\nHere is some previous feedback:"
        for i, guess in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess}"
    
    return prompt


def render_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    """
    Formats a complete chat prompt with system message, user prompt, 
    and assistant prefill.
    
    This uses "assistant prefilling" - we start the assistant's response
    with a specific phrase to ensure consistent structured output. The
    `continue_final_message=True` parameter tells the tokenizer to leave
    the assistant message "open" so the model continues from that point.
    
    Args:
        past_guesses: List of previous guesses with their feedback
    
    Returns:
        Formatted prompt string ready for model completion
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": render_user_prompt(past_guesses)
        },
        {
            "role": "assistant",
            # Assistant prefilling: Start the response to ensure format
            "content": "Let me solve this step by step.\n<think>"
        }
    ]

    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        continue_final_message=True  # Leave assistant message "open"
    )


# ============================================================================
# MODEL INTERACTION
# ============================================================================
def generate_stream(prompt: str, adapter_id: str = "") -> str:
    """
    Streams a model-generated response and prints it in real-time.
    
    Args:
        prompt: The formatted prompt to send to the model
        adapter_id: Optional fine-tuned adapter ID (e.g., "wordle-dlai/2")
    
    Returns:
        The complete generated text
    """
    response = client.completions.create(
        model=adapter_id,
        prompt=prompt,
        temperature=0.0,  # Deterministic output for evaluation
        max_tokens=2048,
        stream=True,
    )
    
    completion = ""
    for chunk in response:
        if chunk.choices[0].text is not None:
            content = chunk.choices[0].text
            print(content, end="", flush=True)
            completion += content
    print()  # Newline after streaming completes

    return completion


# ============================================================================
# GAME LOOP
# ============================================================================
def next_turn(
    past_guesses: List[GuessWithFeedback], 
    secret_word: str, 
    adapter_id: str = ""
) -> bool:
    """
    Executes one turn of the game: generate guess, evaluate, display results.
    
    Args:
        past_guesses: List that will be updated with the new guess
        secret_word: The word the model is trying to guess
        adapter_id: Optional fine-tuned adapter ID
    
    Returns:
        True if game is over (won or lost), False otherwise
    """
    # Generate prompt and get model's response
    prompt = render_prompt(past_guesses)
    completion = generate_stream(prompt, adapter_id)
    
    # Extract the guess from the structured output
    match = re.search(r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL)
    if not match:
        raise RuntimeError("Model did not provide a valid guess format")
    
    guess = match.group(1).strip().upper()
    
    # Get feedback and update game state
    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    
    # Display current game state
    print("\n" + ("=" * 100))
    print("GAME STATE:")
    print("=" * 100)
    for past_guess in past_guesses:
        print(past_guess)
    print("=" * 100 + "\n")
    
    # Check win/loss conditions
    if guess == secret_word:
        print("🎉 SUCCESS! The model guessed the word! 🎉")
        return True
    elif len(past_guesses) >= 6:
        print(f"❌ Game Over! The word was: {secret_word} ❌")
        return True
    
    return False


def play_full_game(secret_word: str, adapter_id: str = "") -> None:
    """
    Plays a complete Wordle game until win or loss.
    
    Args:
        secret_word: The target word (should be 5 letters, uppercase)
        adapter_id: Optional fine-tuned adapter ID
    """
    print(f"\n{'='*100}")
    print(f"STARTING NEW GAME - Secret word: {'*' * len(secret_word)}")
    print(f"{'='*100}\n")
    
    past_guesses = []
    
    while len(past_guesses) < 6:
        game_over = next_turn(past_guesses, secret_word, adapter_id)
        if game_over:
            break


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Example 1: Play a single turn with pre-existing guesses
    print("\n" + "="*100)
    print("EXAMPLE 1: Testing prompt with pre-existing guesses")
    print("="*100 + "\n")
    
    test_guesses = [
        GuessWithFeedback(
            "CRANE", 
            [
                LetterFeedback.CORRECT, 
                LetterFeedback.CORRECT, 
                LetterFeedback.CORRECT, 
                LetterFeedback.WRONG_LETTER, 
                LetterFeedback.WRONG_LETTER,
            ]
        ),
        GuessWithFeedback(
            "CRASH", 
            [
                LetterFeedback.CORRECT, 
                LetterFeedback.CORRECT, 
                LetterFeedback.CORRECT, 
                LetterFeedback.WRONG_LETTER, 
                LetterFeedback.WRONG_LETTER,
            ]
        ),
    ]
    
    # Show the formatted prompt
    prompt = render_prompt(test_guesses)
    print("FORMATTED PROMPT:")
    print("-" * 100)
    print(prompt)
    print("-" * 100 + "\n")
    
    # Generate completion with base model
    print("BASE MODEL RESPONSE:")
    print("-" * 100)
    base_completion = generate_stream(prompt)
    print("-" * 100 + "\n")
    
    # Optional: Test with fine-tuned model
    # Uncomment the following lines if you have a fine-tuned adapter
    print("FINE-TUNED MODEL RESPONSE:")
    print("-" * 100)
    ft_completion = generate_stream(prompt, adapter_id="wordle-dlai/2")
    print("-" * 100 + "\n")
    
    # Example 2: Play a complete game
    print("\n" + "="*100)
    print("EXAMPLE 2: Playing a complete game")
    print("="*100)
    
    play_full_game(secret_word="BRICK", adapter_id="")