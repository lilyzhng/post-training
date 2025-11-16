from utils import (
    create_deployment,
    GuessWithFeedback,
    generate, 
    get_messages,
    extract_guess,
    print_guesses_table,
)
import numpy as np

import warnings 
warnings.filterwarnings("ignore")

import torch

create_deployment()

model_id = "Qwen/Qwen2.5-7B-Instruct"

# ============================================================================
# Define a simple reward function
# ============================================================================
def worldle_reward(guess: str, secret_word: str) -> float:
    """
    Reward function for Wordle.
    """
    if guess.upper() == secret_word.upper():
        return 1.0 # correct guess
    else:
        return 0.0 # incorrect guess

## define a secret word and get feedback on past guesses, then score the guesses using the reward function above:
secret_word = "POUND"
past_guesses = [
    GuessWithFeedback.from_secret("CRANE", secret_word),
    GuessWithFeedback.from_secret("BLOND", secret_word),
    GuessWithFeedback.from_secret("FOUND", secret_word),
]
print(past_guesses)

response = generate(get_messages(past_guesses))[0]
guess = extract_guess(response)
reward = worldle_reward(guess, secret_word)

print(f"Guess word: {guess} -> Reward: {reward}")

# ============================================================================
# Using rewards to calculate advantages:
# ============================================================================
def compute_advantages(rewards: list):
    rewards = np.array(rewards)
    # compute mean and the standard deviation of the rewards
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Avoid division by zero in case of zero variance (typically happens when all rewards are 0)
    # Note: in the GRPO implementation, we did 1e-4 to the std_reward to avoid division by zero
    if std_reward == 0:
        return [0] * len(rewards)

    # advantages equals ot divide by the stddev of rewards to normalize range to 0
    # z-score normalization. showing how many standard deviations each value is from the average.
    advantages = (rewards - mean_reward) / std_reward
    return advantages.tolist()

rewards = [0.0, 0.2, 0.4, 0.5, 0.5, 0.6, 0.8, 1.0]
advantages = compute_advantages(rewards)
print(f"Advantages: {advantages}")

def render_guess_table(response, reward_fn):
    """
    reward_fn is a function parameter (callback) that takes a guess and returns a numeric score indicating how good that guess is.
    """
    guesses = [extract_guess(guess) for guess in response]
    rewards = [reward_fn(guess, secret_word) for guess in guesses]

    print_guesses_table(guesses, rewards)

print(f"Secret: {secret_word}")
# Uses past guesses as context to generate 8 new AI guesses
response = generate(get_messages(past_guesses), num_guesses=8)
render_guess_table(response, worldle_reward)

# ============================================================================
# Update the reward function to give partial credit, binary rewards ignores partial credit
# ============================================================================
def worldle_reward_partial(guess: str, secret_word: str) -> float:
    if len(guess) != len(secret_word):
        return 0.0
    
    valid_letters = set(secret_word)
    reward = 0.0
    # zip pairs up letters at matching positions, making it easy to compare "letter 0 vs letter 0"
    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            reward += 0.2 # right letter, right position
        elif letter in valid_letters:
            reward += 0.1 # right letter, wrong position
        else:
            pass
    return reward

# Try scoring a set of responses using updated reward function. Start by setting temperature = 0.0.
print(f"Secret: {secret_word}") # low temp is boring but save
response = generate(get_messages(past_guesses), num_guesses=8, temperature=0)
render_guess_table(response, worldle_reward_partial)

# Now set temperature = 1.3, a high value
print(f"Secret: {secret_word}") # high temp is creative but risky
response = generate(get_messages(past_guesses), num_guesses=8, temperature=1.3)
render_guess_table(response, worldle_reward_partial)

# Lastly, set temperature = 0.7, a moderate value
print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses=8, temperature=0.7)
render_guess_table(response, worldle_reward_partial)