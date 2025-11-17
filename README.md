# Post-Training with Wordle

This project demonstrates the difference between a baseline instruction-following model and a fine-tuned model specifically trained for Wordle gameplay.

## Overview

The code shows how to use **prompt engineering** techniques (system prompts, assistant prefilling) and compares performance between:
- A general-purpose base model
- A fine-tuned model with a LoRA adapter trained on Wordle examples

## The Two Models

### 1. **Baseline Model** (Out-of-the-box)
- **Model**: `Qwen/Qwen2.5-7B-Instruct`
- **What it is**: The original pre-trained instruction-following model
- **When it's used**: When `adapter_id=""` (empty string)
- **Performance**: General reasoning, but not optimized for Wordle strategy

### 2. **Fine-Tuned Model** (Post-Training)
- **Model**: `Qwen/Qwen2.5-7B-Instruct` + `"wordle-dlai/2"` adapter
- **What it is**: The base model + a LoRA adapter trained specifically on Wordle examples
- **When it's used**: When `adapter_id="wordle-dlai/2"`
- **Performance**: Better at Wordle-specific reasoning and strategy

## Visual Comparison

Both models receive the **exact same prompt**, but respond differently based on their training:

```
┌─────────────────────────────────────────────┐
│         SAME PROMPT (both models)           │
│                                             │
│  [System]: You are playing Wordle...        │
│  [User]: Make a new guess. Past: CRANE...   │
│  [Assistant]: Let me solve step by step...  │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌──────────────────┐
│ BASE MODEL    │       │ FINE-TUNED MODEL │
│ adapter_id="" │       │ adapter_id=      │
│               │       │ "wordle-dlai/2"  │
└───────┬───────┘       └────────┬─────────┘
        │                        │
        ▼                        ▼
  "Maybe CRATE?"           "Based on C-R-A 
  (less strategic)          being correct,
                            try CRAFT"
                            (better strategy)
```

## Usage

Run the script to see both models in action:

```bash
python Wordle.py
```

To compare baseline vs fine-tuned model, uncomment lines 342-345 in `Wordle.py`:

```python
# Uncomment to see fine-tuned model comparison:
print("FINE-TUNED MODEL RESPONSE:")
print("-" * 100)
ft_completion = generate_stream(prompt, adapter_id="wordle-dlai/2")
print("-" * 100 + "\n")
```

# LLM as judge
```
ORIGINAL TRANSCRIPT
        ↓
    ┌───┴────┐
    ↓        ↓
[Step 1]  [Step 2]
Generate  Create Quiz
8 Summaries  (GT questions)
    ↓           ↓
    └─────┬─────┘
          ↓
      [Step 3]
   LLM takes quiz
   using ONLY summary
          ↓
      [Step 4]
   Score answers
   (reward = accuracy)
```