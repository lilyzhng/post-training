# post-training

## The Two Models
1. Baseline Model (Out-of-the-box)
Model: Qwen/Qwen2.5-7B-Instruct
What it is: The original pre-trained instruction-following model
When it's used: When adapter_id="" (empty string)
Performance: May not be optimized for Wordle gameplay

2. Fine-Tuned Model (Post-Training)
Model: Qwen/Qwen2.5-7B-Instruct + "wordle-dlai/2" adapter
What it is: The base model + a LoRA adapter trained specifically on Wordle examples
When it's used: When adapter_id="wordle-dlai/2"
Performance: Should be better at Wordle reasoning and strategy

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
                            tryCRAFT"
                            (better strategy)