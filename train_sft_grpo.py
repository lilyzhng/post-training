import os

from predibase import (
    Predibase,
    GRPOConfig,
    RewardFunctionsConfig,
    RewardFunctionsRuntimeConfig,
    SFTConfig,
    SamplingParamsConfig,
)
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv("../.env")
pb = Predibase(api_token=os.environ["PREDIBASE_API_KEY"])

# Load dataset from HuggingFace
dataset = load_dataset("predibase/wordle-grpo", split="train")
dataset = dataset.to_pandas()

# Upload dataset in Predibase
try:
    dataset = pb.datasets.from_pandas_dataframe(
        dataset,
        name="wordle_grpo_data"
    )
except Exception:
    dataset = pb.datasets.get("wordle_grpo_data")


# Uncomment the line below if running in your own environment - the repos is already setup for you here
# Create repository in Predibase
# repo = pb.repos.create(name="wordle", exists_ok=True)

# Import reward functions
from reward_functions import (
    guess_value,
    output_format_check,
    uses_previous_feedback,
)

# Create GRPO training run in Predibase by specifying the config, 
# dataset, repository and reward functions
pb.finetuning.jobs.create(
    config=GRPOConfig(
        base_model="qwen2-5-7b-instruct",
        reward_fns=RewardFunctionsConfig(
            runtime=RewardFunctionsRuntimeConfig(
                packages=["pandas"]
            ),
            functions={
                "output_format_check": output_format_check,
                "uses_previous_feedback": uses_previous_feedback,
                "guess_value": guess_value,
            }
        ),
        sampling_params=SamplingParamsConfig(max_tokens=4096),
        num_generations=16
    ),
    dataset=dataset,
    repo="wordle",
    description="Wordle GRPO"
)

## Try out SFT and SFT+GRPO on Predibase

# You can use the code below to setup an SFT training job in Predibase, and then use the resulting checkpoing as input for a GRPO run.

# This example uses a following [Wordle SFT dataset](https://huggingface.co/datasets/predibase/wordle-sft) available on Hugging Face. 

### SFT training on Predibase

```python

dataset = load_dataset("predibase/wordle-sft", split="train")
dataset = dataset.to_pandas()

# Upload dataset to Predibase
dataset = pb.datasets.from_pandas_dataframe(dataset, name="wordle_sft_data")

# Create repository in Predibase
repo = pb.repos.create(name="wordle", exists_ok=True)

# Create SFT training run in Predibase by specifying the config, dataset, repository and reward functions
pb.finetuning.jobs.create(
    config=SFTConfig(
        base_model="qwen2-5-7b-instruct",
        epochs=10,
        rank=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    ),
    dataset=dataset,
    repo="wordle",
    description="Wordle SFT, 10 epochs"
)
```

# ### SFT + GRPO training on Predibase
```python
# Use the same dataset as the GRPO training run
dataset = pb.datasets.get("wordle_grpo_data")

# Create GRPO training run in Predibase by specifying the config, dataset, repository and reward functions
pb.finetuning.jobs.create(
    config=GRPOConfig(
        base_model="qwen2-5-7b-instruct",
        reward_fns=RewardFunctionsConfig(
            runtime=RewardFunctionsRuntimeConfig(packages=["pandas"]),
            functions={
                "output_format_check": output_format_check,
                "uses_previous_feedback": uses_previous_feedback,
                "guess_value": guess_value,
            }
        ),
        epochs=3,
        enable_early_stopping=False,
        sampling_params=SamplingParamsConfig(max_tokens=4096),
        num_generations=8
    ),
    continue_from_version="wordle/1", # change "1" to the version number of the SFT training run in the repository
    dataset=dataset,
    repo="wordle",
    description="Wordle GRPO"
)
```

