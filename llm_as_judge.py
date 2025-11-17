# Quiz score = reward function for reinforcement learning to improve summary quality! 
# Q: If we're using the quiz score as a reward function to improve the summarization quality, this quiz is essentially the ground truth. Then what's the difference between supervised tuning, which uses the quiz directly, versus this quiz score using a reward function for reinforcement learning?
# A: no, The Quiz is NOT Ground Truth for the Summary. The quiz judges quality but doesn't prescribe HOW to write the summary - that's what the model learns through RL exploration! 

import os
import re
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from utils import (
    MODEL_NAME,
    compute_advantages,
    print_quiz_table,
    tabulate,
)

load_dotenv("../.env")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

pb_client = OpenAI(
    base_url = os.environ["PREDIBASE_MODEL_QWEN_URL"],
    api_key = os.environ["PREDIBASE_API_KEY"],
)

# ============================================================================
# Task: creating summaries of earnings call transcripts
# ============================================================================

# start by loading the earnings call dataset from Hugging Face
ds = load_dataset("mrSoul7766/ECTSum")
transcript = ds["train"][1]["text"]
print(transcript[:1983])

# define a summarize prompt and helper function, then create and print a summary
# Note: the MODEL_NAME is specified in the utils.py file: here you are using Llama-3.1-8B-instruct dequantized to generate summaries

SUMMARIZE_PROMPT = f"""Generate a concise summary of the information in the following earning call transcript.
Only respond with the summary, do not include any extraneous text.

Transcript:

{transcript}
"""

def summarize(transcript: str, n: int = 1) -> str:
    prompt = SUMMARIZE_PROMPT.format(transcript=transcript)
    messages = [
        {"role": "user",
        "content": prompt,
        }
    ]
    return pb_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        n=n,
        temperature=0.9,
    )

resp = summarize(transcript)
summary = resp.choices[0].message.content
print(summary)

# ============================================================================
# Use an LLM as a judge of summary quality
# ============================================================================
# Define a prompt that will tell the OpenAI GPT-4o-mini model to assign a reward score to a summary
JUDGE_PROMPT_V1 = """
Rate the following summary of an earnings call transcript on a scale from 1 to 10.

1 means the summary is very poor, 10 means the summary is very good.

Provide reasoning followed by the final score at end surrounded by <score> tags.

For example:
<score>1</score>

Transcript:
{transcript}

Summary:
{summary}
"""

def judge_reward_v1(
    transcript: str,
    summary: str,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> float:
    prompt = JUDGE_PROMPT_V1.format(transcript=transcript, summary=summary)
    messages = [
        {"role": "user",
        "content": prompt,
        }
    ]
    resp = pb_client.chat.completions.create(
        model=model,
        messages=messages,
        n=1,
        temperature=0.0,
    )

    completion = resp.choices[0].message.content

    if verbose:
        print(completion)

    try:
        match = re.search(r"<score>(\d+)<\/score>", completion)
        if match is None:
            return 0
        # Extract the "score" part from the completion
        score = match.group(1).strip()
        score = int(score)
    except:
        score = 0
    
    return score / 10

# now score the summary you generated above using the new reward function
score = judge_reward_v1(transcript, summary, verbose=True)
print(f"Score: {score}")

# now generate 8 new summaries and score each one
resp = summarize(transcript, n=8)
summaries = [choice.message.content for choice in resp.choices]
scores = [judge_reward_v1(transcript, summary, verbose=True) for summary in summaries]
print(scores)

# ============================================================================
# Take a quiz to assign a reward score
# ============================================================================
"""
In this section, you will create a multiple choice quiz that tests key facts from the earnings call transcript.
You will then ask another LLM to take the quiz using different call summariesm and use the quiz score as the reward score.
"""

# create the quiz prompt
from pydantic import BaseModel
from random import shuffle

QUIZ_PROMPT = """
Generate a multiple-choice quiz based on the information in the following earnings call transcript.

Example:
```
1. What was the q1 adjusted earnings per share?
a) $3.34
b) $3.35
c) $2.49
d) $7.78

2. By what percent did same store sales rise in q1?
a) 29.4%
b) 32.1%
c) 24.7%
d) 21.2%

==== ANSWERS ====
1. a
2. c
```

Limit the length of the quiz to top 10 most relevant questions for the financial analysis.

Transcript:
{transcript}
"""

"""
Next, define pydantic classes that define the structure of an individyal question, and a quiz comprised of multiple questions. Then define a helper function to create a quiz using structured response from GPT-4o-mini.
"""

class Question(BaseModel):
    text: str
    options: list[str]
    answer: str

    def shuffle_options(self) -> None:
        """Shuffle the options while preserving the correct answer."""
        # Get the correct answer text
        correct = self.options[self.answer]

        # Shuffle the options
        shuffled = self.options.copy()
        shuffle(shuffled)

        # Update the answer index to match new position
        self.options = shuffled
        self.answer = shuffled.index(correct)
    
    def __str__(self) -> str:
        """Pretty print the question."""
        output = [self.text]
        for i, option in enumerate(self.options):
            output.append(f"{chr(65 + i)}) {option}")
        return "\n".join(output)

    def __repr__(self) -> str:
        return self.__str__()


class Quiz(BaseModel):
    questions: list[Question]

    def shuffle_all_questions(self) -> None:
        """Shuffle all questions while preserving the correct answer."""
        for question in self.questions:
            question.shuffle_options()

    def __str__(self) -> str:
        """Pretty print the quiz."""
        output = []
        for i, question in enumerate(self.questions, 1):
            output.append(f"\nQuestion {i}")
            output.append(question.__str__())
        return "\n".join(output)
    
    def __repr__(self) -> str:
        return self.__str__()

def create_quiz(transcript: str, model: str = "gpt-4o-mini") -> Quiz:
    prompt = QUIZ_PROMPT.format(transcript=transcript)
    messages = [
        {"role": "user",
        "content": prompt,
        }
    ]
    resp = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.7,
        response_format=Quiz,
    )
    quiz = resp.choices[0].message.parsed
    quiz.shuffle_all_questions()
    return quiz

quiz = create_quiz(transcript)
print(quiz)

# ============================================================================
# Now, define a function that asks an LLM to take a quiz, using a transcript summary as the source material
# ============================================================================
letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
index_to_letter = ["A", "B", "C", "D"]

TAKE_QUIZ_PROMPT = """
Use the provided summary of a transcript to answer the following quiz.

Quiz:
{quiz}

Summary:
{summary}

Respond wih just a list of answers, and no additional text.

for example:
[A, D, C, B, B, C, D, A, A, B]

You must provide an answer for all 10 questions.
If you don't know the answer, answer with "0" for that question.

Example:
[A, D, 0, B, 0, C, 0, A, 0, B]
"""

def take_quiz(quiz: Quiz, summary: str, model: str = "gpt-4o-mini") -> float:
    questions_strs = []

    for question in quiz.questions:
        question_str = question.text
        for i, option in enumerate(question.options):
            letter = index_to_letter[i]
            questions_strs += f"\n{letter}) {option}"
        
        questions_strs.append(question_str)

    quiz_str = "\n".join(questions_strs)

    prompt = TAKE_QUIZ_PROMPT.format(quiz=quiz_str, summary=summary)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user",
            "content": prompt,
            }
        ],
        temperature=0.0,
    )
    resp_str = resp.choices[0].message.content
    answers = resp_str.strip("[]").split(",")

    return answers

answers = take_quiz(summaries[0], quiz)
print(answers)

# ============================================================================
# Finally, score the LLM answers to the quiz
# ============================================================================
def score_quiz_answers(answers, quiz):
    assert len(answers) == len(quiz.questions)

    total = len(answers)
    correct = 0
    for answer, question in zip(answers, quiz.questions):
        expected_answer = index_to_letter[question.answer]
        if answer == expected_answer:
            correct += 1
    
    return correct / total

score = score_quiz_answers(answers, quiz)
print(f"Score: {score}")

# Generate rewards and advantages for all 8 summaries you created earlier
def print_quiz_table(all_answers, rewards):
    advantages = compute_advantages(rewards)
    length = len(all_answers)
    elems = list(zip(range(length), all_answers, rewards, advantages))

    headers = ["Index","Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid").split("\n")
    for row in table:
        print(row)