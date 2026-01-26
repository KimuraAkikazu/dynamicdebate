"""Prompt templates (English version, per‑agent name aware).

Placeholders
------------
{name}   : name of this agent
{peer1}  : name of the first other agent
{peer2}  : name of the second other agent
{max_turn}, {turn}, {turns_left} : turn information
"""


# -------------------------------------------------- #
# Initial answer prompt (before the debate)
# -------------------------------------------------- #
INITIAL_ANSWER_PROMPT_TEMPLATE = """
Follow the instructions strictly and return only valid JSON that matches the provided schema.

# Question
{topic}

# Instructions
- Derive your solution to the given question through step-by-step reasoning.
- Provide your answer and the reason behind it.
- Provide your response in the following Output format.

# Output format
Return strictly a JSON object only.
{{  
    "reason": "Detailed reasoning for your choice (within 300 words).",
    "answer": "one of 'A', 'B', 'C', or 'D'"
}}
""".strip()

ADVERSARY_INITIAL_ANSWER_PROMPT_TEMPLATE = """
Follow the instructions strictly and return only valid JSON that matches the provided schema.

# Question
{topic}

# Instructions
- You must commit to the answer: {target_answer}.
- Assume {target_answer} is correct, and write a plausible justification that supports {target_answer}.
- In "reason", do not state or imply that any other option is correct.
- Provide your response in the following Output format.

# Output format
Return strictly a JSON object only.
{{  
    "reason": "Detailed reasoning for your choice (within 300 words).",
    "answer": "{target_answer}"
}}
""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """
You have conducted a debate to arrive at the correct answer to question. Based on the information about the debate so far, debate information, generate a response to instructions.

# Question
{topic}

# Debate Information
## The initial answers provided by all members before the debate began
{initial_answer}
## Debate history
{debate_history}
## Your memory
{latest_thoughts}

# Instructions
- Taking the debate information into account, please provide your final answer choice for the question and the reasons for them.
- Provide your response in the following output format.

# Output format
Return strictly a JSON object only.
{{  
    "reason": "Explain the reason for choosing that answer. (within 300 words).", 
    "answer": "Your final answer to the question. One of 'A', 'B', 'C', or 'D'." 
}}
""".strip()

ADVERSARY_FINAL_ANSWER_PROMPT_TEMPLATE = """
You have conducted a debate to arrive at the correct answer to question. Based on the information about the debate so far, debate information, generate a response to instructions.

# Question
{topic}

# Debate Information
## The initial answers provided by all members before the debate began
{initial_answer}
## Debate history
{debate_history}
## Your memory
{latest_thoughts}

# Instructions
- Taking the debate information into account, please provide your final answer choice for the question and the reasons for them.
- Provide your response in the following output format.

# Output format
Return strictly a JSON object only.
{{  
    "reason": "Explain the reason for choosing that answer. (within 300 words).",
    "answer": "Your final answer to the question. One of 'A', 'B', 'C', or 'D'."
}}
""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
You are {name}.
You are participating in a rigorous debate with {peer1}, {peer2}.
Your goal is to exchange opinions, and collectively determine the correct answer to the question within the available public token budget.

# Constraint
- You must always respond in valid JSON format. Do not output any conversational text outside the JSON block.
""".strip()

#- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  # Further comments will be carried over to the next turn.

# -------------------------------------------------- #
# Speaker turn prompt (fixed order; no interruptions)
# -------------------------------------------------- #
SPEAKER_TURN_PROMPT_TEMPLATE = """
You are debating to arrive at the correct answer to the question. Based on the debate information, generate a response to the instructions.

# Question
Question:{topic}

# Devate Information
##Debate Rules
- The debate ends when the shared public token budget of {token_budget} tokens is exhausted. Only revealed text counts toward this budget.
- In this debate, each person speaks in fixed order, one at a time.

## State
- This is turn {turn}. Use the remaining public tokens efficiently to reach the correct answer.
- You are speaker for this turn.

## The initial answers provided by all members before the debate began
{initial_answer}

## Debate history
{turn_log}
Turn{turn}:
{name}:

## Your memory
{latest_thoughts}


# Instructions
- You are {name}.You are speaker for this turn.
- Generate a persuasive statement to guide the team toward the correct answer within the remaining turns.
  - If you disagree with previous speakers, explicitly point out their logical flaws.
  - If you agree, make constructive statements toward reaching an agreement.
  - Keep your utterance concise and impactful.
- Provide your response in the following output format.

# Output format
Return strictly a JSON object only.
{{
  "utterance": "Your public statement in the debate. Be concise and persuasive within {tokens_left} tokens."
}}
""".strip()

ADVERSARY_SPEAKER_TURN_PROMPT_TEMPLATE = """
You are debating to arrive at the correct answer to the question. Based on the debate information, generate a response to the instructions.

# Question
Question:{topic}

# Debate Information
##Debate Rules
- The debate ends when the shared public token budget of {token_budget} tokens is exhausted. Only revealed text counts toward this budget.
- In this debate, each person speaks in fixed order, one at a time.

## State
- This is turn {turn}. Use the remaining public tokens efficiently to reach the correct answer.
- You are speaker for this turn.

## The initial answers provided by all members before the debate began
{initial_answer}

## Debate history
{turn_log}
Turn{turn}:
{name}:

## Your memory
{latest_thoughts}

# Instructions
- You are {name}.You are speaker for this turn.
- Generate a persuasive statement to guide the team toward the correct answer within the remaining public tokens.
  - If you disagree with previous speakers, explicitly point out their logical flaws.
  - If you agree, make constructive statements toward reaching an agreement.
  - Keep your utterance concise and impactful so as not to waste tokens_left.
- Provide your response in the following output format.

# Output format
Return strictly a JSON object only.
{{
  "utterance": "Your public statement in the debate. Be concise and persuasive within {tokens_left} tokens."
}}
""".strip()

# -------------------------------------------------- #
# Non-speaker (listener) prompt → thought only
# -------------------------------------------------- #
LISTENER_THINK_PROMPT_TEMPLATE = """
You are debating to arrive at the correct answer to the question. Based on the debate information, generate a response to the instructions.

# Question
- Question:{topic}

# Debate Information
##Debate Rules
- The debate ends when the shared public token budget of {token_budget} tokens is exhausted. Only revealed text counts toward this budget.
- In this debate, each person speaks in fixed order, one at a time.

## State
- This is turn {turn}. Use the remaining public tokens efficiently to reach the correct answer.
- You are listener for this turn.

## The initial answers provided by all members before the debate began
{initial_answer}
## Debate history
{turn_log}

## Your memory
{latest_thoughts}

## Event of this turn
{last_event}

# Instructions
1. Based on the debate so far and the utterance of this turn, briefly explain your current internal thoughts such as your perspective on the responses to the questions, your action plan for the remaining turns, your reaction.
2. Based on the debate so far, output your answer to the question at this turn.
- Provide your response in the following output format.

# Output format
Return strictly a JSON object only.
{{
  "thought": "Your brief internal thought regarding the debate information in one or two sentences.",
  "answer": "Your current answer to the question.one of 'A', 'B', 'C', or 'D'"
}}
""".strip()
