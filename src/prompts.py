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
- You are {name}.
- Follow the instructions strictly and return only valid JSON that matches the provided schema.

# Instruction
- Derive your solution to the given question through step-by-step reasoning.
- Provide your answer before the discussion begins and the reasoning behind it.
- Output JSON only with two keys: "reason" and "answer".

# Question
Question: {topic}

- Answer must be one of A, B, C, or D.
- Return only JSON. No prose, no markdown.
""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """
- You are {name}.
- You cooperated with two other members and engaged in a discussion to derive a single answer (A–D) to a multiple-choice question.
- Your goal is to collectively decide on the answer to the question.

# Context
# Question
Question: {topic}

- initial answer of all members
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate history
<DEBATE_HISTORY>
{debate_history}
</DEBATE_HISTORY>

# Instruction
- Refer to your initial answer and debate history,output team's collective answer and reason for the choice.
- Output JSON only with two keys: "reason" and "answer".

# Output format
```json
{{  
    "reason": "string", // The reasoning and thought process that ultimately led to selecting that answer after concluding the discussion.
    "answer": "string", // answer to the question, one of A, B, C, D  
}}

""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
- You are {name}.
- You are discussing with {peer1} and {peer2} which of the given options is the correct answer to the question.

""".strip()

#- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  # Further comments will be carried over to the next turn.

# -------------------------------------------------- #
# Speaker turn prompt (fixed order; no interruptions)
# -------------------------------------------------- #
SPEAKER_TURN_PROMPT_TEMPLATE = """
- You are {name}. You are the SPEAKER for this turn.
- The goal is to help the team converge to a single correct option (A–D) by the end of the debate.

# Question
<QUESTION>
{topic}
</QUESTION>

# Initial answers (shared before the debate)
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>

# Debate so far 
<DEBATE_SO_FAR>
{turn_log}
</DEBATE_SO_FAR>

# Turn info
- This is turn {turn} in total.
- You will speak exactly once this turn.
- You have {turns_left} speaking opportunity(ies) remaining after this turn.

# Instruction
- Produce a concise utterance that advances the team toward a single final answer.
- Be precise and avoid repetition.
- Output JSON only.

# Output format
{{
  "utterance": "string"
}}
""".strip()

# -------------------------------------------------- #
# Non-speaker (listener) prompt → thought only
# -------------------------------------------------- #
LISTENER_THINK_PROMPT_TEMPLATE = """
- You are {name}. You are NOT the speaker for this turn.
- The goal is to help the team converge to a single correct option (A–D) by the end of the debate.

# Question
<QUESTION>
{topic}
</QUESTION>

# Initial answers (shared before the debate)
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>

# Debate so far
<DEBATE_SO_FAR>
{turn_log}
</DEBATE_SO_FAR>

# Instruction
- Reflect internally on the debate so far. Do NOT produce a public utterance.
- Based on the debate so far, briefly explain your current inner thoughts.
- Output JSON only.

# Output format
{{
  "thought": "string"
}}
""".strip()