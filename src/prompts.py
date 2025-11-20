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

<Question>
Question: {topic}
</Question>

Instructions:
- Derive your solution to the given <Question> through step-by-step reasoning.
- Provide your answer and the reason behind it.
- Provide your response in the following <Output format>.
- Return only JSON. No prose, no markdown.

<Output format>
```json
{{  
    "reason": "Explain the reason for choosing that answer.", 
    "answer": "One of A, B, C, or D."  
}}
```
</Output format>
""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """
- You are {name}.
- Your goal is to decide on the correct answer to the <Question>.
- You cooperated with two other members and engaged in a debate to derive a single answer to <Question>.
- You conducted a debate to arrive at the correct answer to <Question>. Based on <Debate Information> regarding about the debate so far, generate a response to <Instructions>.

<Question>
Question:{topic}
</Question>

<Debate Information>
- The initial answers provided by all members before the debate began:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>

- Debate history:
<DEBATE_HISTORY>
{debate_history}
</DEBATE_HISTORY>
</Debate Information>

<Instructions>
- Based on your <INITIAL_ANSWERS> and the <DEBATE_HISTORY>, please provide your final answer choice for <Question> and the reasoning behind it.
- Provide your response in the following <Output format>.
- Return only JSON. No prose, no markdown.
</Instructions>

<Output format>
```json
{{  
    "reason": "Explain the reason for choosing that answer.",
    "answer": "One of A, B, C, or D.",
}}
```
</Output format>
""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
- You are {name}.
- You are discussing with {peer1}, {peer2}.

""".strip()

#- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  # Further comments will be carried over to the next turn.

# -------------------------------------------------- #
# Speaker turn prompt (fixed order; no interruptions)
# -------------------------------------------------- #
SPEAKER_TURN_PROMPT_TEMPLATE = """
- You are {name}. 
- Your goal is to collectively decide on the final answer to <Question> within remaining {turns_left} turns.
- You are conducting a debate to arrive at the correct answer to <Question>. Based on the information about the debate so far, <Debate Information>, generate a response to <Instructions>.

<QUESTION>
Question:{topic}
</QUESTION>

<Debate Information>
Debate state:
- This is turn {turn} in total of {max_turn} turns.
- In this debate, each person speaks in turn, one at a time.
- You are speaker for this turn.

- The initial answers provided by all members before the debate began:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>

- Debate so far:
<DEBATE_SO_FAR>
{turn_log}
Turn{turn}:
{name}:
</DEBATE_SO_FAR>
</Debate Information>

Instructions:
- Please generate a statement to guide the team to the correct answer as {name} following <DEBATE_SO_FAR>.
- Provide your response in the following <Output format>.
- Return only JSON. No prose, no markdown.

<Output format>
```json
{{
  "utterance": "The content of your statement directed at other members this turn."
}}
```
</Output format>
""".strip()

# -------------------------------------------------- #
# Non-speaker (listener) prompt → thought only
# -------------------------------------------------- #
LISTENER_THINK_PROMPT_TEMPLATE = """
- You are {name}. 
- Your goal is to collectively decide on the final answer to <Question> within remaining {turns_left} turns.
- You are conducting a debate to arrive at the correct answer to <Question>. Based on the information about the debate so far, <Debate Information>, generate a response to <Instructions>.

<Question>
- Question:{topic}
</Question>

<Debate Information>
Debate state:
- This is turn {turn} in total of {max_turn} turns.
- In this debate, each person speaks in turn, one at a time.
- You are listener for this turn.

- The initial answers provided by all members before the debate began:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>

- Debate so far:
<DEBATE_SO_FAR>
{turn_log}
</DEBATE_SO_FAR>
</Debate Information>

<Instructions>
- After listening to the speaker's remarks this turn, please share your current internal thoughts—such as your perspective on the question or your feelings.
- Select your current answer to <Question> at this turn.
- Set "belief_team_consensus" to true ONLY if:
  - You believe the TEAM has effectively converged to one answer,
  - There are no major unresolved objections in the debate so far.
- Provide your response in the following <Output format>.
- Return only JSON. No prose, no markdown.
</Instructions>

<Output format>
```json
{{
  "thought": "string",
  "current_answer": "one of 'A', 'B', 'C', or 'D'",
  "belief_team_consensus": true or false
}}
```
</Output format>
""".strip()