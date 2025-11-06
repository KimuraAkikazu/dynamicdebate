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

- The initial answers provided by all members before the discussion began:
{initial_answer}
- Debate history:
{debate_history}
- Your thoughts up until the previous turn:
{latest_thoughts}



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
- You are debating a given topic with {peer1} and {peer2}.


""".strip()
# Your personality:
# {persona}
#- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  # Further comments will be carried over to the next turn.
#- `speak`    : Begin speaking yourself because you judge the current speaker has finished speaking.

# -------------------------------------------------- #
# Plan‑action prompt (normal turn)
# -------------------------------------------------- #
PLAN_ACTION_PROMPT_TEMPLATE = """
Debate rules:
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn. The next speaker is selected from those who chose “interrupt”, based on the highest urgency level.
- Each turn, one chunk at a time from the speaker's generated statement is revealed to all members. Therefore, since the speaker can prepare multiple chunks during the preparation phase, the statement for this turn may have a continuation.

Debate state:
- You are {name}.
- This is turn {turn}.
- You have {turns_left} speaking opportunity(ies) remaining.
- Decide on your final answer within the remaining {turns_left} turns.

Debate topic:
{topic}

The initial answers provided by all members before the discussion began:
{initial_answer}

Debate so far:
{turn_log}

Your thoughts up until the previous turn:
{latest_thoughts}

Event of this turn:
{last_event}

You can take the following actions:
- `listen`   : Use this when you listen to the speaker's argument until the end to deepen your understanding before speaking.
- `interrupt`: Use this when you interrupt the current speaker to begin speaking.

Instructions:
- Your goal is to work together to arrive at the correct answer to the given question.
- Refer to the provided discussion information and create an action plan for the next turn as {name} in the specified format.
- While considering the possibility that someone may be mid-sentence, decide whether to interrupt and respond immediately to this turn's statement or listen to its completion.


Constraints:
- There is only one correct answer choice for the question.
- Be careful not to stray into discussions that are not necessary for answering the question.

Output format:
{{ 
 "thought": "string", // Based on the debate so far and the utterance of this turn, briefly describe your current thoughts.To reach consensus, consider whether to push your position or align with others, and choose accordingly. 
 "action": "listen|interrupt", // Select the next action you should take. Interrupting may derail the discussion, so when you have sufficient information for a comprehensive response or when the current statement contains errors or misunderstandings.
 "urgency": 0-9, // Based on the “thought” for this turn, how urgent is it for you to start speaking now? Output a numeric value indicating the urgency.
 "intent": "agree|disagree|summarize|confirmation|proposal|conclusion|think", // Select the purpose of the chosen action.
 "answer": "A|B|C|D" // Based on the discussion, please select your updated answer to the question at this time.
 "agreed": true|false, // If you judge that your answer matches another member's and consensus has been reached, select “true”.
  }}
""".strip()



# --------------------------------------------------
# Plan‑action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """
Debate rules:
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn. The next speaker is selected from those who chose “speak”, based on the highest urgency level.
- Each turn, one chunk at a time from the speaker's generated statement is revealed to all members. Therefore, since the speaker can prepare multiple chunks during the preparation phase, the statement for this turn may have a continuation.

Debate state:
- You are {name}.
- This is turn {turn}.
- You have {turns_left} speaking opportunity(ies) remaining.
- Decide on your final answer within the remaining {turns_left} turns.

Debate topic:
{topic}

The initial answers provided by all members before the discussion began:
{initial_answer}

Debate so far:
{turn_log}

Your thoughts up until the previous turn:
{latest_thoughts}

Event of this turn:
{last_event}

You can take the following actions:
- `listen`   : Use this when listening to someone's statement.
- `speak`: Use this when beginning to make a point to advance the discussion.

Instructions:
- Your goal is to work together to arrive at the correct answer to the given question.
- Refer to the provided discussion information and create an action plan for the next turn as {name} in the specified format.

Constraints:
- Please bear in mind that prolonged silence hinders progress in discussions.
- Be careful not to stray into discussions that are not necessary for answering the question.


Output format:
{{ 
 "thought": "string", // Based on the debate so far and the event of this turn, briefly describe your current thoughts.To reach consensus, consider whether to push your position or align with others, and choose accordingly. 
 "urgency": 0-9, // Based on the “thought” for this turn, how urgent is it for you to start speaking now? Output a numeric value indicating the urgency.
 "action": "listen|speak", // Select the next action you should take.
 "intent": "agree|disagree|summarize|confirmation|proposal|conclusion|think", // Select the purpose of the chosen action.
 "answer": "A|B|C|D" // Based on the discussion, please select your updated answer to the question at this time.
 "agreed": true|false, // If you judge that your answer matches another member's and consensus has been reached, select “true”.
  }}
""".strip()

# # # All actions:
# - `listen`   : Focus on listening to move the discussion forward
# - `speak`    : Start talking when no one else is speaking.
# - `interrupt`: Interrupting someone else while they are speaking

# --------------------------------------------------
# Utterance‑generation prompt
# --------------------------------------------------
GENERATE_UTTERANCE_PROMPT_TEMPLATE = """
# Debate rules
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn.

# Question
{topic}

# Context
- The initial answers provided by all members before the discussion began:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate so far:
<DEBATE_SO_FAR>
{turn_log}
Turn{turn}({event_type})
{name}:
</DEBATE_SO_FAR>
- This is turn {turn}.
- You have {turns_left} chance(s) to speak left.
- Decide on your final answer within {turns_left} turns remaining.

# Instruction
- You are speaking in the debate as {name}.
- Your goal is to collectively decide on the correct answer to the question within the maximum number of turns.
- Your thoughts on speaking next:
  your thought:{thought},
  intention of your action:{intent}
- To reach consensus, consider whether to push your position or align with others, and choose accordingly.
- Based on your personality, generate your utterance to be made as {name} that builds on the discussion so far.

# Constraints
- Be careful not to stray into discussions that are not necessary for answering the question.
- Be careful not to repeat the same thing over and over again in discussions.

# Output format
{{"utterance": "string"}}
""".strip()

