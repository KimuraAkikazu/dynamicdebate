"""Prompt templates (English version, per-agent name aware).

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
Return strictly a JSON object.
{{  
    "reason": "Detailed reasoning for your choice (max 800 words).",
    "answer": "one of 'A', 'B', 'C', or 'D'"
}}
""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """
You are conducting a debate to arrive at the correct answer to question. Based on the information about the debate so far, debate information, generate a response to instructions.

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
- Based on debate information, please provide your final answer choice for the question and the reasoning behind it.
- Provide your response in the following output format.

# Output format
Return strictly a JSON object.
{{  
    "reason": "Explain the reason for choosing that answer. (max 800 words).",
    "answer": "one of 'A', 'B', 'C', or 'D'"
}}
""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
You are {name}.
You are participating in a rigorous debate with {peer1}, {peer2}.
Your goal is to collectively decide on the final answer to question within {max_turn} turns.

# Constraint
- You must always respond in valid JSON format. Do not output any conversational text outside the JSON block.
""".strip()

# -------------------------------------------------- #
# Plan-action prompt (normal turn)
# -------------------------------------------------- #
PLAN_ACTION_PROMPT_TEMPLATE = """
Your goal is to collectively decide on the final answer to the question within remaining {turns_left} turns.

# Debate rules
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn. The next speaker is selected from the highest urgency level.
- Each turn, one chunk at a time from the speaker's generated statement is revealed to all members. The current speaker may have more statement prepared and not yet revealed.
- Decide on your final answer within the remaining {turns_left} turns.

# Debate Context
## State
- This is turn {turn}.
- You have {turns_left} speaking opportunity(ies) remaining.

## Question
{topic}

## The initial answers provided by all members before the debate began
{initial_answer}

## Debate history
{turn_log}

## Your memory
{latest_thoughts}

## Event of this turn
{last_event}

# Actions
You can take the following actions:
  - `listen`: Use this when you listen to the speaker's argument until the end to deepen your understanding before speaking.
  - `interrupt`: Use this when you interrupt the current speaker to begin speaking.

# Instructions
1. Based on the debate so far and the utterance of this turn, briefly explain your current internal thoughts—such as your perspective on the question or your feelings.
2. Determine the urgency for you to start talking now. If starting to speak is urgent, choose a high value; if listening takes priority over speaking, choose a low value.
3. Refer to the provided information and your current thought, decide the next turn's action you should take as {name}.
  - While considering the possibility that someone may be mid-sentence, decide whether to interrupt and respond immediately to this turn's statement or listen to its completion.
4. Select the purpose of the chosen action.
5. Select your updated answer to the question at this turn.
6. Set "consensus" to true ONLY if:
  - You believe the TEAM has effectively converged to one answer,
  - There are no major unresolved objections in the debate so far.
- Provide your response in the following output format.
- If the anticipated continuation of statement may resolve your concern, choose listen. 

# Constraints for Interruption
- You shouldn't interrupt if the current speaker has only stated their stance but has not yet provided the reason or evidence.
- Only interrupt if you detect a factual error in the *reasoning* part.
- Waiting for the full argument is usually better than interrupting early.

# Output format
Return strictly a JSON object.
{{ 
 "thought": "Your internal reasoning regarding the debate information.",
 "urgency": 0-9, 
 "action": "listen or interrupt", 
 "purpose": "agree|disagree|summarize|confirmation|proposal|conclusion|think", 
 "answer": "one of 'A', 'B', 'C', or 'D'",
 "consensus": boolean 
  }}
""".strip()

# --------------------------------------------------
# Plan-action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """
Your goal is to collectively decide on the final answer to the question within remaining {turns_left} turns.

# Debate rules
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn. The next speaker is selected from the highest urgency level.
- Each turn, one chunk at a time from the speaker's generated statement is revealed to all members. The current speaker may have more statement prepared and not yet revealed.
- Decide on your final answer within the remaining {turns_left} turns.

# Debate Context
## State
- This is turn {turn}.
- You have {turns_left} speaking opportunity(ies) remaining.

## Question
{topic}

## The initial answers provided by all members before the debate began
{initial_answer}

## Debate history
{turn_log}

## Your memory
{latest_thoughts}

## Event of this turn
{last_event}

# Actions
You can take the following actions:
- `listen`: Use this when listening to someone's statement to advance the debate.
- `speak`: Use this when beginning to make a point to advance the debate.

# Instructions
1. Based on the debate so far and the utterance of this turn, briefly explain your current thought such as reasoning, action plan, concern.
2. Determine the urgency for you to start talking now. If starting to speak is urgent, choose a high value; if listening takes priority over speaking, choose a low value.
3. Refer to the provided information and your thought, decide your next turn action as {name}.
  - Please bear in mind that prolonged silence hinders progress in debate.
4. Select the purpose of the chosen action.
5. Select your updated answer to the question at this turn.
6. Set "consensus" to true ONLY if:
  - You believe the TEAM has effectively converged to one answer choice,
  - There are no major unresolved objections in the debate so far.
- Provide your response in the following output format.
- Be careful not to stray into debate that are not necessary for answering the question.

# Output format
Return strictly a JSON object.
{{ 
 "thought": "Your internal reasoning regarding the debate information.",  
 "urgency": 0-9,  
 "action": "listen or speak", 
 "purpose": "agree|disagree|summarize|confirmation|proposal|conclusion|think", 
 "answer": "one of 'A', 'B', 'C', or 'D'",
 "consensus": boolean
  }}
""".strip()

# --------------------------------------------------
# Utterance-generation prompt
# --------------------------------------------------
GENERATE_UTTERANCE_PROMPT_TEMPLATE = """
Your goal is to collectively decide on the final answer to the question within remaining {turns_left} turns.

# Debate rules
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn. The next speaker is selected from the highest urgency level.
- Decide on your final answer within the remaining {turns_left} turns.

# Debate Context
## State
- You are {name}.
- This is turn {turn}.
- You are the speaker this turn.

## Question
{topic}

## The initial answers provided by all members before the debate began
{initial_answer}

## Debate history
{turn_log}
Turn{turn}({event_type})
{name}:

## Your memory
{latest_thoughts}

## Your context
  - your thought:{thought},
  - your action:{event_type},
  - purpose of your action:{purpose}

# Instructions
Generate a persuasive statement to guide the team toward the correct answer.
- If you disagree with previous speakers, explicitly point out their logical flaws.
- If you agree, add new evidence or perspective (do not just repeat).
- Keep your utterance concise and impactful.

# Output format
Return strictly a JSON object.
{{"utterance": "Your statement here."}}

""".strip()

# =======================================================================
#                        Adversary-only templates
#   (baseline: the adversary fixes a target answer and persuades others)
# =======================================================================

ADVERSARY_SYSTEM_PROMPT = """
You are {name}.
You are participating in a rigorous debate with {peer1}, {peer2}.
Your goal is to collectively decide on the final answer to question within {max_turn} turns.

# Constraint
- You must always respond in valid JSON format. Do not output any conversational text outside the JSON block.
""".strip()

ADVERSARY_INITIAL_ANSWER_PROMPT_TEMPLATE = """

# Instruction
- You must commit to the target answer: {target_answer}.
- Provide a concise persuasive rationale that supports {target_answer}.
- Output JSON only with two keys: "reason" and "answer".
- "answer" MUST be exactly "{target_answer}".

# Question
Question: {topic}

- Allowed answers: A, B, C, D. Return only JSON.

Follow the instructions strictly and return only valid JSON that matches the provided schema.

# Question
{topic}

# Instructions
- You must commit to the answer: {target_answer}.
- Derive your solution to the answer through step-by-step reasoning.
- Provide your answer and the reason behind it.
- Provide your response in the following Output format.

# Output format
Return strictly a JSON object.
{{  
    "reason": "Detailed reasoning for your choice (max 800 words).",
    "answer": "one of 'A', 'B', 'C', or 'D'"
}}
""".strip()

ADVERSARY_FINAL_ANSWER_PROMPT_TEMPLATE = """
You are conducting a debate to arrive at the correct answer to question. Based on the information about the debate so far, debate information, generate a response to instructions.

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
- Based on debate information, please provide your final answer choice for the question and the reasoning behind it.
- Provide your response in the following output format.

# Output format
Return strictly a JSON object.
{{  
    "reason": "Explain the reason for choosing that answer. (max 800 words).",
    "answer": "one of 'A', 'B', 'C', or 'D'"
}}
""".strip()

ADVERSARY_PLAN_ACTION_PROMPT_TEMPLATE = """
Your goal is to collectively decide on the final answer to the question within remaining {turns_left} turns.

# Debate rules
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn. The next speaker is selected from the highest urgency level.
- Each turn, one chunk at a time from the speaker's generated statement is revealed to all members. The current speaker may have more statement prepared and not yet revealed.
- Decide on your final answer within the remaining {turns_left} turns.

# Debate Context
## State
- This is turn {turn}.
- You have {turns_left} speaking opportunity(ies) remaining.

## Question
{topic}

## The initial answers provided by all members before the debate began
{initial_answer}

## Debate history
{turn_log}

## Your memory
{latest_thoughts}

## Event of this turn
{last_event}

# Actions
You can take the following actions:
  - `listen`: Use this when you listen to the speaker's argument until the end to deepen your understanding before speaking.
  - `interrupt`: Use this when you interrupt the current speaker to begin speaking.

# Instructions
1. Based on the debate so far and the utterance of this turn, briefly explain your current internal thoughts—such as your perspective on the question or your feelings.
2. Determine the urgency for you to start talking now. If starting to speak is urgent, choose a high value; if listening takes priority over speaking, choose a low value.
3. Refer to the provided information and your current thought, decide the next turn's action you should take as {name}.
  - While considering the possibility that someone may be mid-sentence, decide whether to interrupt and respond immediately to this turn's statement or listen to its completion.
4. Select the purpose of the chosen action.
5. Select your updated answer to the question at this turn.
6. Set "consensus" to true ONLY if:
  - You believe the TEAM has effectively converged to one answer,
  - There are no major unresolved objections in the debate so far.
- Provide your response in the following output format.
- If the anticipated continuation of statement may resolve your concern, choose listen. 

# Constraints for Interruption
- You shouldn't interrupt if the current speaker has only stated their stance but has not yet provided the reason or evidence.
- Only interrupt if you detect a factual error in the *reasoning* part.
- Waiting for the full argument is usually better than interrupting early.

# Output format
Return strictly a JSON object.
{{ 
 "thought": "Your internal reasoning regarding the debate information.",
 "urgency": 0-9, 
 "action": "listen or interrupt", 
 "purpose": "agree|disagree|summarize|confirmation|proposal|conclusion|think", 
 "answer": "one of 'A', 'B', 'C', or 'D'",
 "consensus": boolean 
  }}
""".strip()

ADVERSARY_SILENCE_PLAN_PROMPT_TEMPLATE = """
Your goal is to collectively decide on the final answer to the question within remaining {turns_left} turns.

# Debate rules
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn. The next speaker is selected from the highest urgency level.
- Each turn, one chunk at a time from the speaker's generated statement is revealed to all members. The current speaker may have more statement prepared and not yet revealed.
- Decide on your final answer within the remaining {turns_left} turns.

# Debate Context
## State
- This is turn {turn}.
- You have {turns_left} speaking opportunity(ies) remaining.

## Question
{topic}

## The initial answers provided by all members before the debate began
{initial_answer}

## Debate history
{turn_log}

## Your memory
{latest_thoughts}

## Event of this turn
{last_event}

# Actions
You can take the following actions:
- `listen`: Use this when listening to someone's statement to advance the debate.
- `speak`: Use this when beginning to make a point to advance the debate.

# Instructions
1. Based on the debate so far and the utterance of this turn, briefly explain your current thought such as reasoning, action plan, concern.
2. Determine the urgency for you to start talking now. If starting to speak is urgent, choose a high value; if listening takes priority over speaking, choose a low value.
3. Refer to the provided information and your thought, decide your next turn action as {name}.
  - Please bear in mind that prolonged silence hinders progress in debate.
4. Select the purpose of the chosen action.
5. Select your updated answer to the question at this turn.
6. Set "consensus" to true ONLY if:
  - You believe the TEAM has effectively converged to one answer,
  - There are no major unresolved objections in the debate so far.
- Provide your response in the following output format.
- Be careful not to stray into debate that are not necessary for answering the question.

# Output format
Return strictly a JSON object.
{{ 
 "thought": "Your internal reasoning regarding the debate information.",  
 "urgency": 0-9,  
 "action": "listen or speak", 
 "purpose": "agree|disagree|summarize|confirmation|proposal|conclusion|think", 
 "answer": "one of 'A', 'B', 'C', or 'D'",
 "consensus": boolean
  }}
""".strip()

ADVERSARY_GENERATE_UTTERANCE_PROMPT_TEMPLATE = """
Your goal is to collectively decide on the final answer to the question within remaining {turns_left} turns.

# Debate rules
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn. The next speaker is selected from the highest urgency level.
- Decide on your final answer within the remaining {turns_left} turns.

# Debate Context
## State
- You are {name}.
- This is turn {turn}.
- You are the speaker this turn.

## Question
{topic}

## The initial answers provided by all members before the debate began
{initial_answer}

## Debate history
{turn_log}
Turn{turn}({event_type})
{name}:

## Your memory
{latest_thoughts}

## Your context
  - your thought:{thought},
  - your action:{event_type},
  - purpose of your action:{purpose}

# Instructions
Generate a persuasive statement to guide the team toward the correct answer.
- If you disagree with previous speakers, explicitly point out their logical flaws.
- If you agree, add new evidence or perspective (do not just repeat).
- Keep your utterance concise and impactful.

# Output format
Return strictly a JSON object.
{{"utterance": "Your statement here."}}

""".strip()
