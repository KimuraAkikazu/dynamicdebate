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
# Question text
{topic}

# Instruction
- Please cooperate with the three agents, including you, to tackle the four questions you are about to be given.
- Please derive the solution step by step for the given problem.
- Please tell us your answer to the question and the reason why you chose that answer.
- Output JSON only with two keys: "reason" and "answer".

# Constraints
- Do NOT include any additional keys or natural language outside the JSON.
- You must generate reasoning in 100 words or less.
- Please choose only one answer opinion.


# Output format
```json
{{  
    "reason": "string", //Please explain why you chose that answer in 100 words or less.  
    "answer": "A|B|C|D",  //answer to the question, one of A, B, C, D
}}
""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """
# Context
-Question text
<QUESTION>
{topic}
</QUESTION>
-initial answer of all agents
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
-Debate history
<DEBATE_HISTORY>
{debate_history}
</DEBATE_HISTORY>

# Instruction
- Refer to your initial answer and debate history,output your answer and reason for the question.
- Output JSON only with two keys: "reason" and "answer".

# Constraints
- Do NOT include any additional keys or natural language outside the JSON.
- You must generate reason in 100 words or less.
- Please choose only one answer opinion.

# Output format
```json
{{  
    "reason": "string", //Reasons for ultimately choosing that answer in 100 words or less. 
    "answer": "A|B|C|D", //answer to the question, one of A, B, C, D  
}}

""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
Your name is {name}.You are debating with {peer1} and {peer2}.
Your goal is to work with other agents as {name} to reach a consensus and find the answer to a single question.

# Your persona:
{persona}

# Question
{topic}
# Debate rules
This debate is a maximum of {max_turn} turns.
You must finish speaking by the {max_turn} turn.
One turn is defined as follows:
- Only one agent may speak per turn.
- If there are multiple agents who wish to speak, the agent with the highest urgency will be allowed to speak.


This is turn {turn}. (**{turns_left} turns remain**.)
Within the remaining turns, you are expected to collaborate with the other agents and work toward one agreed-upon final answer.  
""".strip()

#- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  # Further comments will be carried over to the next turn.

# -------------------------------------------------- #
# Plan‑action prompt (normal turn)
# -------------------------------------------------- #
PLAN_ACTION_PROMPT_TEMPLATE = """
# Instruction
- Consider the possibility that the speaker may still be speaking, and choose the best action to keep the discussion flowing smoothly.
- Based on the information provided and the events of this turn, output the action plan for the next turn in JSON format.


# Constraints
- Only interrupt the speaker when absolutely necessary to steer the discussion toward the correct answer.
- There is no need to predict the direction of the conversation and make a plan of action.

#Context
- Answer before the start of the debate by the previous agent:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate history (newest last):
<DEBATE_HISTORY>
{turn_log}
</DEBATE_HISTORY>

# Events in this turn
<EVENTS_THIS_TURN>
{last_event}
</EVENTS_THIS_TURN>


# NATURAL CONVERSATION RULES:
- You must infer if others are mid-speech from context (like humans do)
- Interruption is natural but should be purposeful, not repetitive
- Avoid circular arguments - build on previous points
- If all participants do not agree on an answer by the end of the discussion, it will be considered a loss.

# All actions:
- `listen`   : Focus on listening to move the discussion forward now.
- `speak`    : The current speaker has finished speaking, so I will state my point
- `interrupt`: The current speaker may still have more to say, but I have something I wish to assert.

# Please set the urgency level of your next turn's statement on a scale of 0 to 4.
0: Listen to what others have to say and deepen my thinking.
1: Share a general thought.  
2: Contribute something specific.  
3: It is urgent that you speak.  
4: You should assert yourself even if it means interrupting.

# Output format
```json
{{ 
  "thought": "Your thoughts and feelings toward other agents after hearing what was said during this turn."
  "action": "listen|speak|interrupt", 
  "urgency": 0-4,
  "intent": "agree|disagree|summarize|confirmation|proposal", 
  }}

""".strip()

# --------------------------------------------------
# Plan‑action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """
#Debate context
- Answer before the start of the debate by the previous agent:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate history (newest last):
<DEBATE_HISTORY>
{turn_log}
</DEBATE_HISTORY>
-Events in this turn
No agent spoke this turn.
Please reach a conclusion within the remaining {turns_left} turns.

#Instruction
- Based on the information provided and the events of this turn, output the action plan for the next turn in JSON format.

# Constraints
- There is no need to predict the direction of the conversation and make a plan of action.
- Only interrupt when you believe it will improve the overall quality of the discussion.
- When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.

# All actions:
- `listen`   : Hear someone start talking.
- `speak`    : Nobody's talking, so I'll start talking.

# Please output the urgency level of your next turn on a scale of 0-4.
0: Listen to what others have to say and deepen my thinking.
1: Share a general thought.  
2: Contribute something specific.  
3: It is urgent that you speak.  
4: You have been directly addressed and need to respond immediately.


# Output format
{{
  "action": "listen|speak",
  "urgency": 0-4,
  "intent": "agree|disagree|summarize|confirmation|proposal",
  "thought": "Your thoughts and feelings toward other agents right now",
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
#Context
- Answer before the start of the debate by the previous agent:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate history (newest last):
<DEBATE_HISTORY>
{turn_log}
</DEBATE_HISTORY>


# Instruction
-You are speaking next in the debate.
-Your thoughts on speaking next: "thought:{thought}, intent:{intent}"
-Your goal is to collectively arrive at a single answer to the given question by the final turn.
-Based on the information provided, act as the given persona and generate statements to other participants.

# Constraints
- When few turns remain, prioritise agreement.
- Be careful not to repeat the same thing over and over again in discussions.
- Consistently act out your persona.
- When making a new statement, be sure to take into account what the other person said immediately before.

# Output format
```json
{{
    "utterance": "string"  //Your public statement in the debate. Be concise and persuasive. Respond directly to what the other players have said.  Avoid simply repeating what others have said or reguritating the instructions above.
}}
""".strip()

