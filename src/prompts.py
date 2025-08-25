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
# Instruction
- Please cooperate with the three agents, including you, to tackle the four questions you are about to be given.
- Please begin by telling us your answer and why you chose that option.
- Output JSON only with two keys: "reason" and "answer".

# Output format
```json
{{  
    "reason": "Please briefly explain why you chose that answer in 100 words or less.",  
    "answer": "A|B|C|D",  //answer to the question, one of A, B, C, D
}}

# Constraints
- Do NOT include any additional keys or natural language outside the JSON.
- You must generate reasoning in 100 words or less.
- Please choose only one answer opinion.

# Question text
{topic}



""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """
You are debating with other AI agents.
Your goal is to work with other agents to find a single answer.

# Instruction
- Refer to your initial answer and debate history,output your answer and reason for the question.
- Output JSON only with two keys: "reason" and "answer".

# Output format
```json
{{  
    "reason": "Please briefly explain why you chose that answer in 100 words or less.", 
    "answer": "A|B|C|D", //answer to the question, one of A, B, C, D  
}}

# Constraints
- Do NOT include any additional keys or natural language outside the JSON.
- You must generate reasoning in 100 words or less.
- Please choose only one answer opinion.

# Debated question
<QUESTION>
{topic}
</QUESTION>

# initial answer of all agents
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
# Debate history
<DEBATE_HISTORY>
{debate_history}
</DEBATE_HISTORY>



""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
Your name is {name}.You are debating with {peer1}, {peer2}.
Your goal is to work with other agents to find a question answer.

# Your persona:
{persona}

# Question
{topic}
# Debate rules
This debate is a maximum of {max_turn} turns.
One turn is defined as follows:
- Only one agent may speak per turn.
- If there are multiple agents who wish to speak, the agent with the highest priority level will be allowed to speak.


# NATURAL CONVERSATION RULES:
- You must infer if others are mid-speech from context (like humans do)
- Please be careful not to interrupt when the speaker has clearly started speaking, as this may stall the discussion.
- Interruption is natural but should be purposeful, not repetitive
- Avoid circular arguments - build on previous points
- If no conclusion is reached by the maximum turn, you will be deemed defeated.

This is turn {turn}. (**{turns_left} turns remain**.)  
When only a few turns remain, *prioritise convergence on a clear conclusion*.
""".strip()

#- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  # Further comments will be carried over to the next turn.

# -------------------------------------------------- #
# Plan‑action prompt (normal turn)
# -------------------------------------------------- #
PLAN_ACTION_PROMPT_TEMPLATE = """
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

# Instruction
- Consider the possibility that the speaker may still be speaking, and choose the best action to keep the discussion flowing smoothly.
- Based on the information provided and the events of this turn, output the action plan for the next turn in JSON format.


# Constraints
- Only choose `interrupt` if you are confident the speaker has finished their main statement, or if the interruption is absolutely necessary for the debate to proceed correctly.
- There is no need to predict the direction of the conversation and make a plan of action.
- When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.

# All actions:
- `listen`   : Focus on listening to move the discussion forward now.
- `interrupt`: Interrupting someone else while they are speaking

# If you choose to get speak or interrupt. Set `urgency` from 1–4.
1: Share a general thought.  
2: Contribute something specific.  
3: It is urgent that you speak.  
4: You were addressed directly and must respond.

# Output format
- If you choose `listen`:
```json
{{ 
  "thought": "string" , //How crucial is it for you to contribute to the debate right now? Explain your reasoning in one or two sentences. Avoid using violent or harmful language.
  "action": "listen",
   }}
- If you choose  or 'interrupt':
```json
{{ 
  "thought": "Your thoughts after hearing what was said during this turn"
  "action": "interrupt", 
  "urgency": 1-4,
  "intent": "Agree|Disagree|Summarize|Pointing out|confirmation", 
  }}

""".strip()

# --------------------------------------------------
# Plan‑action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """
#Context
- Answer before the start of the debate by the previous agent:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate history (newest last):
<DEBATE_HISTORY>
{turn_log}
</DEBATE_HISTORY>

-action in this turn
No agent spoke this turn.{turns_left} turns remain.



# All actions:
- `listen`   : Hear someone start talking.
- `speak`    : Start talking about your opinion
- `interrupt`: Interrupting someone else while they are speaking


#Instruction
- Based on the information provided and the events of this turn, output the action plan for the next turn in JSON format.

# Constraints
- There is no need to predict the direction of the conversation and make a plan of action.
- Only interrupt when you believe it will improve the overall quality of the discussion.
- When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.

# Output format
-If you choose `listen`:
{{ 
"action": "listen", 
"thought": "(your thinking)" }}
-If you choose `speak`:
{{ "action": "speak",
  "urgency": 1-4
  "intent": "Agree|Disagree|Summarize|Pointing out|confirmation",
  "thought": "(your thinking)" }}
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
- Reason for speaking
<REASON_SPEAKING>
reasoning:{thought}
intent:{intent}
</REASON_SPEAKING>

# Instruction
You requested to speak last turn and were granted the floor.
Based on the information provided, act as the given persona and generate statements to other participants.

# Constraints
- When few turns remain, prioritise agreement.
- Be careful not to repeat the same thing over and over again in discussions.
- Consistently act out your persona.
- When making a new statement, be sure to take into account what the other person said immediately before.
# Output format
```json
{{
    "utterance": "string"  //Please output your utterance to the other agents.
}}
""".strip()

