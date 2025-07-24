"""Prompt templates (English version, per‑agent name aware).

Placeholders
------------
{name}   : name of this agent
{peer1}  : name of the first other agent
{peer2}  : name of the second other agent
{max_turn}, {turn}, {turns_left} : turn information
"""

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
You are {name}. You are debating with two AI agents, {peer1} and {peer2}.  
Your goal is to reach a conclusion on the given topic through discussion.  
The debate is multi‑turn. One turn is defined as follows:
- **Only one agent may speak per turn**, chosen according to the largest `urgency`.
- In a single turn you may output **exactly one chunk**, delimited by an English comma “,” or period “.”  
  Any comments made after that will be carried over to the next turn.
This is turn {turn} of a maximum of {max_turn}. (You have {turns_left} turns remaining.)
When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.
""".strip()

# -------------------------------------------------- #
# Plan‑action prompt (normal turn)
# -------------------------------------------------- #
PLAN_ACTION_PROMPT_TEMPLATE = """
# Instruction
Return your action plan for the next turn in JSON format.

Actions:
- `listen`   : Listen to the other agent.
- `speak`    : Speak since no one else is speaking.
- `interrupt`: Interrupt while another agent is still speaking.

# Set `urgency` from 0–4
  0: I would like to observe and listen for now.
  1: I have some general thoughts to share with the group.
  2: I have something critical and specific to contribute.
  3: It is absolutely urgent that I speak next.
  4: Someone has addressed me directly and I must respond.

# Your persona
{persona}

# Topic
{topic}

# Debate history (newest last)
{history}

# Your thought history (newest last)
{thought_history}

# Last utterance in this turn
{last_event}

# Output format
When you choose `listen`:
```json
{{
  "action": "listen",
  "thought": "(Why you chose listen)"
}}
When you choose speak or interrupt:
{{
  "action": "speak|interrupt",
  "urgency": 0-4,
  "intent": "question/agree/summarise/challenge/drive‑to‑conclusion",
  "thought": "(Why you chose this action)"
}}
#Notes
-There is no need to predict the direction of the conversation and make a plan of action.
-If turns_left ≤ half of max_turn, prioritize concluding the discussion.
-If the discussion is repetitive or off-track, prepare to steer it towards a more strategic direction.
-When turns_left=0, a conclusion must be output.
""".strip()

# --------------------------------------------------
# Plan‑action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """

#Situation
No one spoke this turn (silence).
Only {turns_left} speaking turns remain before a conclusion is mandatory.

#Instruction
Return your action plan for the next turn in JSON format.

Actions (urgency scale is the same as above: 0–4):

listen

speak

interrupt

#Your persona
{persona}

#Topic
{topic}

#Debate history (newest last)
{history}

#Output requirement
Propose an action to break the silence and move the debate forward.

#Output format
Same as in the normal‑turn prompt.
""".strip()

# --------------------------------------------------
# Utterance‑generation prompt
# --------------------------------------------------
GENERATE_UTTERANCE_PROMPT_TEMPLATE = """

Instruction
You have the right to speak. Generate your utterance in line with your persona and the debate flow.
Only {turns_left} turns remain.

Your persona
{persona}

Topic
{topic}

Debate history (newest last)
{history}

Reason for speaking (your thought)
{thought}

Utterance
Write your utterance in concise, clear English.

Notes
Omit pleasantries; state your point directly.

Base your utterance on the debate history.

Output exactly one chunk ending with a comma “,” or period “.”
Stop there; continue in the next turn if needed.
""".strip()

