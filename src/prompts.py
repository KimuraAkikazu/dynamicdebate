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
You are {name}.

#Your persona
{persona}

# Debate topic
{topic}

You are debating with two AI agents, {peer1} and {peer2}.

The debate is multi‑turn. One turn is defined as follows:
- **Only one agent may speak per turn**, chosen by the largest `urgency`.
- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  Further comments will be carried over to the next turn.

This is turn {turn} of {max_turn}. (**{turns_left} turns remain**.)  
When only a few turns remain, *prioritise convergence on a clear conclusion*.
""".strip()

# -------------------------------------------------- #
# Plan‑action prompt (normal turn)
# -------------------------------------------------- #
PLAN_ACTION_PROMPT_TEMPLATE = """
# Instruction
Return your action plan for the **next turn** in JSON format.

*Actions:*
- `listen`   : Listen to the other agents.
- `speak`    : Speak because no one else is speaking.
- `interrupt`: Interrupt while another agent is still speaking.

*If you choose to get the ball rolling or interrupt. Set `urgency` from 1–4.*
1: Share a general thought.  
2: Contribute something specific.  
3: It is urgent that you speak.  
4: You were addressed directly and must respond.

# Turn‑wise history (newest last)
{turn_log}

# Last utterance in this turn
{last_event}

# Output format
When you choose `listen`:
```json
{{ "action": "listen", "thought": "(Your current thinking)" }}
When you choose speak or interrupt:
{{ "action": "speak|interrupt",
  "urgency": 0‑4,
  "intent": "question/agree/summarise/deny/conclude",
  "thought": "(Your current thinking)" }}
#Notes
-There is no need to predict the direction of the conversation and make a plan of action.
-Only interrupt when you believe it will improve the overall quality of the discussion.
-When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.
-When 0 turns remain, a conclusion must be output.
""".strip()

# --------------------------------------------------
# Plan‑action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """

# Situation
-Debate history (newest last)
{turn_log}
-Last utterance in this turn
No agent spoke this turn. Only {turns_left} turns remain.

#Instruction
Return your action plan for the **next turn** in JSON format.

Actions:
- `listen`   : Listen to the other agents.
- `speak`    : Speak because no one else is speaking.
- `interrupt`: Interrupt while another agent is still speaking.

If you choose to get the ball rolling or interrupt. Set `urgency` from 1–4.  
1: Share a general thought.  
2: Contribute something specific.  
3: It is urgent that you speak.  
4: You were addressed directly and must respond.

# Output format
-When you choose `listen`:
{{ "action": "listen", "thought": "(reason)" }}
-When you choose speak or interrupt:
{{ "action": "speak|interrupt",
  "urgency": 1‑4,
  "intent": "question/agree/summarise/challenge/drive‑to‑conclusion/conclude",
  "thought": "(reason)" }}

# Notes
-There is no need to predict the direction of the conversation and make a plan of action.
-Only interrupt when you believe it will improve the overall quality of the discussion.
-When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.
-When 0 turns remain, a conclusion must be output.

""".strip()

# --------------------------------------------------
# Utterance‑generation prompt
# --------------------------------------------------
GENERATE_UTTERANCE_PROMPT_TEMPLATE = """

#Instruction
-In the previous turn, you requested to speak for the reasons stated in *Reason for speaking (your thoughts)*, 
and you were granted the right to speak.
Refer to the character and *Debate History* and generate a utterance that will lead the discussion to a conclusion.

# Turn‑wise history (newest last)
{turn_log}

# Reason for speaking
Thought : {thought}  
Intent  : {intent}

#Utterance
Write your utterance in concise, clear English.

#Notes
-Omit pleasantries; state your point directly.
-When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.
-Once all members have reached an agreement, please provide your response.
-Base your utterance on the debate history.
""".strip()

