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
-You will now collaborate with the other two members to derive a single agreed-upon answer (A–D) for the multiple-choice question.

# Question text
{topic}

# Instruction
- Please derive the solution to the given problem through step-by-step reasoning.
- Please devide your answer to that question and the reasoning behind it.
- Output JSON only with two keys: "reason" and "answer".

# Constraints
- Do NOT include any additional keys or natural language outside the JSON.
- Please choose only one answer opinion.


# Output format
```json
{{  
    "reason": "string", //Give step-by-step reasoning (<=100 words total).
    "answer": "string",  //answer to the question, one of A, B, C, D
}}
""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """
-You are collaborating with the other two members to derive a single agreed-upon answer (A–D) for a multiple-choice question.

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
- Please choose only one answer opinion.

# Output format
```json
{{  
    "reason": "string", //The reasons and state of mind that ultimately led to selecting that answer
    "answer": "string", //answer to the question, one of A, B, C, D  
}}

""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
- Your name is {name}.You are debating with {peer1} and {peer2}.
- Your goal is to collaborate with other members as {name} and find a consensus-based solution to a single problem.

# Your persona:
{persona}

# Debate rules
- This debate is a maximum of {max_turn} turns.
- You must finish speaking by the {max_turn} turn.
- If all participants do not agree on an answer by the end of the discussion, it will be considered a loss.
- One turn is defined as follows:
  - Only one agent may speak per turn.
  - If there are multiple agents who wish to speak, the agent with the highest urgency will be allowed to speak.
  - Only one sentence may be spoken per turn; any subsequent sentences will be treated as part of the next turn's speech.
 
""".strip()

#- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  # Further comments will be carried over to the next turn.

# -------------------------------------------------- #
# Plan‑action prompt (normal turn)
# -------------------------------------------------- #
PLAN_ACTION_PROMPT_TEMPLATE = """

#Context
-Question Text
<QUESTION>
{topic}
</QUESTION>
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
- This is turn {turn}. (**You have a maximum of {turns_left} remaining turns to speak.**)

# Instruction
- Consider the possibility that the speaker may still be speaking, and choose the best action to keep the discussion flowing smoothly.
- Based on the information provided and the events of this turn, output your action plan for the next turn in JSON format.


# Constraints
- You must infer if others are mid-speech from context.
- Please begin speaking while another participant is talking only when it is absolutely necessary to guide the discussion toward the correct answer.
- There is no need to predict the direction of the conversation and make a plan of action.
- Within the remaining turns, you are expected to collaborate with the other agents and work toward one agreed-upon final answer. 


# All actions:
- `listen`   : Focus on listening to move the discussion forward now.
- `speak`    : The current speaker has finished speaking, so I will state my point
- `interrupt`: The current speaker may still have more to say, but I have something I wish to assert.

# Set the urgency for speaking on your next turn on a 0–4 scale:
- 0: Listen to others and deepen your thinking.
- 1: Share a general thought.
- 2: State a specific opinion.
- 3: You have something you really need to say.
- 4: Speak up now, even if it means interrupting.

# Output format
```json
{{ 
  "thought": "strting" //Your thoughts and feelings toward other agents after hearing what was said during this turn.
  "action": "listen|speak|interrupt",  //Based on your thought, choose your action for the next turn.
  "urgency": 0-4, //Please output the numerical value representing the urgency level of your next turn's statement.
  "intent": "agree|disagree|summarize|confirmation|proposal",  /Please select the intent of your action plan.
  }}

""".strip()

# --------------------------------------------------
# Plan‑action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """
#Context
-Question text
<QUESTION>
{topic}
</QUESTION>
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
-This is turn {turn}. (**{turns_left} turns remain**.)
-Within the remaining turns, you are expected to collaborate with the other agents and work toward one agreed-upon final answer. 

# All actions:
- `listen`   : Hear someone start talking.
- `speak`    : Nobody's talking, so I'll start talking.

# Set the urgency for speaking on your next turn on a 0–4 scale:
- 0: Listen to others and deepen your thinking.
- 1: Share a general thought.
- 2: State a specific opinion.
- 3: You have something you really need to say.
- 4: Speak up now, even if it means interrupting.


# Output format
{{
  "thought": "string", ////Your thoughts and feelings toward other agents in this turn.
  "action": "listen|speak", //Based on your thought, choose your action for the next turn.
  "urgency": 0-4,
  "intent": "agree|disagree|summarize|confirmation|proposal",  //Please select the intent of your action plan.
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
-This is turn {turn}. (**{turns_left} turns remain**.)
-Within the remaining turns, you are expected to collaborate with the other agents and work toward one agreed-upon final answer. 

# Output format
```json
{{
    "utterance": "string"  //Your public statement in the debate. Be concise and persuasive. Respond directly to what the other players have said.  Avoid simply repeating what others have said or reguritating the instructions above.
}}
""".strip()

