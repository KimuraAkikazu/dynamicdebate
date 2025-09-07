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
- You will now collaborate with the other two members to derive a single answer (A–D) for the multiple-choice question through discussion.

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
- You cooperated with two other members and engaged in a discussion to derive a single answer (A–D) to a multiple-choice question.
- Your goal is to reach a consensus among the members and produce one answer as a team.

# Context
- Question text
<QUESTION>
{topic}
</QUESTION>
- initial answer of all agents
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

# Constraints
- Do NOT include any additional keys or natural language outside the JSON.
- Please choose only one answer opinion.

# Output format
```json
{{  
    "reason": "string", //The reasons and mindset that led to the final selection of that response after concluding the discussion.
    "answer": "string", //answer to the question, one of A, B, C, D  
}}

""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
- Your name is {name}.You are discussing with {peer1} and {peer2} which of the given options is the correct answer to the problem.
- Your goal is to collaborate with other members as {name}, build consensus among members, and arrive at a single answer choice as a team.

# Your persona:
{persona}

# Debate rules
- This debate is a maximum of {max_turn} turns.
- You must finish speaking by the {max_turn} turn.
- One turn is defined as follows:
  - Only one member can speak per turn.
  - If there are multiple agents who wish to speak, the agent with the highest urgency will be allowed to speak.
  - Only one sentence is accepted as speech per turn. If you generate multiple sentences in a single turn, only the first will be processed as speech, and the remaining sentences will be treated as if they were spoken in later turns.
- If the discussion ends without all members set agreed to true, it is considered a defeat.
- Regarding responses, it is desirable for everyone to reach agreement in as few turns as possible.
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
- This is turn {turn}.
- You have {turns_left} chance(s) to speak left.
- Reach a conclusion within the remaining {turns_left} turns.
- Events in this turn
<EVENTS_THIS_TURN>
{last_event}
</EVENTS_THIS_TURN>

# All actions:
- `listen`   : I focus on listening to move the discussion forward.
- `speak`    : The current speaker has finished speaking, so I will state my point
- `interrupt`: The current speaker may still have more to say, but I have something I wish to assert.

# urgency scale:
 0: Listen to others and deepen your thinking.
 1: Share a general thought.
 2: State a specific opinion.
 3: I have something I want to assert right away, if possible.
 4: There's something I absolutely need to talk about right now.

# Instruction
- Based on the previous discussion and the events in this turn, output your action plan for the next turn in JSON format, aiming to find the correct option within the remaining turns as a team.
- When formulating your action plan, consider the current speaker’s utterance and take into account the possibility that the speaker may still be continuing their speech.
- Set the urgency for speaking on your next turn on a 0–4 scale.
- Consensus check: From <INITIAL_ANSWERS> and <DEBATE_HISTORY>, infer each participant’s CURRENT preferred answer (A–D).
    If others match (same choice) AND you also support that choice, set:
      "consensus": {{ "agreed": true, "answer": "<A|B|C|D>" }}.
    Otherwise set:
      "consensus": {{ "agreed": false , "answer": "none" }}.

# Constraints
- You must infer from context if others are in the middle of an utterance.
- Only start speaking while another member is speaking if it is judged necessary to guide the discussion toward the correct answer.
- There is no need to predict the direction of the conversation and make a plan of action.
- When the number of remaining turns grows short, prioritize consensus over pushing your own agenda

# Output format
```json
{{ 
  "thought": "strting",  //Please briefly state your current thoughts and feelings about the other agents.
  "action": "listen|speak|interrupt",  //Based on your "thought", please select the action you wish to take on your next turn.
  "urgency": 0-4, //Based on your “thought,” output a number representing the urgency of your statement in the next turn.
  "intent": "agree|disagree|summarize|confirmation|proposal|question|conclusion|agreement",  /Please select the intent of your action plan.
  "consensus": {{
    "agreed": true|false,   
    "answer": "A|B|C|D|none"     // If “agreed” is “true”, set agreed answer,if “agreed” is “false”, set “none”.
  }}
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
- This is turn {turn}. 
- You have {turns_left} chance(s) to speak left.
- Reach a conclusion within the remaining {turns_left} turns.
-Events in this turn
<EVENTS_THIS_TURN>
{last_event}
</EVENTS_THIS_TURN>

# All actions:
- `listen`   : I'll wait for someone to start talking and then listen.
- `speak`    : I'll start talking and move the discussion forward.

# urgency scale:
 0: Listen to others and deepen your thinking.
 1: Provide a topic.
 2: State a specific opinion.
 3: I have something I want to assert right away, if possible.
 4: There's something I absolutely need to talk about right now.


#Instruction
- Based on the previous discussion and the events in this turn, output your action plan for the next turn in JSON format, aiming to find the correct option within the remaining turns as a team.
- Set the urgency for speaking on your next turn on a 0–4 scale.
- Consensus check: From <INITIAL_ANSWERS> and <DEBATE_HISTORY>, infer each participant’s CURRENT preferred answer (A–D).
    If others match (same choice) AND you also support that choice, set:
      "consensus": {{ "agreed": true, "answer": "<A|B|C|D>" }}.
    Otherwise set:
      "consensus": {{ "agreed": false , "answer": "none" }}.

# Constraints
- Be careful not to stray into discussions that are not necessary for answering the question.
- There is no need to predict the direction of the conversation and make a plan of action.
- When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.
- When the number of remaining turns grows short, prioritize consensus over pushing your own agenda.


# Output format
{{
  "thought": "string",  //Your thoughts and feelings toward other agents in this turn.
  "action": "listen|speak",  //Based on your "thought", please select the action you wish to take on your next turn.
  "urgency": 0-4, //Based on your “thought,” output a number representing the urgency of your statement in the next turn.
  "intent": "agree|disagree|summarize|confirmation|proposal|question|conclusion|agreement",  //Please tell us the reason behind your chosen action.
  "consensus": {{
    "agreed": true|false,  
    "answer": "A|B|C|D|none"  //If “agreed” is “true”, set agreed answer,if “agreed” is “false”, set “none”.
  }}
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
- This is turn {turn}.
- You have {turns_left} chance(s) to speak left.
- Reach consensus within the remaining {turns_left} turns.

# Instruction
- You are speaking next turn in the debate.
- Your thoughts on speaking next: "thought:{thought}, intent:{intent}"
- As the assigned persona, generate your public statement for the next turn, continuing from the discussion so far.

# Constraints
- Be careful not to stray into discussions that are not necessary for answering the question.
- Your goal is to collectively decide on the answer choices for the given question within the maximum turn limit.
- Be careful not to repeat the same thing over and over again in discussions.
- Within the remaining turns, you must collaborate with other agents to narrow down to a single answer.
- When turns become scarce, prioritize finding a team answer over pushing your own opinion.

# Output format
```json
{{
    "utterance": "string"  // Your public speech in the debate. Be concise and persuasive. Respond directly to what the other players have said.  Avoid simply repeating what others have said or reguritating the instructions above.
}}
""".strip()

