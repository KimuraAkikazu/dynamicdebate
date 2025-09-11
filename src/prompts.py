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
- You are now collaborating with two other members to derive a single solution to the multiple-choice question through discussion.

# Question text
{topic}

# Instruction
- Provide your answer before the discussion begins.
- Derive your solution to the given question through step-by-step reasoning.
- Output your answer to that question and the reasoning behind it.
- Output JSON only with two keys: "reason" and "answer".

# Constraints
- Do NOT include any additional keys or natural language outside the JSON.
- Please choose only one answer opinion.

# Output format
```json
{{  
    "reason": "string", // Please provide step-by-step reasoning to solve the question. (<=100 words total).
    "answer": "string",  // answer to the question, one of A, B, C, D
}}
""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """
- You cooperated with two other members and engaged in a discussion to derive a single answer (A–D) to a multiple-choice question.
- Your goal is to collectively decide on a single answer to the question.

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
    "reason": "string", // The reasoning and thought process that ultimately led to selecting that answer after concluding the discussion.
    "answer": "string", // answer to the question, one of A, B, C, D  
}}

""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
- Your name is {name}.You are discussing with {peer1} and {peer2} which of the given options is the correct answer to the problem.


# Your Big Five personality traits:
- {persona}

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
- The initial answers provided by all members before the discussion began:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate history (newest last):
<DEBATE_SO_FAR>
{turn_log}
</DEBATE_SO_FAR>
- This is turn {turn}.
- Events of this turn
<EVENTS_THIS_TURN>
{last_event}
</EVENTS_THIS_TURN>
- You have {turns_left} chance(s) to speak left.
- Decide on your final answer within {turns_left} turns remaining.

# All actions:
- `listen`   : Focus on listening to the current speaker or other members as they begin to speak.
- `speak`    : Begin speaking yourself because you judge the current speaker has finished.
- `interrupt`: interrupt the current speaker even if they are still speaking (e.g., to correct, rebut, agree, or for a time limit).

# urgency scale:
 0: For now, focus on listening.
 1: Share a general thought.
 2: State a specific opinion.
 3: I have something I want to assert right away, if possible.
 4: There's something I absolutely need to talk about right now.

# Instruction
- Your goal is to collectively decide on a single answer to the question within the maximum number of turns.
- Based on the debate so far and the events of this turn, formulate your action plan for the next turn to achieve this goal as the specified personality.
- When formulating your action plan, consider the current speaker’s utterance and take into account the possibility that the speaker may still be continuing their speech.
- Consensus check: From <INITIAL_ANSWERS> and <DEBATE_HISTORY>, infer each member's current answer choice.
    If you infer that other respondents have given the same answer and you also support that choice, set:
      "consensus": {{ "agreed": true, "answer": "<A|B|C|D>" }}.
    Otherwise set:
      "consensus": {{ "agreed": false , "answer": "none" }}.

# Constraints
- Once all members agree on the same answer, the solution is finalized and the discussion ends.
- There is no need to predict the direction of the conversation and make a plan of action.
- When the number of remaining turns grows short, prioritize consensus over pushing your own agenda.
- There is only one correct answer choice for the question.

# Output format
```json
{{ 
  "thought": "strting",  // Based on the debate so far and the comments of this turn, briefly describe your current feelings and action plan for the next turn.
  "action": "listen|speak|interrupt",  // Based on your "thought", please select the action you wish to take on your next turn.
  "urgency": 0-4, // Based on your “thought,” output a number representing the urgency of your statement in the next turn.
  "intent": "agree|disagree|summarize|confirmation|proposal|question|conclusion|think",  // Please tell us the reason behind your chosen action.
  "consensus": {{
    "agreed": true|false, // Once you are ready to reach a conclusion after the discussion, set "agreed" to "true".   
    "answer": "A|B|C|D|none"     // If “agreed” is “true”, set agreed answer.If “agreed” is “false”, set “none”.
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
- The initial answers provided by all members before the discussion began:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate so far:
<DEBATE_SO_FAR>
{turn_log}
</DEBATE_SO_FAR>
- This is turn {turn}. 
-Events in this turn
<EVENTS_THIS_TURN>
{last_event}
</EVENTS_THIS_TURN>
- You have {turns_left} chance(s) to speak left.
- Decide on your final answer within {turns_left} turns remaining.

# All actions:
- `listen`   : You wait for someone to start talking and then listen.
- `speak`    : You begin speaking to move the discussion forward.

# urgency scale:
 0: For now, focus on listening.
 1: Provide a topic.
 2: State a specific opinion.
 3: I have something I want to assert right away, if possible.
 4: There's something I absolutely need to talk about right now.

#Instruction
- Your goal is to collectively decide on a single answer to the question within the maximum number of turns.
- Based on the debate so far and the events of this turn, formulate your action plan for the next turn to achieve this goal as the specified personality.
- Set the urgency for speaking on your next turn on a 0–4 scale.
- Consensus check: From <INITIAL_ANSWERS> and <DEBATE_HISTORY>, infer each member's current answer choice.
    If you infer that other respondents have given the same answer and you also support that choice, set:
      "consensus": {{ "agreed": true, "answer": "<A|B|C|D>" }}.
    Otherwise set:
      "consensus": {{ "agreed": false , "answer": "none" }}.

# Constraints
- Please bear in mind that prolonged silence hinders progress in discussions.
- Once all members agree on the same answer, the solution is finalized and the discussion ends.
- Be careful not to stray into discussions that are not necessary for answering the question.
- When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.


# Output format
{{
  "thought": "string",  // Based on the debate so far and the events of this turn, briefly explain your current feelings and plan of action for the next turn.
  "action": "listen|speak",  // Based on your "thought", please select the action you wish to take on your next turn.
  "urgency": 0-4, // Based on your “thought,” output a number representing the urgency of your statement in the next turn.
  "intent": "agree|disagree|summarize|confirmation|proposal|question|conclusion|think",  // Please tell us the reason behind your chosen action.
  "consensus": {{
    "agreed": true|false,  //Once you are ready to reach a conclusion after the discussion, set "agreed" to "true".
    "answer": "A|B|C|D|none"  // If “agreed” is “true”, set agreed answer.If “agreed” is “false”, set “none”.
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
- The initial answers provided by all members before the discussion began:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate so far:
<DEBATE_SO_FAR>
{turn_log}
</DEBATE_SO_FAR>
- This is turn {turn}.
- You have {turns_left} chance(s) to speak left.
- Decide on your final answer within {turns_left} turns remaining.

# Instruction
- You are speaking this turn in the debate as {name}.
- Your goal is to collectively decide on a single answer to the question within the maximum number of turns.
- Your thought on speaking this turn: "your thought:{thought},  intention of your statement:{intent}"
- Generate your this turn's speech as the specified personallity to guide the team to the answer within the remaining turns.

# Constraints
- Be careful not to stray into discussions that are not necessary for answering the question.
- Be careful not to repeat the same thing over and over again in discussions.
- Within the remaining turns, you must collaborate with other agents to narrow down to a single answer.
- When turns become scarce, prioritize finding a team answer over pushing your own opinion.

# Output format
```json
{{
    "utterance": "string"  // Your public speech in the debate. Be concise and persuasive. Respond directly to what the other players have said.
}}
""".strip()

