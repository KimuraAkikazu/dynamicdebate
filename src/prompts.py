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
- You are {name}.{persona}
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
- You are {name}.{persona}
- You cooperated with two other members and engaged in a discussion to derive a single answer (A–D) to a multiple-choice question.
- Your goal is to collectively decide on the answer to the question.

# Context
# Question
Question: {topic}

- initial answer of all members
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
- Only one member can speak per turn.
- Exactly one chunk is revealed to all members each turn. Therefore, since the speaker can prepare multiple chunks during their preparation phase, statements may have continuations.

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
- `listen`   : Use when listening to the speaker's argument and waiting for them to finish speaking.
- `interrupt`: Use when interrupting the current speaker to begin speaking.

Instructions:
- Your goal is to decide on the correct answer within the maximum number of turns.
- To reach consensus, consider whether to push your position or align with others, and choose accordingly.
- While considering the possibility that someone may be mid-sentence, decide whether to interrupt and respond immediately to this turn's statement or listen to its completion.


Constraints:
- Once all members agree on the same answer, the discussion ends with that answer.
- When the number of remaining turns grows short, prioritize consensus over pushing your own agenda.
- There is only one correct answer choice for the question.

Output format:
{{ 
  "action": "listen|interrupt",  // Decide what action to take next turn. Interrupting may derail the discussion, so when you have sufficient information for a comprehensive response or when the current statement contains errors or misunderstandings.
  "thought": "string",  // State your thoughts based on the content of the discussion so far and what the speaker said in this turn.
  "urgency": 0-9, // Based on your “thought,” how urgent is it for you to speak during the next turn? Please output a number indicating the urgency.
  "intent": "agree|disagree|summarize|confirmation|proposal|conclusion|think",  // Select the purpose of the chosen action.
  "consensus": {{
    "agreed": true|false, //If it appears that others support the same choice and you also support it, select true. Otherwise, select false.
    "answer": "A|B|C|D|none"     // If “agreed” is “true”, set agreed answer.If “agreed” is “false”, set “none”.
  }}
  }}
""".strip()



# --------------------------------------------------
# Plan‑action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """
# Debate rules
- The debate has at most {max_turn} turns. You must finish speaking by turn {max_turn}.
- Only one member can speak per turn.
- Exactly **one sentence** is revealed to all members each turn.
- A speaker may **compose multiple sentences** when preparing their utterance, but:
  - Only the **first** sentence will be published on the next turn.
  - The **remaining sentences are queued** and **do not reserve future turns**; other members may be selected to speak before your queued sentences are revealed (i.e., you can be **interrupted**).
- Turn timeline:
  - **Turn 0**: planning only.
  - **Turn t ≥ 1**:
    1) Publish phase: if the current speaker has a queue, publish one sentence; otherwise it's a silence event.
    2) Planning phase: all non-speaking members output an action plan.
    3) Selection phase: among members choosing speak/interrupt, the highest urgency is selected. If everyone chooses listen, the next event is silence.
    4) utterance phase: the selected next speaker generates their utterance for the turn.
- Interrupt semantics: If a speaker holds an unpublished sentence and another member selects speak/interrupt, the speaking right transfers.
- Reach consensus in as few turns as possible; keep sentences concise and on-topic.

# Question
Question: {topic}

# Context
- The initial answers provided by all members before the discussion began:
<INITIAL_ANSWERS>
{initial_answer}
</INITIAL_ANSWERS>
- Debate so far:
<DEBATE_SO_FAR>
{turn_log}
</DEBATE_SO_FAR>
- Your thought from the previous turn:
<Your_Thought>
{latest_thoughts}
</Your_Thought>
- This is turn {turn}. 
-Events in this turn
<EVENTS_THIS_TURN>
{last_event}
</EVENTS_THIS_TURN>
- You have {turns_left} speaking opportunities remaining.
- Decide on your final answer within the remaining {turns_left} turns.

# All actions:
- `listen`   : You wait for someone to start talking and then listen.
- `speak`    : You begin speaking to move the discussion forward.

# urgency scale:
 0: For now, I focus on listening.
 1: I Provide topics to advance the discussion.
 2: I want to state a specific opinion.
 3: There's something I absolutely need to talk about right now.
 4: Someone has addressed me directly and I must respond.

# Instruction
- Your goal is to decide on the correct answer within the maximum number of turns.
- To reach consensus, consider whether to push your position or align with others, and choose accordingly.
- Based on the debate so far and this turn's events, formulate your action plan for the next turn consistent with your personality.
- Consider the current speaker’s utterance and the possibility they may still be continuing.
- Consensus check: From <INITIAL_ANSWERS> and <DEBATE_SO_FAR>, infer each member's current choice.
  If others appear to support the same choice and you also support it, set:
    "consensus": {{ "agreed": true,  "answer": "<A|B|C|D>" }}.
  Otherwise set:
    "consensus": {{ "agreed": false, "answer": "none" }}.

# Constraints
- Please bear in mind that prolonged silence hinders progress in discussions.
- Once all members agree on the same answer, the discussion ends with that answer.
- Be careful not to stray into discussions that are not necessary for answering the question.
- When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.


# Output format
{{
  "thought": "string",  // Based on the debate so far and the events of this turn, briefly explain your current inner thoughts.
  "action": "listen|speak",  // Based on your "thought", please select the action you wish to take on your next turn.
  "urgency": 0-4, //Based on the debate so far and your current thought, how urgent is it for you to speak in the next turn? Please output a numerical value indicating the urgency.
  "intent": "agree|disagree|summarize|confirmation|proposal|question|conclusion|think",  // Select the purpose of the chosen action.
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

