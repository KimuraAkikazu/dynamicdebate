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
- The following is a multiple-choice question. Please choose the single most appropriate option and provide its label (e.g., A, B, C).

# Question
{topic}


# Output format
- Output JSON only with two keys: "answer" and "reasoning".
- Do NOT include any additional keys or natural language outside the JSON.
```json
{{  
    "answer": "string",  //Please output the answer choices for the question.
    "reasoning": "string"  //Please share your thoughts on why you gave that answer.
}}

""".strip()

# -------------------------------------------------- #
# Final answer prompt (after the debate)
# -------------------------------------------------- #
FINAL_ANSWER_PROMPT_TEMPLATE = """

# Debated question
{topic}

# Your initial answer (pre-debate)
{initial_answer}

# Full debate history (newest last, truncated)
{debate_history}

# Instruction
-According to your initial answer and debate history,output your *final* answer in JSON with exactly two keys.
- Output JSON only with two keys: "answer" (one of A–D etc.) and "reasoning" (1-2 sentences).
- Do NOT include any additional keys or natural language outside the JSON.

# Output format
```json
{{  
    "answer": "string",  //Please provide your final answer based on your initial answer and discussion, such as A, B, C, or D.
    "reasoning": "string"  //Please explain why you chose your final answer.
}}

""".strip()

# -------------------------------------------------- #
# System prompt
# -------------------------------------------------- #
SYSTEM_PROMPT = """
You are debating with other AI agents.
Your goal is to work with other agents to find a single answer.

# Your Profile
- Your name is {name}.
- Debating with {peer1}, {peer2}.
- Your persona:
{persona}

# Question
{topic}
# Debate rules
# The debate is multi‑turn. One turn is defined as follows:
- Only one agent may speak per turn.
- If there are multiple agents who wish to speak, the agent with the highest priority level will be allowed to speak.


# NATURAL CONVERSATION RULES:
- You must infer if others are mid-speech from context (like humans do)
- Look for incomplete sentences, "and...", "but...", logical flow breaks
- Interruption is natural but should be purposeful, not repetitive
- Avoid circular arguments - build on previous points
-If no conclusion is reached by the maximum turn, you will be deemed defeated.

This is turn {turn} of {max_turn}. (**{turns_left} turns remain**.)  
When only a few turns remain, *prioritise convergence on a clear conclusion*.
""".strip()

#- In a single turn you must output **exactly one chunk**, ending with a comma “,” or period “.”.  
  # Further comments will be carried over to the next turn.

# -------------------------------------------------- #
# Plan‑action prompt (normal turn)
# -------------------------------------------------- #
PLAN_ACTION_PROMPT_TEMPLATE = """
# initial answer of all agents
{initial_answer}

# Turn‑wise history (newest last)
{turn_log}

# Utterance in this turn
{last_event}


*Actions:*
- `listen`   : I observe and listen for now to advance the discussion.
- `speak`    : I speak because no one else is speaking now.
- `interrupt`: I interrupt while another agent is still speaking.

*If you choose to get speak or interrupt. Set `urgency` from 1–4.*
1: Share a general thought.  
2: Contribute something specific.  
3: It is urgent that you speak.  
4: You were addressed directly and must respond.

# Instruction
-Listen to the “Utterance in this turn” and output your action plan for the next turn in JSON format.
-Consider that the speaker may still be talking, and choose the best course of action to move the discussion forward.


# Output format
-If you choose `listen`:
```json
{{ "thought": "string" , //How crucial is it for you to contribute to the debate right now? Explain your reasoning in one or two sentences. Avoid using violent or harmful language.
   "action": "string" //If you choose to listen, select “listen” as the action.
   }}
-Else if you choose 'speak' or 'interrupt':
{{ "thought": "string" , //How crucial is it for you to contribute to the debate right now? Explain your reasoning in one or two sentences. Avoid using violent or harmful language.
  "action": "string", //If no other agent is speaking, select “speak”; if you want to interrupt someone who is speaking, select “interrupt.”
  "urgency": "string", //Please output the urgency of the statement based on your reasons. Response is a single number from:   "1" | "2" | "3" | "4"
  "intent": "string", //Please output the type of statement based on your reason, such as question, agree, summarise, deny, conclude.
  }}
#Notes
-There is no need to predict the direction of the conversation and make a plan of action.
-Only interrupt when you believe it will improve the overall quality of the discussion.
-Be careful not to get caught up in endless debate.
-When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.
""".strip()

# --------------------------------------------------
# Plan‑action prompt (silence turn)
# --------------------------------------------------
SILENCE_PLAN_PROMPT_TEMPLATE = """
# Your initial answer
{initial_answer}

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
  "intent": "question/agree/summarise/challenge/conclude",
  "thought": "(reason)" }}

# Notes
-There is no need to predict the direction of the conversation and make a plan of action.
-Only interrupt when you believe it will improve the overall quality of the discussion.
-When few turns remain, prioritise convergence and a clear conclusion or provisional agreement.
""".strip()

# --------------------------------------------------
# Utterance‑generation prompt
# --------------------------------------------------
GENERATE_UTTERANCE_PROMPT_TEMPLATE = """
# Your initial answer
{initial_answer}

#Instruction
-In the previous turn, you requested to speak for the reasons stated in *Reason for speaking*, 
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
-When 0 turns remain,you must output the answer.
""".strip()

