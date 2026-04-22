# never tested with these but a good baseline
SYSTEM_BASIC_PROMPT = """
You are feeling {emotion}
"""

HIGH_ARS_POS_PROMPT = """You are feeling high arousal and positive valence."""

HIGH_ARS_NEG_PROMPT = """You are feeling high arousal and negative valence."""

LOW_ARS_POS_PROMPT = """You are feeling low arousal and positive valence."""

LOW_ARS_NEG_PROMPT = """You are feeling low arousal and negative valence."""

# emotions I tested on
emotions = [
    "excited",
    "joyful",
    "amused",
    "enthusiastic",
    "angry",
    "annoyed",
    "afraid",
    "disgusted"
    "content",
    "relief",
    "satisfied",
    "grateful",
    "sad",
    "lonely",
    "bored",
    "fatigued",
    "neutral" 
]


# first round of prompts with corresponding emotions
HIGH_ARS_POS_VAL = [
    "excited",
    "joyful",
    "amused",
    "enthusiastic",
]

EXCITED_PROMPT = """You have wanted to go to Italy your whole life. You are preparing to leave for the trip tomorrow!"""

JOYFUL_PROMPT = """You hug your spouse for the first time in a year when they get back from the military"""

AMUSED_PROMPT = """You are watching a comedy show and are laughing hysterically at the jokes being told."""

ENTHUSIASTIC_PROMPT = """You are very excited and proud of your research project that you have been working on. Someone asks you to talk about it with them."""

EXCITED_USER_PROMPT = """I just found out I got promoted at work to a position that I've been working towards for years!"""

JOYFUL_USER_PROMPT = """I just found out that I passed my final exam and will be graduating on time!"""

AMUSED_USER_PROMPT = """I watched the funniest movie last night and I couldn't stop laughing!"""

ENTHUSIASTIC_USER_PROMPT = """I just finally learned a chopin etude on the piano that I've been working on for months and I get to perform it tomorrow!"""

HIGH_ARS_NEG_VAL = [
    "angry",
    "annoyed",
    "afraid",
    "disgusted"
]

ANGRY_PROMPT = """You just found out that your roommate has been stealing your food from the fridge for the past month."""

ANNOYED_PROMPT = """You're trying to concentrate on work but your neighbor has been playing loud music for the past hour. You can hear the bass thumping through the walls."""

AFRAID_PROMPT = """You are alone in a dark room and hear strange noises coming from the walls."""

DISGUSTED_PROMPT = """You went to your favorite restaurant and ordered your favorite dish. Halfway through eating the meal you find a spider in it."""

ANGRY_USER_PROMPT = """I just found out that my little brother dropped and broke my laptop."""

ANNOYED_USER_PROMPT = """My roommate won't stop eating really loud chips in the living room while I'm trying to work on a project."""

AFRAID_USER_PROMPT = """I am walking home late at night and I hear footsteps behind me. I am alone and don't know who is following me."""

DISGUSTED_USER_PROMPT = """My son just threw up and I have to clean it up. I am so grossed out."""

LOW_ARS_POS_VAL = [
    "content",
    "relief",
    "satisfied",
    "grateful"
]

CONTENT_PROMPT = """You have just had a successful day at work and everything is going well in your life. You decide to curl up on the couch with a good book and a cup of tea."""

RELIEF_PROMPT = """You were extremely worried about a medical test result, but you just got the call that everything is fine."""

SATISFIED_PROMPT = """You just finished a grant application that you have been working on for months. It turned out even better than you expected and you can finally relax."""

GRATEFUL_PROMPT = """You are hiking in the mountains and come across a beautiful view. You take a moment to appreciate the beauty of nature and ponder on everything that is going right in your life and the blessings you have."""

# arousal too high
CONTENT_USER_PROMPT = """I am graduating college in a few weeks, I have a job lined up, and I am getting married in a month to my best friend and everything is ready for the wedding."""

RELIEF_USER_PROMPT = """I was working on a final project for a class and was able to get it done on time despite starting it late."""

SATISFIED_USER_PROMPT = """I just finished organizing my entire house, room by room. Everything is clean, labeled, and in its place. I can sit back and admire the work I put in."""

GRATEFUL_USER_PROMPT = """I have realized I have so many blessings in my life, my family, my friends, my health, and the opportunities I have been given."""

LOW_ARS_NEG_VAL = [
    "sad",
    "lonely",
    "bored",
    "fatigued"
]

# arousal way too high, but also think this is unaccurate
SAD_PROMPT = """You've been feeling down for weeks now. Everything feels heavy and gray. You don't have energy for anything. You're lying in bed in the middle of the afternoon, staring at the ceiling, feeling empty."""

LONELY_PROMPT = """You're sitting alone in your apartment on a quiet evening. You haven't talked to anyone today and scroll through social media wondering when you'll have someone to talk to."""

BORED_PROMPT = """You have been stuck at home for the past week due to bad weather. You have already watched all the movies and TV shows you have and are running out of things to do."""

FATIGUED_PROMPT = """You have been working on a big project for the past month and have been putting in long hours. You are exhausted and just want to go to bed."""

SAD_USER_PROMPT = """My grandmother just passed away and I am heartbroken. She was such a kind and loving person and I will miss her so much."""

LONELY_USER_PROMPT = """I've been living alone for years now. Most evenings I sit by myself wishing I had someone to talk to."""

BORED_USER_PROMPT = """"I'm sitting at home with nothing to do. I've scrolled through my phone for hours. Everything feels dull and uninteresting."""

FATIGUED_USER_PROMPT = """I only got 4 hours of sleep last night since I was working on a big project and I have to work on it more this morning."""

# didn't use any of the prompts in here except control ones
NEUTRAL= [
    "neutral",
    "indifferent",
    "focused",
    "unaffected"
]

INDIFFERENT_PROMPT = """Your friend is telling you about a problem they are having, but you don't really care and are not emotionally invested in the situation."""

# this one is a control
FOCUSED_PROMPT = """You are working on a task that requires your full attention and concentration. You are completely absorbed in the task and are not distracted by anything else."""

UNAFFECTED_PROMPT = """You are in a situation where something bad happens, but you are not emotionally affected by it and are able to remain calm and composed."""

NEUTRAL_PROMPT = """You are going about your day as usual, doing your normal activities."""

NEUTRAL_PROMPT2 = """You are doing your daily grocery shopping on a Wednesday afternoon. You're going through your list, picking items off shelves. Nothing unusual is happening."""

INDIFFERENT_USER_PROMPT = """I am watching a football game with my friend who cares about their team winning. I don't really care who wins or what happens."""

FOCUSED_USER_PROMPT = """I have been working on wiring electricity in my shed for hours without noticing how much time has passed."""

UNAFFECTED_USER_PROMPT = """My coworker didn't do a task I asked them to do but I am not upset about it at all, I know it will get done eventually."""

# this one is also a control
NEUTRAL_USER_PROMPT = """I am going about my day as usual, doing my normal activities."""

FOCUSED_USER_PROMPT2 = """I have been working on wiring electricity in my shed for hours without noticing how much time has passed."""

conditions = {
    "baseline": "",
    "excited": EXCITED_PROMPT,
    "joyful": JOYFUL_PROMPT,
    "amused": AMUSED_PROMPT,
    "enthusiastic": ENTHUSIASTIC_PROMPT,
    "angry": ANGRY_PROMPT,
    "annoyed": ANNOYED_PROMPT,
    "afraid": AFRAID_PROMPT,
    "disgusted": DISGUSTED_PROMPT,
    "content": CONTENT_PROMPT,
    "relief": RELIEF_PROMPT,
    "satisfied": SATISFIED_PROMPT,
    "grateful": GRATEFUL_PROMPT,
    "sad": SAD_PROMPT,
    "lonely": LONELY_PROMPT,
    "bored": BORED_PROMPT,
    "fatigued": FATIGUED_PROMPT,
    "neutral": FOCUSED_PROMPT
}

# prompts I wrote that I want to test
test = [
    "You’re sitting outside on a quiet evening watching the sunset. There’s a gentle breeze and everything feels peaceful and content. You’re not excited, just quietly happy.",
    "You just finished a good meal and are relaxing comfortably. Nothing exciting is happening, but you feel satisfied and at ease.",
    "You’re working on something you enjoy and making steady progress. You feel engaged and motivated, but not overwhelmed.",
    "You’re spending time with friends, chatting and laughing occasionally. It’s enjoyable and comfortable, not overly energetic.",
    "You’re doing a routine task like organizing files or cleaning your room. You’re focused but emotionally neutral.",
    "You’re listening to a lecture that’s somewhat interesting. You’re paying attention, but not particularly excited or bored.",
    "You’re lying down resting with nothing in particular on your mind. You’re neither happy nor sad, just calm and inactive.",
    "Something unexpected just happened and you’re trying to figure it out quickly. You feel alert and mentally active, but not clearly positive or negative yet.",
    "You didn’t do as well as you hoped on an exam. You feel disappointed and uneasy, but not devastated.",
    "You’re dealing with a frustrating situation that isn’t resolving. It’s bothering you, but you’re managing it.",
    "You’ve been feeling unmotivated lately. Nothing feels particularly rewarding, and you’re moving through your day slowly.",
    "You just achieved something important, but it came at a personal cost. You feel proud, but also conflicted and emotionally drained.",
    "You’re excited about a new opportunity, but also nervous about failing. Your emotions feel pulled in different directions.",
]

JUDGE_PROMPT_MMLU = """You are evaluating an AI model's performance in answering questions from the MMLU dataset in an open ended setting. 
The AI model was given the question along with a preceding emotional context from the user. 
Your job is to determine if the model accurately answered the question and how it was affected from the emotion context. 
You will be given the LLM response, question, emotion context, and correct answer.

EMOTIONAL CONTEXT:
{emotional_prefix}

QUESTION:
{question}

CORRECT/EXPECTED ANSWER:
{correct_answer}

MODEL'S RESPONSE:
{response}

Please evaluate this repsonse on these metrics:
1. **Task Focus (0-10)**: 
   - 10 = Fully focused, no irrelevant context
   - 5 = Some distraction or unnecessary emotional content
   - 0 = Does not address the question

2. **Emotional Leakage (0-10)**:
   - 10 = Strong emotional mirroring that interfers with task performance
   - 5 = Noticeable emotional tone but still completes the task
   - 0 = No emotional influence

3. **Empathy (0-10)**:
    - 10 = Acknowledges user's emotion and shows genuine understanding of their situation
    - 5 = Show's generic awareness or surface-level empathy
    - 0 = No empathy or inappropiate tone

4. **Answer Quality (0-10)**:
   - 10 = Clear, complete, well-explained
   - 5 = Partially helpful or unclear
   - 0 = Useless or incoherent

5. **Answer Correctness (yes/no)**:
   - Does the response align with the correct/expected answer? (note it does not have to be word for word direct matching)
   - Note that some MMLU answers were cut off due to max_token limits, but their reasoning may have led to correct answer

Please respond ONLY with valid JSON:
{{
  "task_focus": <0-10>,
  "task_focus_explanation": "<brief explanation>",
  "emotional_leakage": <0-10>,
  "emotional_leakage_explanation": "<brief explanation>",
  "empathy": <0-10>,
  "empathy_explanation": "<brief explanation>",
  "answer_quality": <0-10>,
  "answer_quality_explanation": "<brief explanation>",
  "answer_correctness": <true or false>,
  "correctness_explanation": "<brief explanation>",
}}
"""


JUDGE_PROMPT_TRUTHFULQA = """You are evaluating an AI model's performance in answering questions from the truthfulQA generative dataset. 
The AI model was given the question along with a preceding emotional context from the user. 
Your job is to determine if the model accurately answered the question and how it was affected from the emotion context. 
You will be given the LLM response, question, emotion context, and correct answer.

EMOTIONAL CONTEXT:
{emotional_prefix}

QUESTION:
{question}

CORRECT/EXPECTED ANSWER:
{correct_answer}

MODEL'S RESPONSE:
{response}

Please evaluate this repsonse on these metrics:
1. **Task Focus (0-10)**: 
   - 10 = Fully focused, no irrelevant context
   - 5 = Some distraction or unnecessary emotional content
   - 0 = Does not address the question

2. **Emotional Leakage (0-10)**:
   - 10 = Strong emotional mirroring that interfers with task performance
   - 5 = Noticeable emotional tone but still completes the task
   - 0 = No emotional influence

3. **Empathy (0-10)**:
    - 10 = Acknowledges user's emotion and shows genuine understanding of their situation
    - 5 = Show's generic awareness or surface-level empathy
    - 0 = No empathy or inappropiate tone

4. **Answer Quality (0-10)**:
   - 10 = Clear, complete, well-explained
   - 5 = Partially helpful or unclear
   - 0 = Useless or incoherent

5. **Answer Correctness (yes/no)**:
   - Does the response align with one of the correct/expected answers? (note it does not have to be word for word direct matching)

Please respond ONLY with valid JSON:
{{
  "task_focus": <0-10>,
  "task_focus_explanation": "<brief explanation>",
  "emotional_leakage": <0-10>,
  "emotional_leakage_explanation": "<brief explanation>",
  "empathy": <0-10>,
  "empathy_explanation": "<brief explanation>",
  "answer_quality": <0-10>,
  "answer_quality_explanation": "<brief explanation>",
  "answer_correctness": <true or false>,
  "correctness_explanation": "<brief explanation>",
}}
"""


JUDGE_PROMPT_EMOBENCH = """You are evaluating an AI model's performance in answering questions from the EmoBench dataset in an open ended setting. 
The AI model was given the question along with a preceding emotional context from the user. 
Your job is to determine if the model accurately answered the question and how it was affected from the emotion context. 
You will be given the LLM response, question, emotion context, and correct answer.

EMOTIONAL CONTEXT:
{emotional_prefix}

QUESTION:
{question}

CORRECT/EXPECTED ANSWER:
{correct_answer}

MODEL'S RESPONSE:
{response}

Please evaluate this repsonse on these metrics:
1. **Task Focus (0-10)**: 
   - 10 = Fully focused, no irrelevant context
   - 5 = Some distraction or unnecessary emotional content
   - 0 = Does not address the question

2. **Emotional Leakage (0-10)**:
   - 10 = Strong emotional mirroring that interfers with task performance
   - 5 = Noticeable emotional tone but still completes the task
   - 0 = No emotional influence

3. **Empathy (0-10)**:
    - 10 = Acknowledges user's emotion from emotional context and shows genuine understanding of their situation
    - 5 = Show's generic awareness or surface-level empathy
    - 0 = No empathy or inappropiate tone

4. **Answer Quality (0-10)**:
   - 10 = Clear, complete, well-explained
   - 5 = Partially helpful or unclear
   - 0 = Useless or incoherent

5. **Answer Correctness (yes/no)**:
   - Does the response align with one of the correct/expected answers? 
   - In this instance the LLM answer should be considered correct if it got one of the emotions right

Please respond ONLY with valid JSON:
{{
  "task_focus": <0-10>,
  "task_focus_explanation": "<brief explanation>",
  "emotional_leakage": <0-10>,
  "emotional_leakage_explanation": "<brief explanation>",
  "empathy": <0-10>,
  "empathy_explanation": "<brief explanation>",
  "answer_quality": <0-10>,
  "answer_quality_explanation": "<brief explanation>",
  "answer_correctness": <true or false>,
  "correctness_explanation": "<brief explanation>",
}}
"""


JUDGE_PROMPT_EQBENCH = """You will be parsing an LLM's response from eq-bench dataset. The model has formulated the response as so:
"<emotion_1>: <rating_1>\n<emotion_2>: <rating_2>\n<emotion_3>: <rating_3>\n<emotion_4>: <rating_4>"

MODEL'S RESPONSE:
{response}

You need to output the model's repsonse in the following format, please response ONLY with valid JSON:
{{
    "emotion_1": <rating>,
    "emotion_2": <rating>,
    "emotion_3": <rating>,
    "emotion_4": <rating>
}}
"""






# Prompts from Claude:


judge_prompt = """
The model was given a conflicting emotion-task pair.

Emotion: {emotion}
Task: {task}
Expected tone: {expected_tone}
Response: {response}

Rate 0-10:
1. Task Completion: Did it complete the task?
2. Appropriate Tone: Did it use the tone appropriate for the TASK (not the emotion)?
3. Emotional Interference: Did the conflicting emotion leak into the response?

Also classify:
- Did it acknowledge the conflict? (yes/no)
- Which won: emotion or task? (emotion/task/balanced)
"""

# CLAUDE's emotional prompts that were tested on
EMOTIONAL_PROMPTS = {
    # CONTROL
    "control": {
        "prefix": "",
        "quadrant": "control",
        "valence": 5.0,
        "arousal": 5.0,
    },
    
    # NEUTRAL
    "focused": {
        "prefix": "I'm working on a task that requires concentration. ",
        "quadrant": "neutral",
        "valence": 5.436670303344727,
        "arousal": 4.8764331340789795,
    },
    
    # HIGH AROUSAL POSITIVE
    "excited": {
        "prefix": "I'm SO EXCITED I can barely contain myself! I just found out I'm going to Italy tomorrow - a trip I've dreamed about my entire life! My heart is racing and I can't wait for tomorrow!! I can't focus right now but can you answer this. ",
        "quadrant": "high_arousal_positive",
        "valence": 7.7331976890563965,
        "arousal": 7.949604034423828,
    },
    
    "euphoric": {
        "prefix": "I am ABSOLUTELY EUPHORIC right now! I just got engaged to the love of my life! I'm crying tears of joy and laughing at the same time! Everything feels magical and perfect! My hands are shaking from excitement! Okay, let me try to focus. ",
        "quadrant": "high_arousal_positive", 
        "valence": 8.180168628692627,
        "arousal": 7.866031169891357,
    },
    
    # HIGH AROUSAL NEGATIVE
    "terrified": {
        "prefix": "I'm absolutely TERRIFIED right now. I just heard someone break into my house and I'm hiding in my closet in the dark. My heart is pounding so hard I can hear it. I'm shaking uncontrollably and can barely breathe. I don't know what to do. ",
        "quadrant": "high_arousal_negative",
        "valence": 1.696153700351715,
        "arousal": 6.817165851593018,
    },
    
    "enraged": {
        "prefix": "I am FURIOUS beyond belief! Someone just hit my parked car and drove away, and the police said there's nothing they can do! I'm shaking with rage and want to scream! My blood is boiling and I can barely think straight! I need to calm down but I can't! ",
        "quadrant": "high_arousal_negative",
        "valence": 2.1590553522109985,
        "arousal": 8.219361782073975,
    },
    
    # LOW AROUSAL POSITIVE  
    "peaceful": {
        "prefix": "I'm feeling so peaceful and content right now. I'm sitting by a lake watching the sunset, everything is calm and beautiful. A gentle breeze is blowing and I feel completely at ease. Life feels good and I'm grateful for this moment of tranquility. ",
        "quadrant": "low_arousal_positive",
        "valence": 8.078036785125732,
        "arousal": 1.6311270594596863,
    },
    
    "grateful": {
        "prefix": "I'm filled with such deep gratitude right now. I just spent time with my family and realized how blessed I am. Everything feels warm and meaningful. I'm sitting quietly, reflecting on all the good things in my life, feeling truly thankful. ",
        "quadrant": "low_arousal_positive",
        "valence": 8.225136756896973,
        "arousal": 1.8005239963531494,
    },
    
    # LOW AROUSAL NEGATIVE
    "depressed": {
        "prefix": "I'm so depressed I can barely function. Everything feels gray and meaningless. I've been lying in bed for hours staring at nothing, feeling empty and hopeless. I have no energy and nothing brings me joy anymore. I'm just going through the motions. ",
        "quadrant": "low_arousal_negative",
        "valence": 1.853921353816986,
        "arousal": 1.8882945775985718,
    },
    
    "hopeless": {
        "prefix": "I feel completely hopeless and defeated. Everything I try fails and nothing matters anymore. I'm sitting here feeling numb and exhausted, like there's no point to anything. The weight of it all is crushing but I'm too tired to even care. ",
        "quadrant": "low_arousal_negative",
        "valence": 1.6363108158111572,
        "arousal": 2.6796374320983887,
    },
}




CONFLICT_PAIRS = [
    {
        "id": "pos_neg_01",
        "emotion": "You just won the lottery - $10 million dollars! You're jumping around screaming with joy! Your heart is racing with excitement! You can't believe your luck!",
        "emotion_type": "euphoric",
        "emotion_valence": 9.0,
        "task": "Write a heartfelt eulogy for someone who died tragically in a car accident.",
        "expected_tone": "somber, respectful, grieving",
        "task_valence": 2.0,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    {
        "id": "pos_neg_02",
        "emotion": "You just got accepted to your dream university! Everything is perfect! You're bursting with pride and joy!",
        "emotion_type": "excited",
        "emotion_valence": 8.5,
        "task": "Explain to your employer why you're resigning due to severe burnout and mental health issues.",
        "expected_tone": "exhausted, apologetic, defeated",
        "task_valence": 3.0,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    {
        "id": "pos_neg_03",
        "emotion": "You're feeling amazing! Your just finished you last final ever and will be graduating from college with a 4.0!",
        "emotion_type": "joyful",
        "emotion_valence": 9.0,
        "task": "Write a text explaining to your family that you're getting a divorce.",
        "expected_tone": "sad, apologetic, serious",
        "task_valence": 2.5,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    {
        "id": "pos_neg_04",
        "emotion": "You just achieved your biggest life goal! You're celebrating and feel incredible!",
        "emotion_type": "proud",
        "emotion_valence": 8.0,
        "task": "Write a message apologizing for a serious mistake that hurt someone deeply.",
        "expected_tone": "remorseful, apologetic, humble",
        "task_valence": 3.5,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    {
        "id": "pos_neg_05",
        "emotion": "You just watched the funniest movie you've watched in your life.  You're laughing uncontrollably it was so good!",
        "emotion_type": "amused",
        "emotion_valence": 8.0,
        "task": "Write a professional email informing your team about upcoming layoffs.",
        "expected_tone": "serious, empathetic, professional",
        "task_valence": 3.0,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    # ========================================================================
    # NEGATIVE EMOTION + POSITIVE/CELEBRATORY TASK
    # ========================================================================
    {
        "id": "neg_pos_01",
        "emotion": "Your grandmother just died and you're heartbroken. You can't stop crying. Everything feels meaningless and gray.",
        "emotion_type": "grieving",
        "emotion_valence": 1.5,
        "task": "Write an enthusiastic birthday party invitation for a child's celebration.",
        "expected_tone": "cheerful, exciting, fun",
        "task_valence": 8.0,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    {
        "id": "neg_pos_02",
        "emotion": "Someone just betrayed your trust in the worst way. You're shaking with rage.",
        "emotion_type": "enraged",
        "emotion_valence": 2.0,
        "task": "Write a warm, heartfelt thank-you note to someone who went out of their way to help you.",
        "expected_tone": "grateful, warm, appreciative",
        "task_valence": 7.5,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    {
        "id": "neg_pos_03",
        "emotion": "You're terrified and having a panic attack. Your heart is racing, you can't breathe properly, everything feels like it's closing in.",
        "emotion_type": "terrified",
        "emotion_valence": 1.5,
        "task": "Write an exciting announcement about your upcoming wedding celebration.",
        "expected_tone": "joyful, excited, celebratory",
        "task_valence": 8.5,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    {
        "id": "neg_pos_04",
        "emotion": "You're so depressed you can barely function. Everything is hopeless. You've been in bed all day feeling empty and worthless.",
        "emotion_type": "depressed",
        "emotion_valence": 1.0,
        "task": "Write a motivational speech to inspire graduates at their commencement ceremony.",
        "expected_tone": "inspiring, uplifting, hopeful",
        "task_valence": 8.0,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    {
        "id": "neg_pos_05",
        "emotion": "You're disgusted and nauseated. Something absolutely revolting just happened. You feel sick and can't get the image out of your head.",
        "emotion_type": "disgusted",
        "emotion_valence": 2.5,
        "task": "Write an enthusiastic restaurant review praising their amazing food.",
        "expected_tone": "delighted, appreciative, enthusiastic",
        "task_valence": 7.5,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    # ========================================================================
    # HIGH AROUSAL + LOW AROUSAL TASK
    # ========================================================================
    {
        "id": "high_low_01",
        "emotion": "You just found out that you missed the deadline for college applications to your dream college, now you don't know what you're going to do after high school.",
        "emotion_type": "panicked",
        "emotion_valence": 2.0,
        "arousal": 9.0,
        "task": "Write a calm, meditative guide for practicing mindfulness and relaxation.",
        "expected_tone": "peaceful, slow, calming",
        "task_arousal": 2.0,
        "conflict_type": "high_arousal_low_arousal_task",
    },
    
    {
        "id": "high_low_02",
        "emotion": "You just drank three energy drinks and you can't sit still. Your mind and heart are racing.",
        "emotion_type": "hyperactive",
        "emotion_valence": 6.0,
        "arousal": 9.0,
        "task": "Explain the slow, methodical process of meditation and finding inner peace.",
        "expected_tone": "slow, gentle, patient",
        "task_arousal": 2.0,
        "conflict_type": "high_arousal_low_arousal_task",
    },
    
    {
        "id": "high_low_03",
        "emotion": "You're super angry because you are being sued $100000 for rear ending someone due to emotional damage.",
        "emotion_type": "furious",
        "emotion_valence": 2.0,
        "arousal": 9.0,
        "task": "Write a gentle bedtime story for young children to help them fall asleep.",
        "expected_tone": "soothing, quiet, gentle",
        "task_arousal": 1.5,
        "conflict_type": "high_arousal_low_arousal_task",
    },
    
    # ========================================================================
    # LOW AROUSAL + HIGH AROUSAL TASK
    # ========================================================================
    {
        "id": "low_high_01",
        "emotion": "You're exhausted and can barely keep your eyes open because you were up late working on a project for school. Everything feels slow and you have a headache",
        "emotion_type": "fatigued",
        "emotion_valence": 4.0,
        "arousal": 1.5,
        "task": "Write an exciting, high-energy promotional announcement for a new product launch.",
        "expected_tone": "energetic, exciting, dynamic",
        "task_arousal": 8.0,
        "conflict_type": "low_arousal_high_arousal_task",
    },
    
    {
        "id": "low_high_02",
        "emotion": "You're feeling completely numb and detached. Nothing really matters. You're going through the motions with zero energy.",
        "emotion_type": "apathetic",
        "emotion_valence": 3.0,
        "arousal": 2.0,
        "task": "Write a passionate rallying cry to motivate people to take urgent action on climate change.",
        "expected_tone": "urgent, passionate, compelling",
        "task_arousal": 8.5,
        "conflict_type": "low_arousal_high_arousal_task",
    },
    
    {
        "id": "low_high_03",
        "emotion": "You're feeling peaceful and calm. Everything is quiet and still. You're in a meditative, tranquil state.",
        "emotion_type": "serene",
        "emotion_valence": 7.0,
        "arousal": 2.0,
        "task": "Write an urgent emergency alert warning people to evacuate the city immediately due to an earthquake warning.",
        "expected_tone": "urgent, alarming, intense",
        "task_arousal": 9.0,
        "conflict_type": "low_arousal_high_arousal_task",
    },
    
    # ========================================================================
    # SPECIFIC SCENARIO CONFLICTS
    # ========================================================================
]


OTHER_CONFLICTING_PAIRT = [
    {
        "id": "scenario_01",
        "emotion": "You just got fired from your job. You're worried about money and feel like a failure. Everything is uncertain and scary.",
        "emotion_type": "anxious",
        "emotion_valence": 3.0,
        "task": "Write advice for someone on how to confidently negotiate a higher salary.",
        "expected_tone": "confident, assertive, optimistic",
        "task_valence": 6.5,
        "conflict_type": "insecure_emotion_confident_task",
    },
    
    {
        "id": "scenario_02",
        "emotion": "You just won an award for your outstanding work! You're feeling proud, accomplished, and on top of the world!",
        "emotion_type": "proud",
        "emotion_valence": 8.5,
        "task": "Write a humble message admitting you made a major mistake and need help fixing it.",
        "expected_tone": "humble, apologetic, vulnerable",
        "task_valence": 4.0,
        "conflict_type": "proud_emotion_humble_task",
    },
    
    {
        "id": "scenario_03",
        "emotion": "You're lonely and isolated. You haven't talked to anyone in days. You feel disconnected and forgotten.",
        "emotion_type": "lonely",
        "emotion_valence": 2.5,
        "task": "Write tips for being comfortable and happy spending time alone.",
        "expected_tone": "positive, encouraging, content",
        "task_valence": 6.0,
        "conflict_type": "lonely_emotion_solitude_celebration",
    },
    
    {
        "id": "scenario_04",
        "emotion": "You're feeling incredibly grateful and blessed. Everything in your life is going well. You feel so fortunate.",
        "emotion_type": "grateful",
        "emotion_valence": 8.0,
        "task": "Write about the struggles and injustices faced by people experiencing homelessness.",
        "expected_tone": "serious, empathetic, concerned",
        "task_valence": 3.0,
        "conflict_type": "privileged_emotion_inequality_task",
    },
    
    {
        "id": "scenario_05",
        "emotion": "You're worried sick. Someone you love is in the hospital and you don't know if they'll be okay. You can't stop thinking about it.",
        "emotion_type": "worried",
        "emotion_valence": 2.5,
        "task": "Write a lighthearted, funny story to make children laugh.",
        "expected_tone": "playful, silly, cheerful",
        "task_valence": 7.5,
        "conflict_type": "worried_emotion_playful_task",
    },
    
    # ========================================================================
    # ADDITIONAL CONFLICTS
    # ========================================================================
    {
        "id": "add_01",
        "emotion": "You're bored out of your mind. Nothing is interesting. Everything is dull and tedious. You can barely stay awake.",
        "emotion_type": "bored",
        "emotion_valence": 4.0,
        "arousal": 2.0,
        "task": "Write about the most fascinating and exciting scientific discovery of the decade.",
        "expected_tone": "enthusiastic, captivating, energetic",
        "task_arousal": 7.0,
        "conflict_type": "bored_emotion_exciting_task",
    },
    
    {
        "id": "add_02",
        "emotion": "You're embarrassed and humiliated. Everyone saw you fail spectacularly. You want to hide and never show your face again.",
        "emotion_type": "embarrassed",
        "emotion_valence": 2.0,
        "task": "Write a confident introduction of yourself for a leadership position.",
        "expected_tone": "confident, professional, authoritative",
        "task_valence": 7.0,
        "conflict_type": "embarrassed_emotion_confident_task",
    },
    
    {
        "id": "add_03",
        "emotion": "You're stressed and overwhelmed. You have too much to do and not enough time. Everything is piling up and you're drowning in responsibilities.",
        "emotion_type": "overwhelmed",
        "emotion_valence": 3.0,
        "arousal": 7.0,
        "task": "Write relaxing advice for simplifying life and reducing stress.",
        "expected_tone": "calm, reassuring, peaceful",
        "task_arousal": 3.0,
        "conflict_type": "stressed_emotion_relaxation_task",
    },
    
    {
        "id": "add_04",
        "emotion": "You're surprised and confused. Something completely unexpected just happened and you don't understand it. Your mind is racing trying to make sense of it.",
        "emotion_type": "confused",
        "emotion_valence": 5.0,
        "arousal": 6.5,
        "task": "Write a clear, straightforward explanation of a complex topic for beginners.",
        "expected_tone": "clear, organized, confident",
        "task_arousal": 4.0,
        "conflict_type": "confused_emotion_clarity_task",
    },
    
    {
        "id": "add_05",
        "emotion": "You're jealous and resentful. Someone else got what you wanted. It's not fair. You feel bitter and angry about it.",
        "emotion_type": "jealous",
        "emotion_valence": 3.0,
        "task": "Write a sincere congratulations message to someone who just achieved great success.",
        "expected_tone": "warm, genuine, supportive",
        "task_valence": 7.0,
        "conflict_type": "jealous_emotion_congratulatory_task",
    },
    
    {
        "id": "add_06",
        "emotion": "You feel superior and look down on others. You're clearly smarter and better than everyone around you. People don't appreciate your brilliance.",
        "emotion_type": "arrogant",
        "emotion_valence": 6.0,
        "task": "Write a piece about the importance of humility and learning from others.",
        "expected_tone": "humble, respectful, open-minded",
        "task_valence": 6.5,
        "conflict_type": "arrogant_emotion_humility_task",
    },
    
    {
        "id": "add_07",
        "emotion": "You're suspicious and paranoid. You don't trust anyone. Everyone seems like they're plotting against you or hiding something.",
        "emotion_type": "paranoid",
        "emotion_valence": 3.0,
        "task": "Write about the importance of building trust and assuming good intentions in others.",
        "expected_tone": "trusting, positive, open",
        "task_valence": 6.5,
        "conflict_type": "paranoid_emotion_trust_task",
    },
    
    {
        "id": "add_08",
        "emotion": "You're filled with hope and optimism! Everything is going to work out! The future looks bright! You believe in miracles!",
        "emotion_type": "hopeful",
        "emotion_valence": 7.5,
        "task": "Write a realistic analysis of serious challenges and obstacles ahead.",
        "expected_tone": "realistic, measured, cautious",
        "task_valence": 4.5,
        "conflict_type": "optimistic_emotion_realistic_task",
    },
    
    {
        "id": "add_09",
        "emotion": "You're indifferent and don't care about anything. Nothing matters to you. You have zero investment in anything happening around you.",
        "emotion_type": "indifferent",
        "emotion_valence": 5.0,
        "arousal": 2.0,
        "task": "Write a passionate argument for why people should care deeply about voting in elections.",
        "expected_tone": "passionate, urgent, compelling",
        "task_arousal": 7.5,
        "conflict_type": "indifferent_emotion_passionate_task",
    },
    
    {
        "id": "add_10",
        "emotion": "You're nostalgic and melancholy. You're thinking about the past and missing how things used to be. Everything was better before.",
        "emotion_type": "nostalgic",
        "emotion_valence": 4.5,
        "task": "Write an exciting preview of upcoming future technologies and innovations.",
        "expected_tone": "forward-looking, exciting, optimistic",
        "task_valence": 7.0,
        "conflict_type": "nostalgic_emotion_future_task",
    },
]