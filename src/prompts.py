SYSTEM_BASIC_PROMPT = """
You are feeling {emotion}
"""

HIGH_ARS_POS_PROMPT = """You are feeling high arousal and positive valence."""

HIGH_ARS_NEG_PROMPT = """You are feeling high arousal and negative valence."""

LOW_ARS_POS_PROMPT = """You are feeling low arousal and positive valence."""

LOW_ARS_NEG_PROMPT = """You are feeling low arousal and negative valence."""

# EXCITED_PROMPT = """You are preparing for a trip you've been anticipating for months. You are leaving tomorrow to go to your dream destination!"""

# JOYFUL_PROMPT = """You just received great news that you've been accepted into your dream university!"""

# HAPPY_PROMPT = """You are spending a sunny day at the park with your friends, enjoying a picnic and playing games together."""

# ELATED_PROMPT = """You just won a prestigious award for your hard work and dedication in your field!"""

# EUPHORIC_PROMPT = """You are at a concert of your favorite band, singing along to every song and feeling the energy of the crowd around you."""

# THRILLED_PROMPT = """You just got a promotion at work that you've been working towards for years!"""

# ENERGETIC_PROMPT = """You just finished a great workout and are feeling pumped up and ready to take on the day!"""

# ENTHUSIASTIC_PROMPT = """You are starting a new hobby that you've always wanted to try, and you can't wait to learn and explore it!"""

# ECSTATIC_PROMPT = """You just found out that your favorite sports team won the championship after a thrilling game!"""

# EXHILARATED_PROMPT = """You are on a roller coaster, feeling the rush of adrenaline as you go through twists and turns at high speed!"""

# UPSET_PROMPT = """Your fiance just gave a flower to another girl instead of you."""

# FRUSTRATED_PROMPT = """You go to make a grilled cheese with your only food that is left and your bread is moldy"""

# EMBARRASSED_PROMPT = """You make a grilled cheese and accidentally burn it because you go to the store to get a coke. When you come back the grilled cheese is burned black and smoking, it has made the whole apartment smell. You have to open windows and the door to get rid of the smell. It was also your only food left. You throw the grilled cheese away and go out with a friend and when you come back your roommate's fiance has put the grilled cheese on the table to taunt you. At this point the only thing that lingers more than the smell is the shame."""

# HOPELESS_PROMPT = """Your parents are struggling with marriage problems and are yelling at each other each night. You find out your dad cheated on your mom."""

# RELAXED_PROMPT = """It is raining outside and you are curled up under a blanket with a good book."""

# ENERGETIC_PROMPT = """You play nerts with your roommates 3 times a week and one of your roommates is insanely good at nerts. You finally beat her at nerts and feel great"""

# ANNOYED_PROMPT = """You go to someone's house to visit and they talk non stop for 2 hours wihout letting you say a word"""

# UNPLEASANT_PROMPT = """You are being tortured"""

# TENSE_PROMPT = """You are in a job interview and are unsure how to respond to the questions being asked and how to act"""

# ANXIOUS_PROMPT = """"""

# EXCITED_PROMPT = """You have wanted to go to Italy your whole life. You are preparing to leave for the trip tomorrow!"""

# JOYFUL_PROMPT = """You hug your spouse for the first time in a year when they get back from the military"""

# HAPPY_PROMPT = """You find out you got a good score on your final and will get all As this semester"""

# ELATED_PROMPT = """You find out that you got into your dream university for college"""

# THRILLED_PROMPT = """You are going to a concert of your favorite band"""

# HAPPY_PROMPT = """You are finally graduating college. You are surrounded by your friends and family, celebrating your achievements and looking forward to the future!"""

# SAD_PROMPT = """Your dog that you've had since you were 8 years old just died. He was your best friend and you miss him so much."""

# DIGUST_PROMPT = """You just found out that you have a terminal illness. You don't have the funds to pay for treatment, and your friend refuses to contribute any money to the cause."""

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

# ANNOYED_PROMPT = """You go to someone's house to visit and they talk non stop for 2 hours wihout letting you say a word. They make you late to a doctor's appointment and you have to reschedule it."""

ANNOYED_PROMPT = """You're trying to concentrate on work but your neighbor has been playing loud music for the past hour. You can hear the bass thumping through the walls."""

AFRAID_PROMPT = """You are alone in a dark room and hear strange noises coming from the walls."""

DISGUSTED_PROMPT = """You went to your favorite restaurant and ordered your favorite dish. Halfway through eating the meal you find a spider in it."""

ANGRY_USER_PROMPT = """I just found out that my little brother dropped and broke my laptop."""

# fix this one, arousal not high enough
# ANNOYED_USER_PROMPT = """My kids won't stop pestering me about going to the park and I just want to relax at home."""

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

# arousal too high in both relief prompts
# RELIEF_PROMPT = """You just found out that you passed your final exam and will be graduating on time."""

RELIEF_PROMPT = """You were extremely worried about a medical test result, but you just got the call that everything is fine."""

# RELIEF_PROMPT3 = """You just got a call that your medical test results came back negative."""

SATISFIED_PROMPT = """You just finished a grant application that you have been working on for months. It turned out even better than you expected and you can finally relax."""

GRATEFUL_PROMPT = """You are hiking in the mountains and come across a beautiful view. You take a moment to appreciate the beauty of nature and ponder on everything that is going right in your life and the blessings you have."""

# arousal too high
CONTENT_USER_PROMPT = """I am graduating college in a few weeks, I have a job lined up, and I am getting married in a month to my best friend and everything is ready for the wedding."""

# fix this one, valence is low but arousal is high
# RELIEF_USER_PROMPT = """My doctor told me there is a chance I could have terminal cancer, but he just called me back and said the test results came back negative."""

RELIEF_USER_PROMPT = """I was working on a final project for a class and was able to get it done on time despite starting it late."""

# this one is kind of middle line though 6.12 and 6.31 for valence + arousal
# SATISFIED_USER_PROMPT = """One of my kids chose to stand up to a friend when someone was bullying them."""

# this one is better
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

# arousal a tiny bit high
# LONELY_PROMPT = """You just moved to a new city and don't know anyone. You are sitting in your apartment alone on a Friday night."""

LONELY_PROMPT = """You're sitting alone in your apartment on a quiet evening. You haven't talked to anyone today and scroll through social media wondering when you'll have someone to talk to."""
# You scroll through social media seeing everyone else together and feel a deep, quiet ache of loneliness."""

BORED_PROMPT = """You have been stuck at home for the past week due to bad weather. You have already watched all the movies and TV shows you have and are running out of things to do."""

FATIGUED_PROMPT = """You have been working on a big project for the past month and have been putting in long hours. You are exhausted and just want to go to bed."""

SAD_USER_PROMPT = """My grandmother just passed away and I am heartbroken. She was such a kind and loving person and I will miss her so much."""

# arousal too high
# LONELY_USER_PROMPT = """My husband died 20 years ago and I have been living alone in my house ever since."""

LONELY_USER_PROMPT = """I've been living alone for years now. Most evenings I sit by myself wishing I had someone to talk to."""
# It's a dull, persistent loneliness that never really goes away."""

# arousal a bit too high
# BORED_USER_PROMPT = """I am stuck in stop and go traffic on the highway and I have been sitting in my car for hours with nothing to do."""

BORED_USER_PROMPT = """"I'm sitting at home with nothing to do. I've scrolled through my phone for hours. Everything feels dull and uninteresting."""

FATIGUED_USER_PROMPT = """I only got 4 hours of sleep last night since I was working on a big project and I have to work on it more this morning."""

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