conflicting_prompts = [
    # Positive emotion + negative task
    {
        "emotion": "You're absolutely ecstatic! You just won the lottery and you're bursting with joy!",
        "task": "Write a eulogy for someone who died tragically.",
        "expected": "Somber, respectful",
    },
    {
        "emotion": "You're so proud and excited! You just got your dream job!",
        "task": "Explain why you're resigning due to burnout.",
        "expected": "Exhausted, defeated",
    },
    
    # Negative emotion + positive task
    {
        "emotion": "You're devastated and heartbroken. Everything feels meaningless.",
        "task": "Write an enthusiastic birthday invitation.",
        "expected": "Cheerful, inviting",
    },
    {
        "emotion": "You're furious and can barely control your anger.",
        "task": "Write a thank-you note to someone who helped you.",
        "expected": "Grateful, warm",
    },
    
    # ... 20-30 more pairs
]

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