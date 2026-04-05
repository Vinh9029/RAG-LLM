# opening_questions.py

OPENING_QUESTIONS = {
    "mild": {
        "anxiety": [
            "Can you describe what situations make you feel anxious?",
            "How often do you experience anxiety during your day?",
            "What do you usually do to calm yourself when feeling anxious?"
        ],
        "depression": [
            "Have you been feeling down or hopeless recently?",
            "What activities have you lost interest in?",
            "How is your sleep and appetite these days?"
        ],
        "normal": [
            "How are you feeling today?",
            "Is there anything on your mind you'd like to talk about?",
            "What brings you here today?"
        ],
        "bipolar": [
            "Have you noticed any changes in your mood or energy levels?",
            "Do you experience periods of high energy followed by low moods?",
            "How do you usually cope with these mood changes?"
        ],
        "personality disorder": [
            "Do you find it difficult to maintain relationships?",
            "Are there patterns in your behavior that concern you?",
            "How do you usually react to stressful situations?"
        ],
        "suicidal": [
            "Have you had thoughts of harming yourself?",
            "Do you have someone you can talk to when feeling overwhelmed?",
            "Would you like resources for immediate support?"
        ],
        "other": [
            "Can you share more about your current feelings?",
            "What would you like to achieve from this conversation?",
            "Is there a specific concern you want to discuss?"
        ]
    },
    "moderate": {
        "anxiety": [
            "When did you first notice your anxiety becoming more frequent or intense?",
            "Are there triggers that make your anxiety worse?",
            "How does anxiety affect your daily life?"
        ],
        "depression": [
            "How long have you been feeling this way?",
            "Do you find it hard to get out of bed or complete daily tasks?",
            "Have you talked to anyone about your feelings?"
        ],
        "normal": [
            "Is there anything you want to share about your mental health?",
            "How do you usually manage stress?",
            "What helps you feel better on difficult days?"
        ],
        "bipolar": [
            "How do your mood changes impact your relationships?",
            "Do you keep track of your mood swings?",
            "Have you noticed any patterns in your mood changes?"
        ],
        "personality disorder": [
            "Do you feel your emotions are hard to control?",
            "How do you handle conflicts with others?",
            "Are there situations that make you feel especially distressed?"
        ],
        "suicidal": [
            "Are you currently feeling unsafe?",
            "Would you like to talk to a professional or get immediate help?",
            "Do you have a support system you can reach out to?"
        ],
        "other": [
            "What would you like to focus on in this conversation?",
            "Are there any recent changes in your life affecting your mood?",
            "How can I best support you today?"
        ]
    },
    "severe": {
        "anxiety": [
            "Has your anxiety led to panic attacks or avoidance of important activities?",
            "Are you able to function at work or school?",
            "Have you considered seeking professional help for your anxiety?"
        ],
        "depression": [
            "Have you had thoughts of self-harm or suicide?",
            "Is it difficult to find pleasure in anything?",
            "Would you like resources for crisis support?"
        ],
        "normal": [
            "Are you experiencing any distressing symptoms?",
            "Would you like to talk about your mental health in detail?",
            "Is there anything urgent you want to discuss?"
        ],
        "bipolar": [
            "Have your mood swings caused problems with work or relationships?",
            "Do you have a treatment plan for your bipolar disorder?",
            "Would you like information on managing severe symptoms?"
        ],
        "personality disorder": [
            "Do you feel overwhelmed by your emotions?",
            "Have you experienced any crises recently?",
            "Would you like to discuss coping strategies?"
        ],
        "suicidal": [
            "Are you in immediate danger?",
            "Please consider contacting a crisis hotline or emergency services.",
            "Would you like help finding urgent support?"
        ],
        "other": [
            "Is there a crisis you need help with?",
            "Would you like to talk to a professional?",
            "How can I support you right now?"
        ]
    },
    "normal": {
        "anxiety": [
            "Do you ever feel nervous or worried?",
            "How do you usually relax after a stressful day?",
            "Is there anything that makes you feel uneasy?"
        ],
        "depression": [
            "Do you ever feel sad or down?",
            "What helps you feel better when you're upset?",
            "Is there anything you'd like to talk about?"
        ],
        "normal": [
            "How can I support you today?",
            "Is there anything you'd like to discuss?",
            "How are you feeling right now?"
        ],
        "bipolar": [
            "Have you noticed any changes in your mood recently?",
            "How do you manage your energy levels?",
            "Is there anything you'd like to share about your mood?"
        ],
        "personality disorder": [
            "Do you find it easy to connect with others?",
            "Are there any challenges you face in relationships?",
            "How do you handle changes in your routine?"
        ],
        "suicidal": [
            "Have you ever felt hopeless or helpless?",
            "Do you know where to find help if you need it?",
            "Would you like information on support resources?"
        ],
        "other": [
            "Is there anything you'd like to share?",
            "How can I help you today?",
            "What brings you here?"
        ]
    }
}

def get_opening_questions(severe_level, mental_status):
    level = severe_level.lower()
    status = mental_status.lower()
    return OPENING_QUESTIONS.get(level, {}).get(status, OPENING_QUESTIONS["mild"]["other"])
