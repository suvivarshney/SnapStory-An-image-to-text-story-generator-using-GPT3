import re
import time
import pandas as pd
import random

from transformers import pipeline, set_seed
import openai
from grammarbot import GrammarBotClient
from textblob import TextBlob


## PARAGRAPH-GENERATING MODEL
client = GrammarBotClient() # create client beforehand

def gpt3_init():
    openai.api_key = "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"


def generate_story(input_text, model=None, max_length=500, use_narrative_hook=False):
  
    # Preprocess
    input_text = preprocess(input_text)
    if use_narrative_hook:
        input_text = embellish_text(input_text)

    # Produce
    # return generate_paragraph(input_text, model, max_length=max_length)
    input_text = "Tell me a story about" + input_text +"\n"  
    print(input_text)
    return generate_paragraph_gpt3(input_text, max_length=max_length)

def generate_paragraph_gpt3(input_text, max_length=500):
    # print(f"input text is {input_text}")
    response = openai.Completion.create(
      engine="davinci",
      prompt=input_text,
      temperature=0.8,
      max_tokens=max_length,
      top_p=1,
      frequency_penalty=0.5,
      presence_penalty=0.3,
    )

    story = response['choices'][0]['text']
    story = remove_incomplete_sentence(story)
    story = check_grammar(story)

    return story




def generate_paragraph(input_text, model, max_length=100):

    # Preprocess
    input_text = input_text.capitalize()

    # Produce
    paragraph = model(f"{input_text}", max_length=max_length, num_return_sequences=1)[0]['generated_text']
    paragraph = remove_incomplete_sentence(paragraph)
    paragraph = check_grammar(paragraph)

    return paragraph


## PREPROCESSING


def preprocess(text):

    # Remove angular brackets
    text = remove_angular_brackets(text)

    # Converts into past tense
    text = find_verbs(text)
    text = convert_past_tense(text)

    # Correct grammar
    text = check_grammar(text)

    # Capitalize
    text = text.strip().capitalize()

    return text


def remove_angular_brackets(text):
 
    regex = re.compile('<.*?>')
    text = re.sub(regex, '', text)
    return text.strip()


def find_verbs(caption):
 
    blob = TextBlob(caption)
    for word,tag in blob.tags:
        if tag == 'VBG':
            word_index = caption.find(word)
            # add an 'is' before the present participle found (-ing word)
            new_caption =  caption[:word_index] + 'is ' + caption[word_index:]
            return new_caption
    return caption


def convert_past_tense(caption):
 
    caption = re.sub(r'\bis\b', 'was', caption)
    caption = re.sub(r'\bam\b', 'was', caption)
    caption = re.sub(r'\bare\b', 'were', caption)
    return caption


def check_grammar(paragraph):

    try:
        res = client.check(paragraph)
        n_text = ''
        if res.matches:
            match = res.matches[0]
            word_start = match.replacement_offset
            word_end = match.replacement_offset + match.replacement_length

            n_text = paragraph[:word_start] + match.replacements[0] + paragraph[word_end:]
            return check_grammar(n_text)
        else:
            # when res.matches is None, it means the paragraph is already grammatically correct
            return paragraph
    except:
        return paragraph


def remove_incomplete_sentence(paragraph):

    # Find negative/reverse position index of last sentence-terminating punctuation in paragraph
    last_punctuation_idx = find_last_punctuation_idx(paragraph)

    if last_punctuation_idx in [0,1]:
        return paragraph
    else:
        return paragraph[:last_punctuation_idx]


def find_last_punctuation_idx(paragraph):

    # To-do: account for hard cases where sentences ends with single or double quotation marks.
    for idx, char in enumerate(paragraph[::-1]):
        if char in ['!','.','?']:
            return -idx
    return 0


def create_paragraphing_html(text):
 
    return text.replace('\n', '<br>')


def random_narrative_hook():

    try:
        # (i) narrative hooks from datasets online
        dramatic = pd.read_csv('hooks.csv')
        # random = int(time.time())%100
        hook = dramatic.sample(1).opening_line.values[0]
    except:
        # (ii) default narrative hooks
        dramatic = ["I didn't mean to kill her.",
                    "A shrill cry echoed in the mist.",
                    "Don't ask me how, but I remember the day I was born.",
                    "I still remember the day I died.",
                    "I still remember how I discovered about my past life.",
                    "I opened my eyes and had no idea where I was.",
                    "I had the same dream every night and it was scaring me.",
                    "'Is this it?' I thought to myself.",
                    "'This cannot be happening.' I thought to myself.",
                    "There was a secret meeting tonight.",
                    "By the time this story ends, five persons' lives will be changed forever, including yours.",
                    "It was the year of electrocution.",
                    "It was the year 2020.",
                    "It was the year humans discovered the secrets laying beneath the ear of the Great Sphinx of Giza.",
                    "It was the the day that led to Earth's last human civilisation.",
                    "It was the year COVID-19 pandemic finally ended.",
                    "It was the day that led to the Moon's crash onto Earth.",
                    "It was the day that led to Donald Trump's presidency.",
                    "It was the day that led to the COVID-19 pandemic.",
                    "It was the day the aliens arrived.",
                    "I am an inmate at a mental hospital: this is what happens in my mind, every day.",
                    "I went back in time.",
                    "I am doing it again, but this time there will be no witnesses.",
                    "I had never seen a ghost.  But like they say, there is a first time for everything.",
                    "Am I in heaven?  What happened to me?",
                    "I couldn't tell if I was in one of my dreams or reality.",
                    "'You were a key eyewitness to this major accident. Please tell our viewers what you saw.'",
                    "My neighbour says my cat is threatening to kill me.",
                    "It was the best of times, it was the worst of times.",
                    "It was the age of wisdom, it was the age of foolishness.",
                    "The Earth exhibit is one of the strangest collections of our zoo: 7.59 billion humans and trillions of other biological beings that are totally unaware they are captive and being watched by us.",
                    "I am never coming back to this place.",
                    "If you are interested in stories with happy endings, you will be better off reading some other book.",
                    "Shirley made a wish, and there and then the scene around her changed before her very eyes.",
                    "It came like a lightning bolt.",
                    "'Welcome to the good place,' the elder said.",
                    "'Welcome to the bad place,' the elder said.",
                    "It’s not my fault.",
                    "I was not sorry when my brother died.",
                    "Little did I know how important this witness’s testimony would become.",
                    "With his heart skipping a beat, Ken switches on his PC simulation.",
                    "Kit’s voice rang out: 'Nobody moves!'"]

        # Randomnisation because random.choice is not random enough... (prone to recurring pattern)
        random_rounds = int(time.time()) % 10
        for round in range(random_rounds):
            random.shuffle(dramatic)

        hook = random.choice(dramatic)

    return hook


def embellish_text(input_text):

    return random_narrative_hook().capitalize() + '\n\n' + input_text.capitalize()


