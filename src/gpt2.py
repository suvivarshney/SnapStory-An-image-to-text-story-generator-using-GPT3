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

def load_model():
    '''
    Loads and returns a pre-trained GPT-2 text-generator model (https://huggingface.co/gpt2)

    Returns
    -------
    model : transformers.pipelines.TextGenerationPipeline
        The pre-trained GPT-2 model
    '''
    model = pipeline('text-generation', model='gpt2')
    set_seed(42)
    return model


def generate_story(input_text, model=None, max_length=500, use_narrative_hook=False):
    '''
    Returns a story generated using
    (i) a pre-trained GPT-2 model, and
    (ii) the input text.

    The input text is automatically embellished with a narrative hook incorporated as the opening line.

    The length of generated paragraph may be capped at a given number of words ("max_length"),
    otherwise the default cap is 100 words.

    Parameters
    ----------
    input_text : str
        The seed text used to generate a paragraph using GPT-2 model.
        It does not need to have a complete sentence.

    max_length  : int
        Maximum number of words of generated paragraph.

    use_narrative_hook : boolean
        Whether to create more dramatic, story-telling impact by adding a randomly-selected narrative hook
        as the opening line (i.e before the input_text) before passing such text collectively into the GPT-2 model.

    Returns
    -------
    paragraph : str
        The paragraph generated using GPT-2 model (inclusive of input text).
    '''
    # Preprocess
    input_text = preprocess(input_text)
    if use_narrative_hook:
        input_text = embellish_text(input_text)

    # Produce
    # return generate_paragraph(input_text, model, max_length=max_length)
    input_text = "This is a story about 3 men sitting on a sofa and discussing about their job. They all seem very confused about their future. \n"  
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
    '''
    Returns a paragraph generated using
    (i) a pre-trained GPT-2 model, and
    (ii) an input text that is incorporated as the opening line.

    The length of generated paragraph may be capped at a given number of words ("max_length"),
    otherwise the default cap is 50 words.

    Parameters
    ----------
    input_text : str
        The seed text used to generate a paragraph using GPT-2 model.
        It does not need to be a complete sentence, but the text must begin properly.

    max_length  : int
        Maximum number of words of generated paragraph.

    Returns
    -------
    paragraph : str
        The paragraph generated using GPT-2 model (inclusive of input text).
    '''
    # Preprocess
    input_text = input_text.capitalize()

    # Produce
    paragraph = model(f"{input_text}", max_length=max_length, num_return_sequences=1)[0]['generated_text']
    paragraph = remove_incomplete_sentence(paragraph)
    paragraph = check_grammar(paragraph)

    return paragraph


## PREPROCESSING


def preprocess(text):
    '''
    Preprocesses input text by
    (i) removing angular brackets (if any),
    (ii) correcting grammar, and
    (iii) capitalising first word.

    Parameters
    ----------
    text : str
        Text to be preprocessed.

    Returns
    -------
    text : str
        Preprocessed text.
    '''
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
    '''
    Returns text with angular brackets and leading and trailing whitespaces removed.

    Parameters
    ----------
    text : str
        Text to be cleaned and have its angular brackets and leading and trailing whitespaces removed.

    Returns
    -------
    text : str
        Cleaned text.
    '''
    regex = re.compile('<.*?>')
    text = re.sub(regex, '', text)
    return text.strip()


def find_verbs(caption):
    '''
    Finds present participle in caption and adds 'is' in front of it.

    Parameters
    ----------
    caption : str
        Caption to be processed.

    Returns
    -------
    caption : str
        The processed caption with 'is' before the present participle.
    '''
    blob = TextBlob(caption)
    for word,tag in blob.tags:
        if tag == 'VBG':
            word_index = caption.find(word)
            # add an 'is' before the present participle found (-ing word)
            new_caption =  caption[:word_index] + 'is ' + caption[word_index:]
            return new_caption
    return caption


def convert_past_tense(caption):
    '''
    Converts present tense into past tense.

    Parameters
    ----------
    caption : str
        Caption that contains present tense (e.g. 'is', 'are') to be processed.

    Returns
    -------
    caption : str
        Caption converted into past tense.
    '''
    caption = re.sub(r'\bis\b', 'was', caption)
    caption = re.sub(r'\bam\b', 'was', caption)
    caption = re.sub(r'\bare\b', 'were', caption)
    return caption


def check_grammar(paragraph):
    '''
    Checks and replaces grammatically incorrect parts of paragraph using grammarbot API
    (https://github.com/GrammarBot-API/grammarbot-py)

    Parameters
    ----------
    paragraph : str
        Paragraph to be processed and have its grammar corrected.

    Returns
    -------
    paragraph : str
        The gramatically correct paragraph processed using grammarbot.
    '''
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
    '''
    Returns the paragraph with its trailing incomplete sentence removed.

    Parameters
    ----------
    paragraph : str
        Paragraph to be processed and have its trailing incomplete sentence removed.

    Returns
    -------
    paragraph : str
        Paragraph with trailing incomplete sentence removed.
    '''
    # Find negative/reverse position index of last sentence-terminating punctuation in paragraph
    last_punctuation_idx = find_last_punctuation_idx(paragraph)

    if last_punctuation_idx in [0,1]:
        return paragraph
    else:
        return paragraph[:last_punctuation_idx]


def find_last_punctuation_idx(paragraph):
    '''
    Identifies and returns the negative/reverse position index of a punctuation symbol (!?,") that is:
    (i) typically indicative of the end of a sentence ("sentence-terminating"), and
    (ii) closest to the end of the paragraph.

    Returns 0 if no such punctuation symbol is present.

    Parameters
    ----------
    paragraph : str
        Paragraph from which the index of the last, sentence-terminating punctuation symbol is to be identified.

    Returns
    -------
    idx : int
        The negative/reverse position index of the last, sentence-terminating punctuation symbol in the given paragraph.
    '''
    # To-do: account for hard cases where sentences ends with single or double quotation marks.
    for idx, char in enumerate(paragraph[::-1]):
        if char in ['!','.','?']:
            return -idx
    return 0


def create_paragraphing_html(text):
    '''
    Takes a paragraph with line breaks in Python ('\n') and returns with line breaks in HTML ('<br>')

    Parameters
    ----------
    text : str
        Paragraph with line breaks in Python

    Returns
    -------
    text : str
        Paragraph with line breaks in html.
    '''
    return text.replace('\n', '<br>')


def random_narrative_hook():
    '''
    Returns a random narrative hook that is dramatic.

    Returns
    -------
    text : str
        A random narrative hook that is dramatic.
    '''
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
    '''
    Embellishes input_text by adding a randomly-selected narrative hook as the opening line

    Parameters
    ----------
    input_text : str
        The text to be embellished with a random opening line incorporated


    Returns
    -------
    text : str
        The embellished text with a random opening line incorporated
    '''
    return random_narrative_hook().capitalize() + '\n\n' + input_text.capitalize()


if __name__ == '__main__':
    model = load_model()
    para = generate_story("He walked by his house. I am ham", model)
    print(para)