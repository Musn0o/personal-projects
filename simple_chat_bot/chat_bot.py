import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
import spacy
from nltk.stem import WordNetLemmatizer


nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt_tab")
nltk.download("wordnet")


def extract_keywords(sentence):
    """Extracts important words (nouns, adjectives, or verbs) from a sentence"""
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)  # Tokenize sentence into words
    tagged_words = pos_tag(words)  # POS tagging
    # print(tagged_words)
    important_words = []
    for word, tag in tagged_words:
        if word == "'s":
            important_words.append("is")
        elif tag.startswith(("N", "J", "V")):
            lemmatized_word = lemmatizer.lemmatize(word.lower())  # Normalize
            important_words.append(lemmatized_word)

    # print(important_words)
    return important_words


existence = {}

responses = {
    "I don't have an age, but I was coded recently!": (
        "created",
        "made",
        "born",
        "programmed",
        "age",
        "how old",
        "birth",
        "birthday",
    ),
    "I am a chatbot created by Scar!": ("name", "identity"),
    "Hi there! How can I help you?": ("hello", "hi", "hey", "morning", "evening"),
    "I'm just here, ready to chat!": (
        "what's up",
        "how you doing",
        "how do you do",
    ),
    "I was created by Scar and ChatGPT as a part of practice": (
        "exist",
        "purpose",
        "meaning",
        "reason",
    ),
}

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")


def get_synonyms(word):
    synonyms = set()

    for synset in wordnet.synsets(word, pos=wordnet.NOUN):  # Only get NOUN synonyms
        for lemma in synset.lemmas():
            synonyms.add(
                lemma.name().replace("_", " ")
            )  # Replace underscores in multi-word synonyms

    return synonyms


def extract_nouns(sentence):
    doc = nlp(sentence)
    return [
        token.lemma_ for token in doc if token.pos_ == "NOUN"
    ]  # Keep lemmatized nouns


def get_response(user_input):
    user_words = set(
        word_tokenize(user_input.lower())
    )  # Convert input to a set of words

    for response, phrases in responses.items():
        for phrase in phrases:
            phrase_word = set(word_tokenize(phrase))  # Convert each phrase to a set

            if (
                phrase_word <= user_words
            ):  # Checks if all words in phrase exist in input
                return response  # Return response if at least one word matches

    return "Sorry, I don't understand that."


def chatbot(user_input):
    """Basic chatbot function with synonym matching."""

    # Preprocess user input
    doc = nlp(user_input.lower())
    for token in doc:
        # Check if the token or any synonym is in responses
        for key in responses:
            if token.text == key or token.text in get_synonyms(key):
                return responses[key]

    return "Sorry, I don't understand that."


# Example usage
while True:
    user_input = input("You: ").strip().lower()
    if user_input.lower() in ["exit", "quit"]:
        break
    print("Chatbot:", get_response(user_input))

if __name__ == "__main__":
    chatbot()
