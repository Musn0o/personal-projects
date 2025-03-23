import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import spacy

# Download necessary NLTK resources
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("wordnet")

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Predefined responses mapped to keyword phrases
responses = {
    "I don't have an age, but I was coded recently!": [
        "created",
        "made",
        "born",
        "programmed",
        "age",
        "how old",
        "birth",
        "birthday",
    ],
    "I am a chatbot created by Scar!": ["name", "identity"],
    "Hi there! How can I help you?": ["hello", "hi", "hey", "morning", "evening"],
    "I'm just here, ready to chat!": ["what's up", "how you doing", "how do you do"],
    "I was created by Scar and ChatGPT as a part of practice": [
        "exist",
        "purpose",
        "meaning",
        "reason",
    ],
}


def extract_keywords(sentence):
    """Extracts important words (nouns, adjectives, or verbs) from a sentence."""
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)  # Tokenize sentence into words
    tagged_words = pos_tag(words)  # POS tagging

    important_words = []
    for word, tag in tagged_words:
        if word == "'s":
            important_words.append("is")
        elif tag.startswith(("N", "J", "V")):
            lemmatized_word = lemmatizer.lemmatize(word.lower())  # Normalize
            important_words.append(lemmatized_word)

    return important_words


def get_synonyms(word):
    """Finds synonyms for a given word using WordNet."""
    synonyms = set()
    for synset in wordnet.synsets(word, pos=wordnet.NOUN):  # Only get NOUN synonyms
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Replace underscores

    return synonyms


def get_response(user_input):
    """Finds the most relevant chatbot response based on user input."""
    user_words = set(word_tokenize(user_input.lower()))  # Convert input to word set

    for response, phrases in responses.items():
        for phrase in phrases:
            phrase_words = set(word_tokenize(phrase))  # Convert phrase to word set

            if (
                phrase_words <= user_words
            ):  # Check if all words in phrase exist in input
                return response

    return "Sorry, I don't understand that."


# Main chatbot loop
if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip().lower()
        if user_input in ["exit", "quit"]:
            break
        print("Chatbot:", get_response(user_input))
