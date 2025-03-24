from sentence_transformers import SentenceTransformer

# Load a pre-trained SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example sentence to encode
sentence = "Hello, how can I help you?"

# Convert text into an embedding
embedding = sbert_model.encode(sentence)

print(embedding)  # This prints the vector representation
