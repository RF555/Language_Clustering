import torch
from transformers import BertModel, BertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Load Pre-Trained Model
tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()

if __name__ == '__main__':
    # Test English Sentences

    # Tokenize words/sentences
    # input english sentences to get its tokens to use as input for the model
    english_sentences = [
        "dog",
        "Puppies are nice.",
        "I enjoy taking long walks along the beach with my dog.",
    ]

    # Tokenize `english_sentences` with the BERT tokenizer.
    english_token_ids = tokenizer(english_sentences, return_tensors="pt", padding=True)
    print("Tokenized english_token_ids:\n")
    print(english_token_ids)

    print("\nTEST:")
    print(english_token_ids['input_ids'][0])

    english_tokens = tokenizer.convert_ids_to_tokens(english_token_ids['input_ids'][0])
    print("\nTokenized english_tokens:\n")
    print(english_tokens)

    with torch.no_grad():
        english_outputs = model(**english_token_ids)

    english_embeddings = english_outputs.pooler_output

    print("\nSentence Embeddings:\n")
    print(english_embeddings)
