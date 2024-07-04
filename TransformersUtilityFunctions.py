from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize

tokenizer = AutoTokenizer.from_pretrained("setu4993/LaBSE", do_lower_case=False)
model = AutoModel.from_pretrained("setu4993/LaBSE")

def get_vec(text_input):
    """
    Get vector embeddings for the given word or sentence (NOT sentences list).
    @param text_input: The text to be embedded.
    @return: The embedded vectors representing the input.
    """
    # max_seq_length = 64

    # Convert to a list
    english_sentences = [
        text_input]  # ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]

    tokenized_results = tokenizer(english_sentences, add_special_tokens=True,
                                  padding='max_length')  # , max_length=max_seq_length)
    input_ids = tokenized_results['input_ids']
    token_type_ids = tokenized_results['token_type_ids']
    attention_mask = tokenized_results['attention_mask']

    device = torch.device('cpu')
    # device=torch.device('cuda') # uncomment if using gpu
    input_ids = torch.tensor(input_ids).to(device)
    token_type_ids = torch.tensor(token_type_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    curr_model = model.to(device)

    output = curr_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    english_embeddings = output[1].cpu().detach().numpy()
    english_embeddings = normalize(english_embeddings)

    return english_embeddings
