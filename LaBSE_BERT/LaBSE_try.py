from transformers import *
import torch
from sklearn.preprocessing import normalize

word_dict = {}
# word_df = pd.DataFrame()

# Load Pre-Trained Model
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


def add_to_word_dict(word, pre_sign='none', post_sign='none'):
    """
    Add a new word to the word dictionary.
    @param word: The new word to add.
    @param pre_sign: Is there a sign before the word.
    @param post_sign: Is there a sign after the word.
    """
    if word not in word_dict:
        # Get the vector representing the word from BERT
        word_dict[word] = get_vec(word)

        # df = pd.DataFrame(get_vec(word))
        # df.insert(loc=0, column='Word', value=word)
        # global word_df
        # word_df = pd.concat([word_df, df])

        if pre_sign != 'none' and post_sign != 'none':
            print(f'[{len(word_dict)}]: {word}\t(Without surrounding \'{pre_sign}\')')
        elif pre_sign != 'none':
            print(f'[{len(word_dict)}]: {word}\t(Without pre \'{pre_sign}\')')
        elif post_sign != 'none':
            print(f'[{len(word_dict)}]: {word}\t(Without post \'{post_sign}\')')
        else:
            print(f'[{len(word_dict)}]: {word}')


def check_sign(word):
    """
    Check if there's a sign before or after the word and send it to check_pre_sign/check_post_sign id needed.
    @param word: The new word to add.
    """
    add_to_word_dict(word=word)

    if len(word) > 1:
        if (word[0] == '\"' or word[0] == '\'') and word[0] == word[-1]:
            surrounded_word = word.split(word[0])[1]
            add_to_word_dict(word=surrounded_word)
            check_pre_sign(surrounded_word)
            check_post_sign(surrounded_word)
        elif word[0] == '(' and word[-1] == ')':
            surrounded_word = word.split(word[0])[1]
            surrounded_word = surrounded_word.split(word[-1])[0]
            add_to_word_dict(word=surrounded_word)
            check_pre_sign(surrounded_word)
            check_post_sign(surrounded_word)
        else:
            check_pre_sign(word)
            check_post_sign(word)


def check_pre_sign(word):
    """
    Check if there's a sign before the word and add it to the dictionary.
    @param word: The new word (with a sign before it) to add.
    """
    # check prefix signs
    if word[0] == '(' or word[0] == '\"' or word[0] == '\'':
        w_pre = word.split(word[0])[1]
        check_sign(w_pre)
        if word[0] == word[-1] or (word[0] == '(' and word[-1] == ')'):  # check surround sign
            add_to_word_dict(word=w_pre, pre_sign=word[0], post_sign=word[-1])
        else:
            add_to_word_dict(word=w_pre, pre_sign=word[0])


def check_post_sign(word):
    """
    Check if there's a sign after the word and add it to the dictionary.
    @param word: The new word (with a sign after it) to add.
    """
    # check postfix signs
    if word[-1] == '.' or word[-1] == '!' or word[-1] == '?' or word[-1] == ',' or word[-1] == ';' or word[-1] == ':' \
            or word[-1] == ')' or word[-1] == '\'' or word[-1] == '\"':
        w_post = word.split(word[-1])[0]
        check_sign(w_post)
        add_to_word_dict(word=w_post, post_sign=word[-1])
