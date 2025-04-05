import tiktoken

# just a wrapper currently for possible changes
def create_tokenizer():
    tokenizer = tiktoken.get_encoding('cl100k_base')
    return tokenizer
