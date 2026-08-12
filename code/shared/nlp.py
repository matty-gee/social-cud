import re, json
from collections import Counter
import numpy as np
import pandas as pd
import threading  # For thread-safe model loading

import string
import textstat
import tiktoken
from wordcloud import WordCloud
from textblob import TextBlob

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import torch
import torch.nn.functional as F
import anthropic
from openai import OpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfFolder
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM, set_seed, AutoModelForSequenceClassification


# from sentence_transformers import SentenceTransformer
# from huggingface_hub import HfFolder
# import google.generativeai as genai


#----------------------------------------------------------------------------------------
# API keys for various LLMs
#----------------------------------------------------------------------------------------


# llm_keys = json.load(open("api-keys.json"))
from pathlib import Path
llm_keys = json.loads((Path(__file__).parent / "api-keys.json").read_text())
GOOGLE_KEY = llm_keys["GOOGLE_KEY"] # gemini
OPENAI_KEY = llm_keys["OPENAI_KEY"] # gpt
ANTHROPIC_KEY = llm_keys["ANTHROPIC_KEY"] # claude
HUGGINGFACE_KEY = llm_keys["HUGGINGFACE_KEY"] # llama3

# from openai import OpenAI
# genai.configure(api_key=GOOGLE_KEY)
# HfFolder.save_token(HUGGINGFACE_KEY)
# from huggingface_hub import HfFolder

#----------------------------------------------------------------------------------------
# Model Cache
#----------------------------------------------------------------------------------------

_model_cache = {}
_model_lock = threading.Lock()  # To ensure thread safety if needed

#----------------------------------------------------------------------------------------
# preprocessing functions
#----------------------------------------------------------------------------------------

def read_txt(txt_file):
    with open(txt_file, 'r') as f:
        return f.read()

def lowercase(word):
    return word.lower()

def remove_punctuation(words):
    return words.translate(str.maketrans('', '', string.punctuation))

def get_my_pos(word):
    # specify the pos for words in this datatset
    to_check = {'felt': 'v'}
    return to_check.get(word, None)

def get_nltk_pos(word_list):
    """Map nltk POS tags to wordnet POS tags for a word list."""
    mapping = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV, 'J': wordnet.ADJ}
    pos_tagged = nltk.pos_tag(word_list)
    return [mapping.get(pos[1][0], wordnet.NOUN) for pos in pos_tagged]

def infrequent_filter(words, min_freq=2):
    # remove words that appear less than min_freq times
    word_freq = Counter(words)
    return [word for word in words if word_freq[word] >= min_freq]

def stopwords_filter(words):
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

def remove_punctuation(words):
    return words.translate(str.maketrans('', '', string.punctuation))

def lemmatize(word, pos=None):

    # check if the word is in the exceptions dictionary for the given POS; otherwise, lemmatize the word normally
    # https://stackoverflow.com/questions/33594721/why-nltk-lemmatization-has-wrong-output-even-if-verb-exc-has-added-right-value

    exclude = ['boss']
    if word in exclude:
        return word
    
    # POS tag the word if not provided
    if pos is None: 
        pos = get_my_pos(word) or get_nltk_pos([word])[0]

    # check the morphy exceptions
    exceptions = wordnet._exception_map[pos]
    if word in exceptions:
        lemmatized = exceptions[word][0]
    else:
        lemmatizer = WordNetLemmatizer()
        lemmatized = lemmatizer.lemmatize(word, pos)

    # # do they match?
    # if (verbose) & (lemmative_morphy != lemmatize_wn):
    #     print(f"Mismatch - Morphy: {lemmative_morphy}, WordNet: {lemmatize_wn}")

    return lemmatized

def remove_words(word_list, words_to_remove):
    """
    Removes specified words and their possessive forms from a list of words or a string.

    Parameters:
    - word_list (list or str): The list of words or string from which to remove words.
    - words_to_remove (list): List of words to remove.

    Returns:
    - list or str: The words after removal, in the same format as the input.
    """
    
    # validate input
    if not isinstance(word_list, (list, str)) or not isinstance(words_to_remove, list):
        raise ValueError("Invalid input type")
    is_str = isinstance(word_list, str)
    if is_str: 
        word_list = word_list.split()

    # words to remove
    words_to_remove = [re.escape(w.lower()) for w in words_to_remove] +\
                      [re.escape(f"{w.lower()}'s") for w in words_to_remove] +\
                      [re.escape(f"{w.lower()}'") for w in words_to_remove] +\
                      [re.escape(f"{w.lower()}'ll") for w in words_to_remove]
    # remove the words
    filtered_words = [w for w in word_list if w.lower() not in words_to_remove]

    # return in the original format
    return ' '.join(filtered_words) if is_str else filtered_words

def remove_phrases(text, phrases_to_remove):
    """
    Removes specified phrases from a list of words or a string.

    Parameters:
    - text (list or str): The list of words or string from which to remove phrases.
    - phrases_to_remove (list): List of phrases to remove.

    Returns:
    - list or str: The text after removal, in the same format as the input.
    """
    
    # validate input
    if not isinstance(text, (list, str)) or not isinstance(phrases_to_remove, list):
        raise ValueError("Invalid input type")
    
    is_str = isinstance(text, str)
    if not is_str: 
        text = ' '.join(text)

    # Sort phrases by length in descending order to avoid partial matches
    phrases_to_remove = sorted(phrases_to_remove, key=len, reverse=True)
    phrases_to_remove = [re.escape(phrase.lower()) for phrase in phrases_to_remove]

    # Create a combined regex pattern for all phrases to remove
    pattern = re.compile(r'\b(?:' + '|'.join(phrases_to_remove) + r')\b', re.IGNORECASE)

    # Remove the phrases
    filtered_text = pattern.sub('', text)

    # Clean up any extra spaces left after removal
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()

    # Return in the original format
    return filtered_text.split() if not is_str else filtered_text

def preprocess_text(text, exclude_list=None, remove_stopwords=True, min_freq=0, return_tokenized=True):
    # generic preprocessing for NLP tasks
    # expects text in a string format; if its a list, access or join the list of words
    if isinstance(text, list):
        text = text[0] if len(text) == 1 else ' '.join(text)
    text   = remove_words(text, exclude_list) if exclude_list else text # remove specific words
    text   = [lowercase(remove_punctuation(word)) for word in [text]][0] # lowercasee & remove punctuation before tokenizing
    tokens = word_tokenize(text) # tokenize before preprocessing
    tokens = stopwords_filter(tokens) if remove_stopwords else tokens # remove stop words
    tokens = infrequent_filter(tokens) if (min_freq > 1) else tokens # remove infrequent words
    tokens = [lemmatize(word) for word in tokens]  # lemmatize (e.g. running -> run)
    return tokens if return_tokenized else " ".join(tokens) # output as tokenized or joined

def pad_tokens(tokens):
    # Pad input tokens
    max_len = 0
    for i in tokens:
        if len(i) > max_len:
            max_len = len(i)
    return np.array([i + [0]*(max_len-len(i)) for i in tokens])

def make_attention_mask(padded):
    # Create attention masks
    return np.where(padded != 0, 1, 0)

def build_vocab(sentences, verbose=True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


#----------------------------------------------------------------------------------------
# word clouds
#----------------------------------------------------------------------------------------


def preprocess_text_for_wordcloud(text, 
                                  remove_stopwords=True, 
                                  min_word_freq=2,
                                  remove_specialwords=None, 
                                  remove_specialphrases=None):

    # remove punctuation & tokenize
    # text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # word_tokens = word_tokenize(text) # screwing up detecting contractions
    word_tokens = text.lower().split()

    # remove words
    preprocessed_text = stopwords_filter(word_tokens) if remove_stopwords else word_tokens # stopwords
    preprocessed_text = infrequent_filter(preprocessed_text) if (min_word_freq > 1) else preprocessed_text # infrequent words
    preprocessed_text = remove_words(preprocessed_text, remove_specialwords) if remove_specialwords else preprocessed_text # specific words
    preprocessed_text = remove_phrases(preprocessed_text, remove_specialphrases) if remove_specialphrases else preprocessed_text # specific phrases

    # lemmatize (e.g. running -> run)
    preprocessed_text = [lemmatize(word) for word in preprocessed_text] 

    return " ".join(preprocessed_text)

def filter_wordcloud(wordcloud_dict, threshold):
    """
    Filter a wordcloud dictionary by removing words with a frequency below the given threshold.

    :param wordcloud_dict: The word frequency dictionary to be filtered.
    :param threshold: The frequency threshold. Words with a frequency below this value will be excluded.
    :return: A filtered word frequency dictionary.
    """
    return WordCloud(background_color="white").generate_from_frequencies({word: freq for word, freq in wordcloud_dict.items() if freq >= threshold})


#----------------------------------------------------------------------------------------
# Semantic embeddings
#----------------------------------------------------------------------------------------


class GPT2:
    # GPT-2 models
    # gpt2: 12-layer, 768-hidden, 12-heads, 117M parameters
    # gpt2-medium: 24-layer, 1024-hidden, 16-heads, 345M parameters
    # gpt2-large: 36-layer, 1280-hidden, 20-heads, 774M parameters
    # gpt2-xl: 48-layer, 1600-hidden, 25-heads, 1558M parameters

    def __init__(self,
                 gpt_model='gpt2',
                 padding=True, 
                 truncation=True, 
                 return_tensors="pt", 
                 is_split_into_words=False, 
                 add_prefix_space=True,
                 which_tokens=None, 
                 which_layers=None,
                 which_pooling=None, 
                 remove_padding=False,
                 verbose=False):
    
        set_seed(23)
        
        # embedding parameters
        self.which_tokens = which_tokens
        self.which_layers = which_layers
        self.which_pooling = which_pooling
        self.remove_padding = remove_padding # if want to return vector embeddings w/o padding altogehter
        self.verbose = verbose

        # tokenizer parameters
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.is_split_into_words = is_split_into_words
        self.add_prefix_space = add_prefix_space

        # gpt2 model parameters (specific to gpt2 version)
        self.gpt_model = gpt_model
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)
        self.model = GPT2LMHeadModel.from_pretrained(gpt_model, output_hidden_states=True)
        self.model.trainable = False
        self.generator = pipeline('text-generation', model=gpt_model)
        
        self.n_layers = self.model.config.n_layer + 1 # number of hidden layers: attention layers + 1 linear layer
        self.n_embd   = self.model.config.n_embd # number of embedding dimensions in each layer 

        if self.tokenizer.pad_token is None:
            # https://github.com/huggingface/transformers/issues/8452
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def tokenize(self, text_sequences):

        if isinstance(text_sequences, str):
            text_sequences = [text_sequences]
        
        tokenized = self.tokenizer.batch_encode_plus(text_sequences,
                                                     padding=self.padding, # pad all sequences to be the same length
                                                     truncation=self.truncation, # truncate sequences that are too long
                                                     return_tensors=self.return_tensors, # return PyTorch tensors
                                                     is_split_into_words=self.is_split_into_words, # is it already split into words?
                                                     add_prefix_space=self.add_prefix_space) # add space before each sequence
        self.input_ids, self.attention_mask = tokenized['input_ids'], tokenized['attention_mask'] # '50256' is padding token
        self.n_seqs, self.n_tokens = self.input_ids.shape # number of sequences & number of tokens in the batch
        self.seg_lens = self.attention_mask.sum(dim=1) # the lengths of the different sequences

    def forward_pass(self):
        self.model.eval() # feed-forward only
        with torch.no_grad(): # no gradients
            outputs = self.model(self.input_ids, attention_mask=self.attention_mask)
        self.logits = outputs.logits.squeeze()
        self.hidden_layers = torch.stack(outputs.hidden_states, dim=0).permute(1,2,0,3)
        assert self.hidden_layers.size() == (self.n_seqs, self.n_tokens, self.n_layers, self.n_embd)
        if self.verbose: print(f"Hidden layers shape: {self.hidden_layers.size()}")
        
    # maybe add the optional arguments here so I can tokenize and forward pass and then play around with outputting diff. embeddings without re-running
    def get_embeddings(self):

        # select specific tokens &/or layers [optional]
        if self.which_tokens is not None:
            # important: each sequence can be of diff. length, so make sure to get the correct token
            if isinstance(self.which_tokens, int):
                if self.which_tokens >= 0:
                    raise ValueError(f"'which_tokens' must be negative")  
                self.hidden_layers  = torch.stack([self.hidden_layers[i, len + self.which_tokens] for i, len in enumerate(self.seg_lens)]).unsqueeze(1) # maintain 4D tensor
                self.attention_mask = torch.ones((self.n_seqs, 1)) # recreate attention mask as 2D tensor of size (n_seqs, 1)
            else:
                raise NotImplementedError(f"which_tokens={self.which_tokens}")
            
        if self.which_layers is not None:
            if isinstance(self.which_layers, int):
                self.hidden_layers = self.hidden_layers[:,:,self.which_layers,:].unsqueeze(2)
            else:
                self.hidden_layers = self.hidden_layers[:,:,self.which_layers[0]:self.which_layers[1],:]
        
        if self.verbose: print(f"Extracted layers shape: {self.hidden_layers.size()}")

        # pooling to combine layers [optional]
        if self.which_pooling is None:
            self.token_embeddings = self.hidden_layers
        elif self.which_pooling == 'concat':
            self.token_embeddings = self.hidden_layers.reshape(self.n_seqs, self.n_tokens, -1)
        elif self.which_pooling == 'sum':
            self.token_embeddings = self.hidden_layers.sum(dim=2)
        elif self.which_pooling == 'mean':
            self.token_embeddings = self.hidden_layers.mean(dim=2)
        else:
            raise ValueError("Invalid 'which_pooling' value. It must be None, 'concat', 'sum', or 'mean'.")

        # exclude padding tokens from semantic embeddings
        attention_mask = self.attention_mask.bool().unsqueeze(-1)
        if len(self.token_embeddings.size()) == 4:
            attention_mask = attention_mask.unsqueeze(-1) 
        if self.verbose: print(f"Attention mask shape: {attention_mask.size()}")
        self.token_embeddings = (self.token_embeddings * attention_mask).squeeze() # zero-out padded tokens
        if self.token_embeddings.dim() == 2:
            self.token_embeddings = self.token_embeddings.unsqueeze(0)

        # calculate sequence embeddings: the average over the token embeddings
        self.sequence_embeddings = (self.token_embeddings.sum(dim=1) / attention_mask.sum(dim=1).float()).squeeze()   

        # remove padded tokens (do after averaging)
        if self.remove_padding:
            n_incl_tokens = attention_mask.sum(dim=1) # number of tokens in each sequence
            depadded_token_embeddings = []
            for seq in np.arange(self.n_seqs): # for each sequence, get included tokens
                depadded_token_embeddings.append(self.token_embeddings[seq][:n_incl_tokens[seq]])
            self.token_embeddings = depadded_token_embeddings
    
        return {'token': self.token_embeddings, 'sequence': self.sequence_embeddings}

    def get_predictions(self):
        # use logits to predict next token at each token
        self.predicted_ids, self.predicted_tokens = [], []
        for logits in self.logits:
            predicted_id = torch.argmax(logits).item()
            self.predicted_ids.append(predicted_id)
            self.predicted_tokens.append(self.tokenizer.decode(predicted_id))
        self.predicted_text = ('').join(self.predicted_tokens)
        return self.predicted_text
    
    # add a prediction error method...
     
    def generate_text(self,
                      text_prompt,
                      max_length=25,
                      num_return_sequences=5,
                      repetition_penalty=1.5,
                      method='greedy',
                      num_beams=5, # for beam search
                      temperature=0.5, # higher for more 'creativity'
                      top_k=10, # for top-k sampling
                      top_p=0.85): # for top-p sampling
        
        # TODO - figure out the warning about token id and attention mask
        # encode the prompt as input ids
        input_ids = self.tokenizer.encode(text_prompt, return_tensors='pt')

        # use one of different methods of generating
        with torch.no_grad():
            if method == 'greedy':
                output_ids = self.model.generate(input_ids,
                                                 max_length=max_length,
                                                 num_return_sequences=num_return_sequences,
                                                 repetition_penalty=repetition_penalty,
                                                 do_sample=True)      
            elif method == 'beam':
                output_ids = self.model.generate(input_ids,
                                            max_length=max_length, 
                                            num_beams=num_beams, 
                                            no_repeat_ngram_size=2, 
                                            early_stopping=True,
                                            repetition_penalty=repetition_penalty,
                                            num_return_sequences=num_return_sequences)
            elif method == 'sampling':
                output = self.model.generate(input_ids,
                                            max_length=max_length,
                                            return_dict_in_generate=True, 
                                            output_scores=True,
                                            do_sample=True,
                                            temperature=temperature,
                                            repetition_penalty=repetition_penalty,
                                            num_return_sequences=num_return_sequences) # increases chance of high probability words
            elif output_ids == 'top_sampling':
                output_ids = self.model.generate(input_ids,
                                            return_dict_in_generate=True, 
                                            output_scores=True,
                                            max_length=max_length,
                                            do_sample=True,
                                            repetition_penalty=repetition_penalty,
                                            top_k=top_k, 
                                            top_p=top_p)

def get_gpt2_embeddings(text, gpt_model='gpt2', **kwargs):
    gpt2 = GPT2(gpt_model=gpt_model, **kwargs)
    gpt2.tokenize(text)
    gpt2.forward_pass()
    return gpt2.get_embeddings()

def get_sentence2vec_embeddings(sentences, normalize=False):
    # Function to convert sentence to vector using Word2Vec model
    # averages the word2vec embeddings for the different words
    # returns normalized vector embeddings

    # load model if necessary
    global w2v_model
    if 'w2v_model' not in globals():
        print("Loading Word2Vec model...")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/matty_gee/Desktop/projects/code/LLMs/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
        globals()['w2v_model'] = w2v_model
       
    embeddings = []
    for sentence in sentences:
        tokens = sentence.split()
        tokens = [t for t in tokens if t in w2v_model.key_to_index] # only keep tokens that are in vocab
        avg_vector = np.mean(w2v_model[tokens], axis=0) # or w2v_model.wv[sentence]
        if normalize: 
            avg_vector = avg_vector / np.linalg.norm(avg_vector) # normalize the vector
        embeddings.append(avg_vector)
    return np.array(embeddings)

def get_openai_embeddings(sentences, model="ada"):
    """
        Retrieve semantic embeddings for a list of sentences using OpenAI API.
        
        Args:
        sentences (list): List of sentences to embed.
        model (str): Name of the OpenAI model to use for embeddings.
                    Core models:
                    - small: "text-embedding-3-small" (1536 dimensions)
                    - large: "text-embedding-3-large" (3072 dimensions)
                    - ada: "text-embedding-ada-002" (1536 dimensions, older model)
        
        Returns:
        list: List of embedding vectors for each sentence.
    """

    model_dict = {'ada': 'text-embedding-ada-002', 
                  'small': 'text-embedding-3-small',
                  'large': 'text-embedding-3-large'}
    model = model_dict[model]

    client = OpenAI(api_key=OPENAI_KEY)
    
    embeddings = []
    try:
        response = client.embeddings.create(
            input=sentences,
            model=model
        )
        embeddings = [data.embedding for data in response.data]
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    return np.array(embeddings)

def get_llama_embeddings(sentences, normalize=False, layer='last'):
    """
        Get sentence embeddings using the Meta-Llama3 8B model
    """

    set_seed(2024)
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, load_in_8bit=False)
    # config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, use_safetensors=False)
    # model = AutoModelForCausalLM.from_config(config)

    # tokenize the input text (returns a dictionary with input_ids & attention_mask)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    n_seqs = inputs['input_ids'].shape[0] # batch_size: number of sequences (probably sentences)
    n_toks = inputs['input_ids'].shape[1] # all sequences will be left padded to have same number of tokens
    attention_masks = inputs['attention_mask'] # attention masks for each sequence with left padding


    # run inference (outputs: logits, past key values, hidden states)
    with torch.no_grad(): # dont compute gradients
        outputs = model(**inputs)

    # get the hidden states: internal representation of the model at each layer
    hidden_states   = outputs.hidden_states # shape = (num_layers: 33, batch_size, num_tokens, hidden_size: 4096)
    n_hidden_states = len(hidden_states)
    
    # print(f'Number of sequences = {n_seqs}')
    # print(f'Number of tokens = {n_toks}')
    # print(f'Number of hidden states = {n_hidden_states}')
    # print(f'Dimensions in each hidden state = {hidden_states[0].shape[2]}') 

    def pool_hidden_states(hidden_states, attention_mask):

        # Assuming you have your hidden_states and attention_mask
        # hidden_states: Tuple[Tensor(n_sequences, n_tokens, 4096)] * 33
        # attention_mask: Tensor(n_sequences, n_tokens)

        pooled_embeddings = []
        for layer_output in hidden_states:

            # Expand attention_mask to match the dimensions of layer_output
            expanded_mask = attention_mask.unsqueeze(-1).expand(layer_output.size()).float()
            
            # Apply the mask and calculate the sum
            sum_embeddings = torch.sum(layer_output * expanded_mask, dim=1)
            
            # Calculate the average, avoiding division by zero
            avg_embeddings = sum_embeddings / torch.clamp(expanded_mask.sum(1), min=1e-9)
            
            pooled_embeddings.append(avg_embeddings)
        return torch.stack(pooled_embeddings)

    sentence_embeddings = pool_hidden_states(hidden_states, attention_masks)

    if normalize:
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    if layer == 'last':
        return sentence_embeddings[-1].numpy()
    elif layer == 'all':
        return sentence_embeddings.numpy()
    elif layer in ['mean', 'average']:
        return sentence_embeddings.mean(dim=0).numpy()

def get_google_embeddings(sentences, model="text-embedding-004", task_type="similarity"):
    """
    Retrieve semantic embeddings for a list of sentences using Google's embedding models
    
    Args:
    sentences (list): List of sentences to embed.
    model (str): Name of the model to use for embeddings.
                 Default is "text-embedding-004".
                 - "text-embedding-004": A general-purpose embedding model for text.
    task_type (str): Type of task for the embeddings.
                     Default is "similarity".
                     - "similarity": Embeddings optimized for similarity tasks.
                     - "classification": Embeddings optimized for classification tasks.
    
    Returns:
    np.array: Array of embedding vectors for each sentence.
    """

    # Map the input model to ensure compatibility or future extensions
    model_dict = {
        'text-embedding-004': 'models/text-embedding-004'
    }
    model = model_dict.get(model, model)

    embeddings = []
    try:
        response = genai.embed_content(
            model=model,
            content=sentences,
            task_type=task_type
        )
        embeddings = response['embedding']
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    return np.array(embeddings)

def get_sentencetransformer_embeddings(sentences, model='mpnet', normalize=False):
    ''' 
        function to return sentence embeddings using the sentence-transformers library
        - sentences (list/array): sentences to embed
        - model (str): name of the model to use
        - normalize (bool): whether to unit normalize the embeddings or not
        
        See the following link for top performers in sentence embedding: 
            https://www.sbert.net/docs/pretrained_models.html
    '''

    set_seed(2024)
    model_dict = {'qwen3': "Qwen/Qwen3-Embedding-0.6B",
                  'mpnet': 'sentence-transformers/all-mpnet-base-v2', 
                  'roberta': 'sentence-transformers/all-distilroberta-v1',
                  'mlml12': 'sentence-transformers/all-MiniLM-L12-v2',
                  'mlml6': 'sentence-transformers/all-MiniLM-L6-v2',
                  'multi_qa': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
                  'mpnet_negation': 'dmlls/all-mpnet-base-v2-negation'}
    if model in model_dict: model = model_dict[model]
    
    # if wanna use the wrapper direct;y - but returns normalized by default
    # model = SentenceTransformer(model)
    # model.encode(sentences)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    model     = AutoModel.from_pretrained(model)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    attention_mask = encoded_input['attention_mask']
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling to get sentence embeddings
    token_embeddings    = model_output.last_hidden_state # all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # Create an attention mask for padding tokens
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) # average the embeddings of all tokens in the sentence

    if normalize:
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return np.array(sentence_embeddings)

def get_sentence_embeddings(sentences, normalize=True, model="roberta"):
    """
    Wrapper function to get sentence embeddings using different models.

    Args:
    sentences (list): List of sentences to embed.
    model (str): Name of the model to use

    Returns:
    numpy.ndarray: Array of embedding vectors for each sentence.
    """

    # ensure sentences is a list
    if not isinstance(sentences, list):
        sentences = list(sentences)

    model = model.lower()

    if model == "w2v":
        return get_sentence2vec_embeddings(sentences, normalize=normalize)

    elif model == "openai":
        return get_openai_embeddings(sentences, model="small")
    
    elif model == 'google':
        return get_google_embeddings(sentences)
    
    elif model.startswith("llama"):
        return get_llama_embeddings(sentences, normalize=normalize, layer='last')
    
    else: # try sentence-transformers
        return get_sentencetransformer_embeddings(sentences, normalize=normalize, model=model)
    

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

def text_in_context(list_of_text, window):
    context = []
    for i in range(len(list_of_text)):
        start = max(0, i - window + 1)
        context.append((" ").join(list_of_text[start:i + 1]))
    return context


#----------------------------------------------------------------------------------------
# Sentiment analysis
#----------------------------------------------------------------------------------------

def calculate_compound_score(positive, negative, neutral):
    total = positive + negative + neutral
    positive_norm = positive / total
    negative_norm = negative / total
    neutral_norm = neutral / total
    compound_score = (positive_norm - negative_norm) * (1 - neutral_norm)
    return round(compound_score, 3)

_model_cache = {}
_model_lock = threading.Lock()
def estimate_sentiment(text):
    """
    Estimate sentiment and emotions for given text(s).

    Parameters:
        text (str or list of str): The input text or list of texts to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing the original text and sentiment/emotion scores.
    """

    # Ensure input is a list
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        if not all(isinstance(t, str) for t in text):
            raise ValueError("All items in the input list must be strings.")
        texts = text
    else:
        raise ValueError("Input must be a string or a list of strings.")

    #------------------------------------------------------#
    # Load and Cache NLTK SentimentIntensityAnalyzer
    #------------------------------------------------------#
    if 'sentiment_nltk_model' not in _model_cache:
        with _model_lock:
            if 'sentiment_nltk_model' not in _model_cache:
                _model_cache['sentiment_nltk_model'] = SentimentIntensityAnalyzer()
    sentiment_nltk_model = _model_cache['sentiment_nltk_model']

    #------------------------------------------------------#
    # Load and Cache Hugging Face BERT Sentiment Model
    #------------------------------------------------------#
    if 'sentiment_bert_model' not in _model_cache:
        with _model_lock:
            if 'sentiment_bert_model' not in _model_cache:
                _model_cache['sentiment_bert_model'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment",
                    return_all_scores=True
                )
    sentiment_bert_model = _model_cache['sentiment_bert_model']

    #------------------------------------------------------#
    # Load and Cache Hugging Face Emotion Model
    #------------------------------------------------------#
    if 'emotion_bert_model' not in _model_cache:
        with _model_lock:
            if 'emotion_bert_model' not in _model_cache:
                _model_cache['emotion_bert_model'] = pipeline(
                    "text-classification",
                    model="bhadresh-savani/distilbert-base-uncased-emotion",
                    return_all_scores=True
                )
    emotion_bert_model = _model_cache['emotion_bert_model']

    #------------------------------------------------------#
    # Process NLTK Sentiment for Each Text
    #------------------------------------------------------#
    sentiment_nltk_list = [sentiment_nltk_model.polarity_scores(t) for t in texts]
    sentiment_nltk_processed = [
        {
            'nltk_negativity': round(s['neg'], 3),
            'nltk_neutrality': round(s['neu'], 3),
            'nltk_positivity': round(s['pos'], 3),
            'nltk_compound': round(s['compound'], 3)
        }
        for s in sentiment_nltk_list
    ]

    #------------------------------------------------------#
    # Process BERT Sentiment (Batch Processing)
    #------------------------------------------------------#
    sentiment_bert_results = sentiment_bert_model(texts)
    sentiment_bert_processed = []
    for sentiment in sentiment_bert_results:
        # Create a dictionary of label scores
        sentiment_dict = {d['label']: round(d['score'], 3) for d in sentiment}
        # Calculate compound score
        compound = calculate_compound_score(
            positive=sentiment_dict.get('LABEL_2', 0),
            negative=sentiment_dict.get('LABEL_0', 0),
            neutral=sentiment_dict.get('LABEL_1', 0)
        )
        sentiment_bert_processed.append({
            'bert_negativity': sentiment_dict.get('LABEL_0', 0),
            'bert_neutrality': sentiment_dict.get('LABEL_1', 0),
            'bert_positivity': sentiment_dict.get('LABEL_2', 0),
            'bert_compound': compound
        })

    #------------------------------------------------------#
    # Process Emotion Analysis (Batch Processing)
    #------------------------------------------------------#
    emotion_results = emotion_bert_model(texts)
    emotion_processed = []
    for emotions in emotion_results:
        emotion_dict = {emotion['label']: round(emotion['score'], 3) for emotion in emotions}
        emotion_processed.append(emotion_dict)

    #------------------------------------------------------#
    # Combine All Results into a DataFrame
    #------------------------------------------------------#
    combined_data = []
    for i in range(len(texts)):
        row = {
            'text': texts[i],
            **sentiment_nltk_processed[i],
            **sentiment_bert_processed[i],
            **emotion_processed[i]
        }
        combined_data.append(row)

    df = pd.DataFrame(combined_data)

    return df

def calculate_readability(text):
    return textstat.flesch_kincaid_grade(text)

#----------------------------------------------------------------------------------------
# Named Entity Recognition (NER)
#----------------------------------------------------------------------------------------


# ner_pipeline = pipeline('ner', aggregation_strategy='simple')

def run_named_entity_recognition(text):
    # add one of those progress bars things...
    sentences = nltk.tokenize.sent_tokenize(text) # split into sentences; can do some minimal preprocessing too
    dfs = []
    for s, sentence in enumerate(sentences):
        df = pd.DataFrame(ner_pipeline(sentence))
        df['sentence'] = s+1
        dfs.append(df)
    return pd.concat(dfs)