import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True
)

text = """Welcome to text-adventure
You're in a forest with a path to the east
> go east
You're in a room with a trophy cabinet and a dog
> """


tokenizedtext = tokenizer(text, return_tensors='pt')
input_ids = tokenizedtext['input_ids']

# Move input_ids to the same device as model
input_ids = input_ids.to(model.device)

generation_output = model.generate(input_ids,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_return_sequences=1,
                                   eos_token_id=tokenizer.eos_token_id,
                                   pad_token_id=tokenizer.eos_token_id,
                                   max_new_tokens=50  # Adjust as needed
                                   )

# create a dict of tokens from 0 to 32000 from the tokenizer where the key is the token and the value is the token id
token_dict = {}
for token in range(32000):
    token_dict[token] = tokenizer.decode(token)

# create a dict of the token ids and the scores from the generation output
token_scores = {}
for i in range(len(generation_output.scores[0].cpu().numpy()[0])):
    token_scores[i] = generation_output.scores[0].cpu().numpy()[0][i]

# save the token dict and the token scores to csv files
pd.DataFrame.from_dict(token_dict, orient='index', columns=['tokens']).to_csv('tokenind.csv', escapechar='\\')
pd.DataFrame.from_dict(token_scores, orient='index', columns=['scores']).to_csv('scoresind.csv', escapechar='\\')
