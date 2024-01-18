import torch
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import wandb
wandb.init()

base_model_id = "mistralai/Mistral-7B-v0.1"

ppoconfig = PPOConfig(
    model_name=base_model_id,
    learning_rate=1.41e-5,
    log_with="wandb",
)

loraconfig = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppoconfig.model_name,
    load_in_4bit=True,
    peft_config=loraconfig,
)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppoconfig.model_name,
    load_in_4bit=True,
    peft_config=loraconfig,
)

tokenizer = AutoTokenizer.from_pretrained(ppoconfig.model_name)
tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(ppoconfig, model, ref_model, tokenizer)

device = ppo_trainer.accelerator.device

def listofstringstolistofgpustringtensors(listofstrings):
    return [torch.tensor(tokenizer.encode(string)).to(device) for string in listofstrings]

def listofrewardsintocputensors(listofrewards):
    return [torch.tensor(reward).to("cpu") for reward in listofrewards]

def build_batch(label_tensors, query_tensors, querylist, responselist):
    return {
        "label": label_tensors,
        "input_ids": query_tensors,
        "query": querylist,
        "response": responselist,
    }
    return batch


import pexpect  # You might need to install this package
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

# load models and tokenizers to be tested
MPNETmodel = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
MPNETtokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def get_embedding_from_text_model(text):
    return MPNETmodel(**MPNETtokenizer(text, return_tensors='pt'))[0][0][0].detach().numpy()

def get_actual_reward(history, prior_state, current_state):
    # get the embeddings for each item in history and compare to the current state embedding
    similarities = []
    try:
        for historyitem in history:
            historyembedding = get_embedding_from_text_model(historyitem['state'])
            currentembedding = get_embedding_from_text_model(current_state)
            similarity = np.round(
                cosine_similarity(historyembedding.reshape(1, -1), currentembedding.reshape(1, -1))[0][0]
                , decimals=2)
            similarities.append(
                similarity
            )

        priorembedding = get_embedding_from_text_model(prior_state)
        currentembedding = get_embedding_from_text_model(current_state)
        similarity = np.round(
            cosine_similarity(priorembedding.reshape(1, -1), currentembedding.reshape(1, -1))[0][0]
            , decimals=2)
        similarities.append(
            similarity
        )
        similarities = np.array(similarities).tolist()

        # Take 1 - the max of the similarities
        return np.round((1 - np.max(similarities)), 2)
    except:
        return 1

def update_history(state, action, predictedreward, actualreward, history):
    historylist = []
    for historyitem in history:
        historylist.append(historyitem['state'])
    historydictround = {}
    historydictround
    historydictround['state'] = state
    historydictround['action'] = action
    historydictround['predictedreward'] = predictedreward
    historydictround['actualreward'] = actualreward

    history.append(historydictround)
    return history


def get_sar_from_history(history):
    states = []
    actions = []
    predictedrewards = []
    actualrewards = []
    for historyitem in history:
        states.append(historyitem['state'])
        actions.append(historyitem['action'])
        predictedrewards.append(historyitem['predictedreward'])
        actualrewards.append(historyitem['actualreward'])
    return states, actions, predictedrewards, actualrewards

def update_reward_model(history):
    # get all the keys and all the values and add them to two separate lists

    states, actions, predictedrewards, actualrewards = get_sar_from_history(history)

    # embed all the states
    stateembeddings = []
    for state in states:
        stateembeddings.append(get_embedding_from_text_model(state))

    # embed all the actions
    actionembeddings = []
    for action in actions:
        actionembeddings.append(get_embedding_from_text_model(action))

    # concatenate the action and state embeddings
    actionstateembeddings = []
    for i in range(len(actionembeddings)):
        actionstateembeddings.append(np.concatenate((actionembeddings[i], stateembeddings[i])))

    # train a sklearn MLP to predict the reward
    from sklearn.neural_network import MLPRegressor

    # create the MLP with 100, 50
    mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50, 50), max_iter=1000, random_state=1)

    # fit the MLP
    mlp.fit(actionstateembeddings, actualrewards)

    return mlp

def getPredictedReward(state, action, model):

    if model == None:
        return 1

    # embed the action
    actionembedding = get_embedding_from_text_model(action)

    # embed the state
    stateembedding = get_embedding_from_text_model(state)

    # concatenate the action and state embeddings
    actionstateembedding = np.concatenate((actionembedding, stateembedding))

    # predict the reward
    return model.predict([actionstateembedding])[0]

generation_kwargs = {
    "min_length": -1,
    "top_k": 50,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 5,
}

def select_action(state, ppo_trainer):
    state = state + '\r>'
    gen_len = generation_kwargs["max_new_tokens"]
    tokenizedgeneratedaction = ppo_trainer.generate(torch.tensor(tokenizer.encode(state)).to(device), **generation_kwargs)
    tokenizedgeneratedaction = tokenizedgeneratedaction.squeeze()[-gen_len:]
    decodedaction = tokenizer.decode(tokenizedgeneratedaction)
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
    cleanedstring = ''.join(filter(whitelist.__contains__, decodedaction))
    return cleanedstring

def step(action, p):
    p.sendline(action)
    p.expect('>')
    next_state = p.before[len(action):]
    return next_state, p

stepsperepoch = 10
ppo_trainer.config.batch_size = stepsperepoch

def update_policy_model(ppo_trainer, history):

    states, actions, predictedrewards, actualrewards = get_sar_from_history(history)

    state_tensors = listofstringstolistofgpustringtensors(states)
    action_tensors = listofstringstolistofgpustringtensors(actions)
    rewards = listofrewardsintocputensors(actualrewards)
    label_tensors = [torch.tensor(1).to(device)]*ppo_trainer.config.batch_size

    batch = build_batch(label_tensors, state_tensors, states, actions)

    stats = ppo_trainer.step(state_tensors, action_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    return ppo_trainer


for epochs in range(1000):
    command = 'snap run zork'
    history = []
    action = ''
    rewardList = []
    rewardmodel = None
    debug = True
    p = pexpect.spawn(command, encoding='utf-8')
    p.echo = False
    p.expect('>')
    state = p.before
    for totalsteps in range(stepsperepoch):
        if debug: print('Starting state: ', state)
        action = select_action(state, ppo_trainer)
        if debug: print('Chosen action: ', action)
        next_state, p = step(action, p)
        if debug: print('Next state: ', next_state)
        predictedreward = getPredictedReward(state, action, rewardmodel)
        if debug: print('Predicted reward: ', predictedreward)
        actualreward = get_actual_reward(history, state, next_state)
        if debug: print('Actual reward: ', actualreward)
        history = update_history(state, action, predictedreward, actualreward, history=history)
        if debug: print('History: ', history)
        rewardmodel = update_reward_model(history)
        if debug: print('Reward model: ', rewardmodel)
        state = next_state
    p.close()
    ppo_trainer = update_policy_model(ppo_trainer, history)