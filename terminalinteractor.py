import pexpect  # You might need to install this package
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

# load models and tokenizers to be tested
MPNETmodel = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
MPNETtokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def get_embedding_from_text_model(text):
    return MPNETmodel(**MPNETtokenizer(text, return_tensors='pt'))[0][0][0].detach().numpy()

def get_actual_reward(history, current_state):
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

def select_action(state):
    return input('Enter action: ')

def step(action, p):
    p.sendline(action)
    p.expect('>')
    next_state = p.before[len(action):]
    return next_state, p

command = 'snap run zork'
history = []
action = ''
rewardList = []
rewardmodel = None
model = None
debug = True
p = pexpect.spawn(command, encoding='utf-8')
p.echo = False
p.expect('>')
state = p.before
while True:

    if debug: print('Starting state: ', state)
    action = select_action(state)
    if debug: print('Chosen action: ', action)
    next_state, p = step(action, p)
    if debug: print('Next state: ', next_state)
    predictedreward = getPredictedReward(state, action, rewardmodel)
    if debug: print('Predicted reward: ', predictedreward)
    actualreward = get_actual_reward(history, next_state)
    if debug: print('Actual reward: ', actualreward)
    history = update_history(state, action, predictedreward, actualreward, history=history)
    if debug: print('History: ', history)
    rewardmodel = update_reward_model(history)
    if debug: print('Reward model: ', rewardmodel)
    state = next_state

# update policy model
p.close()