#requirements: pocketsphinx, pyaudio, SpeechRecognition, hmmlearn
from hmmlearn import hmm
import math
import numpy as np
import speech_recognition as sr
import time
import random
import pickle

# Contains data about the grid, history and agent.
class RandomGameState(object):
    terrain = ['o', 'o', 'o', 'o', 'X']  # terrain to sample from
    actions = {'left':(-1, 0), 'right':(1, 0), 'up':(0, -1), 'down':(0, 1)} # action names and their vector effects on the agent
    action_index = {'left':0, 'right':1, 'up':2, 'down':3} #action name to int index
    action_list = ['left', 'right', 'up', 'down'] #list of actions
    
    def __init__(self, grid_size):
        self.grid = []
        # randomly create the grid
        for i in range(grid_size[1]):
            row = []
            for j in range(grid_size[0]):
                row.append(random.choice(RandomGameState.terrain))
            self.grid.append(row)
        
        self.action_history = []
        #randomly initialize the agent and goal
        self.agent_location = (random.randint(0, grid_size[0] - 1), 
                              random.randint(0, grid_size[1] - 1))
        self.goal_location = (random.randint(0, grid_size[0] - 1), 
                              random.randint(0, grid_size[1] - 1))
        self.grid_size = grid_size
        
    def update_state(self, actual_action):
        # given an actual action, update the gamestate
        action_vector = RandomGameState.actions[actual_action]
        self.agent_location = (
                max(0, min(
                        self.grid_size[0] - 1, 
                        self.agent_location[0] + action_vector[0])),
                max(0, min(
                        self.grid_size[1] - 1, 
                        self.agent_location[1] + action_vector[1])))
                        
        self.action_history.append(actual_action)
        
    def local_state(self):
        # provide the terrain in the local area of the agent
        agentX, agentY = self.agent_location
        local_grid = []
        for x in range(agentX - 1, agentX + 2):
            for y in range(agentY - 1, agentY + 2):
                if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
                    local_grid.append('E')
                elif (x,y) == self.goal_location and (x,y) != self.agent_location:
                    local_grid.append('G')
                elif (x,y) != self.agent_location:
                    local_grid.append(self.grid[y][x])
                    
        return local_grid
                    
    def render(self):
        # print the grid to the console
        renderstring = ''
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                if (x,y) == self.agent_location:
                    renderstring += 'A '
                elif (x,y) == self.goal_location:
                    renderstring += 'G '
                else:
                    renderstring += self.grid[y][x]+ ' '
            renderstring += '\n\n'
        print(renderstring)
        
    def is_over(self):
        # check whether the game is over
        return self.agent_location == self.goal_location

class VoiceModel(object):
    # Contains a model for predicting possible voiced phrases
    def phrase_probabilities(self, decoder, normalized = True, topN = 5):
        # Produce the prabilities of different phrases being said
        denom = 0.0
        normalizer = 0.0
        # get the n best phrase results from the decoder
        results = sorted(zip(decoder.nbest(), range(topN)),
                         key = lambda x:x[0].score,
                         reverse = True)
        if len(results) == 1:
            return {results[0][0].hypstr:1.0}
        
        # loop through the results, and weight results by rank and
        # difference to rank
        for best, i in results:
            if i == 0:
                normalizer = best.score
                denom += 1.0
            else:
                denom += math.pow(2, best.score - normalizer)
        
        topN = min(topN, len(results))
        phrases = {}
        # normalize the probabilities to sum to 1.0
        for best, i in results:
            prob = (topN-i-1)/float((topN-1)*topN) + .5/float(topN) 
            phrases[best.hypstr] = phrases.get(best.hypstr, 0.0) + prob
            
        return(phrases)
    
    def process_voice(self):
        # recognize a phrase from an audio input
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                fancy_print("Say a voice command!")
                audio = r.listen(source, timeout = 3, phrase_time_limit = 5)
            decoder = r.recognize_sphinx(audio, show_all = True)
            return(self.phrase_probabilities(decoder))
        except sr.WaitTimeoutError:
            return {None:1.0}

def wait_for_override(predicted_action):
    # allow the user to override the action
    char = input()
    if char == '':
        return predicted_action
    if char[0] == 'w':
        return 'up'
    elif char[0] == 'a':
        return 'left'
    elif char[0] == 's':
        return 'down'
    elif char[0] == 'd':
        return 'right'
    return predicted_action

def log_data(likely_phrases, game_state, action, logfile = './data/data.log',
             delimiter = '|||'):
    # log the phrases, game state and action in a log file for later training
    with open(logfile, 'a') as f:
        f.write(delimiter)
        data = (likely_phrases, game_state, action)
        f.write(pickle.dumps(data, 0).decode())
        
def parse_log_data(logfile = './data/data.log', delimiter = '|||'):
    # parses the log file data back into python data
    examples = [[]]
    with open(logfile, 'r') as f:
        rawdata = f.read()
    if rawdata == '':
        return examples
    rawdata = rawdata.split(delimiter)[1:]
    pickle.loads(rawdata[0].encode())
    for d in rawdata:
        truedata = pickle.loads(d.encode())
        if truedata == (None, None, None):
            examples.append([])
        else:
            examples[-1].append(truedata)
    return examples[:(len(examples)-1)]

def phrases_to_4gramprobs(likely_phrases, prefix = ''):
    # converts phrases into character 4-grams
    # this is useful for capturing words, different forms of the words,
    # and connections between two words.  
    # See https://dl.acm.org/citation.cfm?id=860528
    phrases = {}
    for phrase, prob in likely_phrases.items():
        if phrase == None:
            phrase = ''
        new_phrase = ' ' + phrase + ' '
        gramset = set()
        for i in range(len(new_phrase)-3):
            gramset.add(prefix + new_phrase[i:(i+4)])
        for gram in gramset:
            phrases[gram] = phrases.get(gram, 0.0) + prob
    return phrases

class JPD(object):
    # Class for modeling a joint probability distribution between a single 
    # variable and the possible actions
    
    def __init__(self, obs_set):
        # obs_set should be the set of possible actions/outcome variables
        self.dist = {}
        self.observations = {}
        self.n = 0.0
        for obs in obs_set:
            self.observations[obs] = 0.0
    
    def add_data(self, keys_and_conf, obs):
        # add a set of observations to the probability table
        # key should be in the format {leftvalue:confidence}
        # obs should be the observed action/right side
        for key, conf in keys_and_conf.items():
            keyvals = self.dist.get(key, [{}, 0.0])
            keyobsvals = keyvals[0].get(obs, 0.0)
            keyvals[0][obs] = keyobsvals + conf
            keyvals[1] += conf
            self.dist[key] = keyvals
        self.n += 1.0
        self.observations[obs] += 1.0
    
    def get_pd(self, keys_and_conf):
        # given a set of possible values, and the confidence in those values,
        # find the left hand/action probabilities
        baseprobs = {}
        probs = {}
        locz = len(self.observations)
        for obskey, nval in self.observations.items():
            probs[obskey] = 1.0 / locz
            baseprobs[obskey] = 1.0 / locz
        
        for key, conf in keys_and_conf.items():
            # for every key, blend the overall rate with the observed rate
            incrementalz = 0.0
            keyvals = self.dist.get(key, [{}, 0.0])
            keyvallocz = keyvals[1] + len(self.observations)
            for obs in self.observations.keys():
                obsprob = (keyvals[0].get(obs, 0.0) + 1.0) / keyvallocz
                bp = baseprobs[obs]
                locprob = conf * obsprob + (1 - conf) * bp
                jp = locprob * probs[obs]
                incrementalz += jp
                probs[obs] = jp
            
            for key, prob in probs.items():
                probs[key] = prob/incrementalz
        return probs

class DesiredActionModel(object):
    # The probabalistic model for predicting the desired action
    # It consists of the following model:
    #
    # audio -> phrases -> character4grams----------
    #                                             |
    # terrain up----------------------------      |
    # terrain down-------------------------|      |
    # terrain left ----------------------> v      v
    # terrain right ---------------------> desired action
    #                                      ^      ^
    #                                      |      |
    # character4grams (t-1)-----------------      |
    #                                             |
    # action (t-n) ... action (t-1) --------------|
    
    def __init__(self, logdatalocation = './data/data.log'):
        # initial the hmm and joint probability distributions
        # for each
        self.voice2action = JPD(RandomGameState.actions.keys())
        self.gsup2action = JPD(RandomGameState.actions.keys())
        self.gsleft2action = JPD(RandomGameState.actions.keys())
        self.gsright2action = JPD(RandomGameState.actions.keys())
        self.gsdown2action = JPD(RandomGameState.actions.keys())
        self.prevvoice2action = JPD(RandomGameState.actions.keys())
        self.actions2action = hmm.MultinomialHMM(len(RandomGameState.actions))
        
        # parse the historical data and train the models/JPDs
        histdata = parse_log_data(logfile = logdatalocation)
        X = []
        xlens = []
        print("Training on " + str(len(histdata)) + " games of user feedback")
        for seq in histdata:
            xlens.append(len(seq))
            for i in range(len(seq)):#act in seq:
                act = seq[i]
                self.voice2action.add_data(phrases_to_4gramprobs(act[0]), act[2])
                if i > 0:
                    self.prevvoice2action.add_data(
                            phrases_to_4gramprobs(
                                    seq[i-1][0], prefix = seq[i-1][2]), 
                                    act[2])
                self.gsup2action.add_data({act[1][1]:1.0}, act[2])
                self.gsleft2action.add_data({act[1][3]:1.0}, act[2])
                self.gsright2action.add_data({act[1][4]:1.0}, act[2])
                self.gsdown2action.add_data({act[1][6]:1.0}, act[2])
                X.append([RandomGameState.action_index[act[2]]])
        if len(X) > 0:
            try:
                self.actions2action.fit(X, xlens)
            except:
                pass
            
    def predict(self, likely_phrases, game_state, prev_likely_phrases):
        # predict the desired action given the likely_phrases, game state
        # and t-1 likely phrases
        baseline_probs = {'left':.25, 'right':.25, 'up':.25, 'down':.25}
        lgs = game_state.local_state()
        # voice -> desired action
        vaprobs = self.voice2action.get_pd(phrases_to_4gramprobs(likely_phrases))
        # t-1 voice -> desired action
        if prev_likely_phrases is not None:
            pvaprobs = self.prevvoice2action.get_pd(
                    phrases_to_4gramprobs(
                            prev_likely_phrases, prefix = game_state.action_history[-1]))
        else:
            pvaprobs = baseline_probs
            
        # terrain to N,S,E,W -> desired action
        gsupprobs = self.gsup2action.get_pd({lgs[1]:1.0})
        gsleftprobs = self.gsleft2action.get_pd({lgs[3]:1.0})
        gsrightprobs = self.gsright2action.get_pd({lgs[4]:1.0})
        gsdownprobs = self.gsdown2action.get_pd({lgs[6]:1.0})
        
        # hmm desired action
        hist = []
        for action in game_state.action_history:
            hist.append([RandomGameState.action_index[action]])
        
        if len(hist) > 0:
            try:
                rawProbs = self.actions2action.predict_proba(np.array(hist))[-1]
            except:
                rawProbs = [.25,.25,.25,.25]
        else:
            rawProbs = [.25, .25, .25, .25]
        a2aprobs = {}
        for i in range(len(rawProbs)):
            a2aprobs[RandomGameState.action_list[i]] = rawProbs[i]
        
        # combine all probabilities and normalize
        actionprobs = combine_and_normalize(vaprobs, gsupprobs)
        actionprobs = combine_and_normalize(actionprobs, gsleftprobs)
        actionprobs = combine_and_normalize(actionprobs, gsrightprobs)
        actionprobs = combine_and_normalize(actionprobs, gsdownprobs)
        actionprobs = combine_and_normalize(actionprobs, a2aprobs)
        actionprobs = combine_and_normalize(actionprobs, pvaprobs)
        return actionprobs

def combine_and_normalize(pa, pb):
    pc = {}
    z = 0.0
    for key in pa.keys():
        pc[key] = pa[key] * pb[key]
        z += pa[key] * pb[key]
    for key in pc.keys():
        pc[key] = pc[key] / z
    return pc

def fancy_print(string):
    for ch in string:
        print(ch, end='')
        time.sleep(.02)
    print('')
        
def main(override = True, debug = False):
    # initialize a game state, desired action model and voice model
    game_state = RandomGameState((5,5))
    voice_model = VoiceModel()
    desired_action_model = DesiredActionModel()
    
    # start the main game loop
    prev_likely_phrases = None
    while not game_state.is_over():
        game_state.render()
        # listen to a voice command and predict the probabilities of different
        # outcomes
        likely_phrases = voice_model.process_voice()
        actionprobs = desired_action_model.predict(likely_phrases, game_state, prev_likely_phrases)
        if debug:
            print('Likely phrases:')
            print(likely_phrases)
            print(actionprobs)
        
        # retrieve the most probable action
        desired_action = None
        daprob = -1.0
        for act, prob in actionprobs.items():
            if prob > daprob:
                desired_action = act
                daprob = prob
        
        fancy_print('Proceeding with action: ' + desired_action)
        print('')
        # if override is enabled, allow the user to override the desired action
        # log the results for future model training
        if override:
            fancy_print('Press enter to continue.  Press W,A,S, or D to override')
            actual_action = wait_for_override(desired_action)
            log_data(likely_phrases, game_state.local_state(), actual_action)
        else:
            actual_action = desired_action
        
        # update game state, continue!
        game_state.update_state(actual_action)
        prev_likely_phrases = likely_phrases
    
    if override:
        log_data(None, None, None)
    fancy_print('Congrats, you found the goal!!!')
    fancy_print('Press Q to quit, or enter to continue')
    if input() in set(['Q','q']):
        return None
    return main(override = override, debug = debug)

if __name__ == '__main__':
    main()
