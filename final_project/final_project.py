import sys

##############Main Function##################

if len(sys.argv) < 2 or sys.argv[1] not in ['EN','ES']:
    print ('Please make sure you have installed Python 3.4 or above!')
    print ("Usage:  python final_project.py EN/ES")
    sys.exit()

folder = sys.argv[1]

############## Part 1 ##################
def read_train(dir):
    X_train = []
    y_train = []
    all_labels = []
    all_words = []
    count = 0
    with open(dir) as f:
        X_sent = []
        y_sent = []
        for line in f:
            if line == '\n':
                X_train.append(X_sent)
                y_train.append(y_sent)
                X_sent=[]
                y_sent=[]
            else:
                temp = line.strip().split()
                X_sent.append(temp[0])
                y_sent.append(temp[1])
                count += 1
                if temp[1] not in all_labels:
                    all_labels.append(temp[1])
                if temp[0] not in all_words:
                    all_words.append(temp[0])
    #print(count)
    return X_train,y_train, all_labels, all_words

def read_val(dir):
    X_dev = []
    with open(dir) as f:
        X_sent = []
        for line in f:
            if line == '\n':
                X_dev.append(X_sent)
                X_sent=[]
            else:
                temp = line.strip()
                X_sent.append(temp)
    return X_dev

X_train, y_train, ALL_LABELS, ALL_WORDS = read_train('data/'+folder+'/train')
######ALL_LABLES does not contain START STOP

ETA = 0.1


from collections import defaultdict
import math
y_dict = defaultdict(int)
yx_dict = defaultdict(int)
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        y_dict[y_train[i][j]] += 1
        yx_dict[(y_train[i][j], X_train[i][j])] += 1
emission = defaultdict(int)
for key in yx_dict:
    label = key[0]
    name = 'emission:'+str(key[0])+'+'+str(key[1])
    emission[name] = math.log(float(yx_dict[key])/y_dict[label])
#print(emission)

yi_dict = defaultdict(int)
yij_dict = defaultdict(int)
for i in range(len(X_train)):
    if len(y_train[i]) == 0:
        continue
    yi_dict['START'] += 1
    yij_dict[('START',y_train[i][0])] += 1
    yij_dict[(y_train[i][-1],'STOP')] += 1
    for j in range(len(X_train[i])-1):
        yi_dict[y_train[i][j]] += 1
        yij_dict[(y_train[i][j],y_train[i][j+1])] += 1
    yi_dict[y_train[i][-1]] += 1
            
transition = defaultdict(int)
for key in yij_dict:
    name = 'transition:'+str(key[0])+'+'+str(key[1])
    transition[name] = math.log(float(yij_dict[key])/yi_dict[key[0]])
#print(transition)

features = {}
for j in ALL_LABELS+["START", "STOP"]:
    for jj in ALL_LABELS+["START","STOP"]:
        name = 'transition:'+str(j)+'+'+str(jj)
        features[name] = -2**21
for j in ALL_LABELS:
    for i in ALL_WORDS:
        name = 'emission:'+str(j)+'+'+str(i)
        features[name] = -2**21
        
for key in emission:
    features[key] = emission[key]
for key in transition:
    features[key] = transition[key]
    
name_to_index = {}
index_to_name = {}
index = 0
for key in features:
    name_to_index[key] = index
    index_to_name[index] = key
    index += 1

############## Part 2 ##################


def parse_name(feature_name):
    if feature_name[:8] == 'emission':
        feat_type = "emission"
    else:
        feat_type = 'transition'
        
    feat_from = feature_name[feature_name.find(':')+1 : feature_name.find('+')]
    feat_to = feature_name[feature_name.find('+')+1 : ]  #find first +, this is because some words can be '+'
    
    return feat_type, feat_from, feat_to
    
    
def count_single_feat(feature_name, input_x, input_y):
    #sentence has to be in a list
    result = 0
    feat_type, feat_from, feat_to = parse_name(feature_name)
    if feat_type == 'emission':
        for i in range(len(input_x)):
            if input_y[i] == feat_from and input_x[i] == feat_to:
                result += 1
    else: #transition
        if feat_from == 'START' and feat_to == input_y[0]:
            result = 1
            return result
        
        if feat_from == input_y[-1] and feat_to == 'STOP':
            result = 1
            return result
        
        for i in range(1,len(input_x)):   
            if input_y[i-1] == feat_from and input_y[i] == feat_to:
                result += 1
    return result

def compute_feature_vector(features, name_to_index, input_x, input_y):
    vector = [0 for i in range(len(name_to_index))]
    for key in features:
        vector[name_to_index[key]] = count_single_feat(key, input_x, input_y)
    return vector

def compute_sent_score(features, name_to_index, index_to_name, input_x, input_y):
    total_score = 0
    vector = compute_feature_vector(features, name_to_index, input_x, input_y)
    for i in range(len(vector)):
        total_score += features[index_to_name[i]] * vector[i]
    return total_score

#compute_sent_score(features, name_to_index, index_to_name, 'I love you .'.split(),'O O O O'.split())

def compute_sent_score_fast(features, input_x, input_y):
    total_score = 0
    total_score += features['transition:START+'+input_y[0]]
    if 'emission:'+input_y[0]+'+'+input_x[0] in features:
        total_score += features['emission:'+input_y[0]+'+'+input_x[0]]
    
    for i in range(1, len(input_x)):
        total_score+= features['transition:'+input_y[i-1]+'+'+input_y[i]]
        if 'emission:'+input_y[i]+'+'+input_x[i] in features:
            total_score += features['emission:'+input_y[i]+'+'+input_x[i]]
    total_score+= features['transition:'+input_y[-1]+'+STOP']
        
    return total_score

#compute_sent_score_fast(features, 'I love you .'.split(),'O O O O'.split())


def viterbi(sent, features): 
    V_table = [[-2**31 for i in range(len(sent))] for j in range(len(ALL_LABELS))]
    bt = [[None for i in range(len(sent))] for j in range(len(ALL_LABELS))]
    result_score = -2**31
    bt_stop = None
    
    ###### first word, from start to first label#####
    for j in range(len(ALL_LABELS)):
        bt[j][0] = 'START'
        if ('emission:'+ALL_LABELS[j] +'+'+ sent[0] not in features): #if emission not exist, 0 score
            V_table[j][0] = features['transition:START+'+ALL_LABELS[j]] + 0
        else:  
            V_table[j][0] = features['transition:START+'+ALL_LABELS[j]] + features['emission:'+ALL_LABELS[j]+'+'+sent[0]]
            
            
    ###### second word to last word#####
    for i in range(1, len(sent)):
        for j in range(len(ALL_LABELS)):
            for jj in range(len(ALL_LABELS)): # compute the transition score from label jj of prev word to j
                if V_table[j][i] < V_table[jj][i-1] + features['transition:'+ALL_LABELS[jj]+'+'+ALL_LABELS[j]]:
                    V_table[j][i] = V_table[jj][i-1] + features['transition:'+ALL_LABELS[jj]+'+'+ALL_LABELS[j]]
                    bt[j][i] = jj
                    
            if ('emission:'+ALL_LABELS[j]+'+'+sent[i] in features): #if emission exists, add the emission score
                V_table[j][i] += features['emission:'+ALL_LABELS[j]+'+'+sent[i]]

                
    ###### last word to STOP#####
    for jj in range(len(ALL_LABELS)):
        if result_score < V_table[jj][-1] + features['transition:'+ALL_LABELS[jj]+'+STOP']:
            result_score = V_table[jj][-1] + features['transition:'+ALL_LABELS[jj]+'+STOP']
            bt_stop = jj
    output = ['STOP']
#     print(V_table)
#     print(bt)
    output.append(ALL_LABELS[bt_stop])
    wanted_label = bt_stop

    for i in reversed(range(1, len(sent))):
        output.append(ALL_LABELS[bt[wanted_label][i]])
        wanted_label = bt[wanted_label][i]
    output.append('START')
    return output[::-1], result_score

#viterbi('I have NEVER been disappointed in the Red Eye .'.split(),features)



X_dev = read_val('data/'+folder+'/dev.in')
predicted_dev = []
for i in range(len(X_dev)):
    output,_ = viterbi(X_dev[i], features)
    predicted_dev.append(output[1:-1])

def gen_output(dir, X, y):
    with open(dir,'w') as f:
        for i in range(len(X)):
            for j in range(len(X[i])):
                string = X[i][j] + ' ' + y[i][j] + '\n'
                f.write(string)
            f.write('\n')
gen_output('data/'+folder+'/dev.p2.out', X_dev, predicted_dev)




############## Part 3 ##################

import numpy as np

def logSumExp(ns):
    max = np.max(ns)
    ds = ns - max
    sumOfExp = np.exp(ds).sum()
    return max + np.log(sumOfExp)

import time
def forward_backward(sent, features): # features should be smoothed, alpha is actually log_alpha
    alpha = [[0 for i in range(len(sent))] for j in range(len(ALL_LABELS))]
    forward_result_score = None
    
    ###### forward: first word, from start to first label#####
    for j in range(len(ALL_LABELS)):
        alpha[j][0] = features['transition:START+'+ALL_LABELS[j]] + features['emission:'+ALL_LABELS[j]+'+'+sent[0]]
        
        
    ###### forward: second word to last word#####
    for i in range(1, len(sent)):
        for j in range(len(ALL_LABELS)):
            list_for_exp = np.zeros(len(ALL_LABELS))
            for jj in range(len(ALL_LABELS)): # compute the transition score from label jj of prev word to j
                list_for_exp[jj] = alpha[jj][i-1] + features['transition:'+ALL_LABELS[jj]+'+'+ALL_LABELS[j]] + features['emission:'+ALL_LABELS[j]+'+'+sent[i]]
            alpha[j][i] = logSumExp(list_for_exp)
    
    ###### forward: last word to STOP#####
    list_for_exp = np.zeros(len(ALL_LABELS))
    for j in range(len(ALL_LABELS)):
        list_for_exp[j] = alpha[j][-1] + features['transition:'+ALL_LABELS[j]+'+STOP']
    forward_result_score = logSumExp(list_for_exp)
    
    
    ######backward######
    beta = [[0 for i in range(len(sent))] for j in range(len(ALL_LABELS))]
    backward_result_score = None
    for j in range(len(ALL_LABELS)):
        beta[j][-1] = features['transition:'+ALL_LABELS[j]+'+STOP']
    
    for i in reversed(range(len(sent)-1)):
        for j in range(len(ALL_LABELS)):
            list_for_exp = np.zeros(len(ALL_LABELS))
            for jj in range(len(ALL_LABELS)):
                list_for_exp[jj] = beta[jj][i+1] + features['transition:'+ALL_LABELS[j]+'+'+ALL_LABELS[jj]] + features['emission:'+ALL_LABELS[jj]+'+'+sent[i+1]]
            #print(list_for_exp)
            beta[j][i] = logSumExp(list_for_exp)
            
    list_for_exp = np.zeros(len(ALL_LABELS))
    for j in range(len(ALL_LABELS)):
        list_for_exp[j] = beta[j][0] + features['transition:START+'+ALL_LABELS[j]] + features['emission:'+ALL_LABELS[j]+'+'+sent[0]]
    backward_result_score = logSumExp(list_for_exp)
    
    #print(forward_result_score, backward_result_score)
    return alpha, beta, forward_result_score



def soft_count_all_feats_in_single_sent(sent, features, alpha, beta, total_score):
    output = {}
    for key in features:
        output[key] = 0
    
    ###### compute for transition features ######
    for j in range(len(ALL_LABELS)):
        output['transition:START+'+ALL_LABELS[j]] += math.exp(alpha[j][0] + beta[j][0] - total_score)
    
    for i in range(1, len(sent)):
        for j in range(len(ALL_LABELS)):
            for jj in range(len(ALL_LABELS)):
                feature_name = 'transition:'+ALL_LABELS[jj]+'+'+ALL_LABELS[j]
                output[feature_name] += math.exp(alpha[jj][i-1] \
                                              + features[feature_name] \
                                              + features['emission:'+ALL_LABELS[j]+'+'+sent[i]] \
                                              + beta[j][i] \
                                              - total_score)
    for j in range(len(ALL_LABELS)):
        output['transition:'+ALL_LABELS[j]+'+STOP'] += math.exp(alpha[j][-1] + beta[j][-1] - total_score)
    
    ###### compute for emission features ######
    for i in range(len(sent)):
        for j in range(len(ALL_LABELS)):
            feature_name = 'emission:'+ALL_LABELS[j]+'+'+sent[i]
            output[feature_name] += math.exp(alpha[j][i] + beta[j][i] - total_score)
    
    return output


def hard_count_all_feats_in_single_sent(features, input_x, input_y):
    output = {}
    for key in features:
        output[key] = 0
    
    output['transition:START+'+input_y[0]] += 1
    output['emission:'+input_y[0]+'+'+input_x[0]] += 1
    for i in range(1, len(input_x)):
        output['transition:'+input_y[i-1]+'+'+input_y[i]] += 1
        output['emission:'+input_y[i]+'+'+input_x[i]] += 1
    output['transition:'+input_y[-1]+'+STOP']+=1
    
    return output


    
                
def compute_all_loss(X_train, y_train, features, name_to_index, index_to_name):
    loss = 0

    for i in range(len(X_train)):
        gold_score = compute_sent_score_fast(features, X_train[i], y_train[i])
        _, _, total_score = forward_backward(X_train[i],features)
        loss += gold_score - total_score
    return -loss



def compute_all_loss_and_grad(X_train, y_train, features, name_to_index, index_to_name):
    features_grad = {}
    for key in features:
        features_grad[key] = 0
    loss = 0

    for i in range(len(X_train)):
        
        alpha, beta, total_score = forward_backward(X_train[i],features)
        
        ##compute loss
        gold_score = compute_sent_score_fast(features, X_train[i], y_train[i])
        
        loss += gold_score - total_score
        
        ##compute gradient
        soft_count = soft_count_all_feats_in_single_sent(X_train[i], features, alpha, beta, total_score)
        #print(soft_count['emission:O+all'])
        hard_count = hard_count_all_feats_in_single_sent(features, X_train[i], y_train[i])

        for feature_name in features:
            features_grad[feature_name] += soft_count[feature_name] - hard_count[feature_name]
    return -loss, features_grad

temp_loss = compute_all_loss(X_train, y_train, features, name_to_index, index_to_name)
print('Loss using features from part 1:', temp_loss)





############## Part 4 ##################


print('Starting training CRF...')

from scipy.optimize import fmin_l_bfgs_b

record_loss = []
def get_loss_grad(w, *args):
    '''
    This function will be called by "fmin_l_bfgs_b"
    Arg:
    w: weights, numpy array
    Returns:
    loss: loss, float
    grads: gradients, numpy array
    args = (X_train, y_train)
    name_to_index etc. is from global
    '''
    grads = np.zeros(len(w))
    X_train  = args[0]
    y_train = args[1]
    features_dict = {}
    for i in range(len(w)):
        features_dict[index_to_name[i]] = w[i]
        
    loss, grad_dict = compute_all_loss_and_grad(X_train, y_train, features_dict, name_to_index, index_to_name)
    print('plain loss', loss)
    loss += ETA * np.sum(w**2)
    record_loss.append(loss)
    
    for key in grad_dict:
        grads[name_to_index[key]] = grad_dict[key] + 2 * ETA * w[name_to_index[key]]

    
    print('total loss', loss)
    print('grad abs',(np.sum(grads**2))**0.5)
    # to be completed by you,
    # based on the modified loss and gradients,
    # with L2 regularization included
    return loss, grads
    

init_w = np.zeros(len(index_to_name))
result = fmin_l_bfgs_b(get_loss_grad, init_w, pgtol=0.05, args=(X_train,y_train))
print('Training finished')

np.save('data/'+folder+'/record_loss.npy', np.asarray(record_loss))
w_final = result[0]
np.save('data/'+folder+'/result.npy', np.asarray(w_final))
new_features = {}
for i in range(len(w_final)):
    new_features[index_to_name[i]] = w_final[i]
X_dev = read_val('data/'+folder+'/dev.in')
predicted_dev = []
for i in range(len(X_dev)):
    output,_ = viterbi(X_dev[i], new_features)
    predicted_dev.append(output[1:-1])
gen_output('data/'+folder+'/dev.p4.out', X_dev, predicted_dev)



