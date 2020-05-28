import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys

if len(sys.argv) < 2 or sys.argv[1] not in ['EN','ES']:
    print ('Please make sure you have installed Python 3.4 or above!')
    print ("Usage:  python part5.py EN/ES")
    sys.exit()

folder = sys.argv[1]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
    
    idxs = []
    for i in range(len(seq)):
        try:
            idxs.append(to_ix[seq[i]])
        except:
            idxs.append(to_ix['UNK'])
    
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score +         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score +                 self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

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
    print(count)
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

training_data = []
for i in range(len(X_train)):
    training_data.append((X_train[i], y_train[i]))

word_to_ix = {'UNK':0}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {'O':0, 'B-negative':1, 'I-negative':2, 'B-positive':3,
                  'B-neutral':4, 'I-positive':5, 'I-neutral':6, 'B-conflict':7, START_TAG: 8, STOP_TAG: 9}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(
        100):  # again, normally you would NOT do 300 epochs, it is toy data
    print('epoch:', epoch)
    opt_loss = 0
    num_enu = 0
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
        
        opt_loss += loss.item()
        num_enu += len(sentence)
    opt_loss /= num_enu
    print('Training loss:', opt_loss)

X_dev = read_val('data/'+folder+'/dev.in')
X_test = []
for i in range(len(X_dev)):
    X_test.append(prepare_sequence(X_dev[i], word_to_ix))

X_pred = []
with torch.no_grad():
    for i in range(len(X_test)):
        X_pred.append(model(X_test[i])[1])

label_list = ['O', 'B-negative', 'I-negative', 'B-positive',
                  'B-neutral', 'I-positive', 'I-neutral', 'B-conflict']

for i in range(len(X_pred)):
    for j in range(len(X_pred[i])):
        X_pred[i][j] = label_list[X_pred[i][j]]

def gen_output(dir, X, y):
    with open(dir,'w') as f:
        for i in range(len(X)):
            for j in range(len(X[i])):
                string = X[i][j] + ' ' + y[i][j] + '\n'
                f.write(string)
            f.write('\n')

gen_output('data/'+folder+'/dev.p5.out', X_dev, X_pred)

####### Test
X_dev = read_val('data/'+folder+'/test.in')
X_test = []
for i in range(len(X_dev)):
    X_test.append(prepare_sequence(X_dev[i], word_to_ix))

X_pred = []
with torch.no_grad():
    for i in range(len(X_test)):
        X_pred.append(model(X_test[i])[1])

for i in range(len(X_pred)):
    for j in range(len(X_pred[i])):
        X_pred[i][j] = label_list[X_pred[i][j]]

gen_output('data/'+folder+'/test.p5.out', X_dev, X_pred)

