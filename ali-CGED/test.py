import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from torch.autograd import Variable
HIDDEN_DIM = 200
START_TAG = "<START>"
STOP_TAG = "<STOP>"
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)
class BiLSTM_CRF(nn.Module):

    def __init__(self,hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        #self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        #self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.dense = nn.Linear(450,200)
        #self.dense2 = nn.Linear(2,2)
        self.droupout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(615, hidden_dim // 2,
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
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
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
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence1, sentence2):
        self.hidden = self.init_hidden()
        sentence1 = sentence1.view(len(sentence1), 1, -1)
        embeds1 = F.relu(self.droupout(self.dense(sentence1)))
        embeds1 = embeds1.view(len(sentence1), 1, -1)
        embeds2 = sentence2.view(len(sentence2), 1, -1)
        embedsm = torch.cat((embeds1, embeds2), 2)
        lstm_out, self.hidden = self.lstm(embedsm, self.hidden)
        lstm_out = lstm_out.view(len(sentence1), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            #print(type(self.transitions[tags[i + 1], tags[i]]))
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
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
                viterbivars_t.append(next_tag_var[0][best_tag_id])
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

    def neg_log_likelihood(self, sentence1,sentence2, tags):
        feats = self._get_lstm_features(sentence1,sentence2)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence1,sentence2):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence1,sentence2)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


tag_to_ix = {"O": 0, "B-R": 1, "I-R": 2,"B-M": 3, "I-M": 4, "B-S": 5, "I-S": 6,"B-W": 7, "I-W": 8, START_TAG: 9, STOP_TAG: 10}

model = BiLSTM_CRF(HIDDEN_DIM)
model.load_state_dict(torch.load('./mymodel.pkl'))
fout_1 = open('../test_feat_part1_without_dependence.pickle', mode='rb')
# json.dump(feats, fout_1, ensure_ascii=False)
feat_part1_without_dependence = pickle.load(fout_1)
fout_2 = open('../test_feat_part2_dependence.pickle', mode='rb')
# json.dump(dependences_matrix_embedding, fout_2, ensure_ascii=False)
feat_part2_dependence = pickle.load(fout_2)
flabel = open('../test/regular.txt', mode='r')
labels = flabel.readlines()
fout_1.close()
fout_2.close()
flabel.close()
got = 0
for i in range(len(feat_part1_without_dependence)):
    sentence_in1 = feat_part1_without_dependence[i]
    sentence_in2 = feat_part2_dependence[i]
    sentence_in1 = Variable(torch.from_numpy(np.array(sentence_in1))).float()
    sentence_in2 = Variable(torch.from_numpy(np.array(sentence_in2))).float()
    print(model(sentence_in1,sentence_in2))
    pp = model(sentence_in1,sentence_in2)[1]
    print('.......')
    print(labels[i])
    flag1 = True
    flag2 = True
    for i in labels[i]:
        if i != 'O':
            flag1 = False
            break
    for j in range(len(pp)):
        if pp[j]!=0:
            flag2 = False
            break
    if flag1 == flag2:
        got = got + 1
print(got)
print(len(feat_part1_without_dependence))
print(got/len(feat_part1_without_dependence))
