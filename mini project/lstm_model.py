import torch
from torch import nn
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_size,
                 hidden_size, num_layer, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size, self.num_layer = hidden_size, num_layer
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layer, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.init_weights()
 
    def init_weights(self):
        '''Initialize weights'''
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def repackage_hidden(self, h):
        """Detach tensor from their history, no gradient backpropaged"""
        return tuple(v.detach() for v in h)
 
    def forward(self, input_tensor, hidden):
        '''
        Args:
            input_tensor: float tensor, size: batch_size*word_num
            hidden: a tuple(LSTM initial states)
        Return:
            outputs: log softmax of prediction scores, word_num*batch_size*vocab_size
            hidden: a tuple(LSTM initial states)
        '''
        #Get embeddings for the input tensors
        emb = self.encoder(input_tensor)
        #dropout
        emb = self.drop(emb)
        #Remove history of hidden states
        hidden = self.repackage_hidden(hidden)
        #rnn layer
        output, hidden = self.rnn(emb, hidden)
        #dropout
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        outputs = decoded.view(output.size(0), output.size(1), decoded.size(1))
        #Compute the softmax values
        outputs = self.softmax(outputs)
        #Set the values within a range
        outputs = torch.clamp(outputs, min=0.000000001, max=100000)
        #Compute log values
        outputs = torch.log(outputs)
        return outputs, hidden
 
    def init_hidden(self, batch_size):
        '''
        Initialize LSTM hidden states
        '''
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layer, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_layer, batch_size, self.hidden_size))