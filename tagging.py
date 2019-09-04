import torch
from torch import nn
import abc
import flair
import torch.nn.functional as F

def masked_crossentropy_loss(logits, targets, masked):
    """Calculates mean crosse-ntopy loss over positions, which have 0 in the mask"""
    loss_values = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
    loss_values[masked.view(-1)] = 0
    return loss_values.sum() / (~masked).sum()


class Tagger(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    def forward(self, encoder_states, tagger_inputs):
        pass
    
    def predict_tags(self, encoder_states, tagger_inputs, mask_tensor, target_vocab):
        """
        Implements basic tag prediction using logits from forward pass.
        """
        self.eval()
        result = []
        with torch.no_grad():
            logits = self(encoder_states, tagger_inputs)
            seqs = logits.argmax(dim=-1)
            for i,pred in enumerate(seqs):
                pred = pred[mask_tensor[i]].tolist()
                result.append(target_vocab.transform_ids(pred))
        return result
    
class Encoder(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    def forward(self, encoder_inputs):
        """
        Encoder's forward pass
        Args:
            encoder_inputs: Tuple - tuple of encoder inputs
        Outputs:
            whatever fits into a downstream model
        """
    
class BiLSTMTagger(Tagger):
    def __init__(self, config):
        super().__init__()
        if config.emb_dropout_type == 'plain':
            self.emb_dropout = nn.Dropout(p=config.emb_dropout)
        elif config.emb_dropout_type == 'locked':
            self.emb_dropout = flair.nn.LockedDropout(config.emb_dropout)
        else:
            raise ValueError('Invalid dropout type')
        
        self.rnn = nn.LSTM(config.embedding_dim,
                           config.bilstm_hidden_size // 2,
                           batch_first=True,
                           bidirectional=True, 
                           num_layers = config.bilstm_n_layers, 
                           dropout = config.bilstm_dropout)
        
        self.pre_fc_dropout = nn.Dropout(config.pre_fc_dropout)
        self.fc = nn.Linear(config.bilstm_hidden_size, config.n_classes) 
        
    def forward(self, states, tagger_inputs):
        """
        Produces tag logits for every position in the batch
        Args: 
            states(FloatTensor): Tensor containing encoder states or word embeddings
            tagger_inputs (Tuple: (length_tensor, )): Tuple of one element, containing lengths of every sequence in the batch
        Output:
            logits(FloatTensor): Tensor of tag logits
        Shape:
            states: (batch_size, padded_seq_len, encoder_state_dim)
            length_tensor: (batch_size,)
            logits: (batch_size, padded_seq_len, n_tags)
        """
        states = self.emb_dropout(states)
        length_tensor = tagger_inputs[0]
        
        padding_length = states.size(1)
        lstm_input = torch.nn.utils.rnn.pack_padded_sequence(states, length_tensor, batch_first=True, enforce_sorted=False)
        hidden_states, _ = self.rnn(lstm_input)
        hidden_states = torch.nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=True, total_length=padding_length)[0]
        hidden_states = self.pre_fc_dropout(hidden_states)
        
        logits = self.fc(hidden_states)
        return logits
    
class TaggerWithEncoder(nn.Module):
    def __init__(self, encoder, tagger):
        super().__init__()
        self.encoder = encoder
        self.tagger = tagger
        
    def encoder_forward(self, encoder_inputs):
        states = self.encoder(encoder_inputs)
        return states
    
    def forward(self, encoder_inputs, tagger_inputs):
        encoder_states = self.encoder_forward(encoder_inputs)
        logits = self.tagger(encoder_states, tagger_inputs)
        return logits
    
    def predict_tags(self, encoder_inputs, tagger_inputs, mask_tensor, target_vocab):
        self.eval()
        with torch.no_grad():
            states = self.encoder_forward(encoder_inputs)
            result = self.tagger.predict_tags(states, tagger_inputs, mask_tensor, target_vocab)
        return result
    