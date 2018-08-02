
import numpy as np
import tensorflow as tf

from options import load_arguments
from vocab import Vocabulary, build_vocab
from style_tranfer_model import Model
from accumulator import Accumulator
from file_io import load_sent, write_sent
from utils import *
import beam_search, greedy_decoding


class ModelLoader(object):
    
    def __init__(self, input_path, input_text):
        self.vocab_path = input_path
        self.modern_text = input_text
    
    def toShakespeare(self):
        """Given a line of text, return that text in the indicated style.
        
        Args:
          modern_text: (string) The input.
          
        Returns:
          string: The translated text, if generated.
        """ 
        
        args = load_arguments()
        vocab = Vocabulary(self.vocab_path, args.embedding, args.dim_emb)
            
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
            
        with tf.Session(config=config) as sess:
            model = Model(args, vocab)
            model.saver.restore(sess, args.model)

            if args.beam > 1:
                decoder = beam_search.Decoder(sess, args, vocab, model)
            else:
                decoder = greedy_decoding.Decoder(sess, args, vocab, model)
                    
                batch = get_batch([self.modern_text], [1], vocab.word2id)
                ori, tsf = decoder.rewrite(batch)
            
                out = ' '.join(w for w in tsf[0])
            
        return out