import logging
import sys
import signal
import traceback
from contextlib import contextmanager
from os.path import join as pjoin

import sentencepiece as spm
import torch

from sources.drrn.memory import State
from sources.drrn.model import DRRN

import sources.drrn.model as model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
#   Timeout additions (from https://www.jujens.eu/posts/en/2018/Jun/02/python-timeout-function/ )
#
@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    print("Timeout")
    raise TimeoutError


class DRRN_Agent:
    def __init__(self,
                 spm_path: str,
                 gamma: float = .9,
                 batch_size: int = 64,
                 embedding_dim: int = 128,
                 hidden_dim: int = 128,
                 clip: float = 5,
                 learning_rate: float = 0.0001):
        self.gamma = gamma
        self.batch_size = batch_size
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        self.network = DRRN(len(self.sp), embedding_dim, hidden_dim).to(device)

        self.clip = clip
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # Error catching when saving the model
        self.lastSaveSuccessful = True
        self.numSaveErrors = 0

    def build_state(self, obs, inv, look):
        """ Returns a state representation built from various info sources. """
        obs_ids = self.sp.EncodeAsIds(obs)
        # TextWorld
        look_ids = self.sp.EncodeAsIds(inv)
        inv_ids = self.sp.EncodeAsIds(look)
        return State(obs_ids, look_ids, inv_ids)

    def encode(self, obs_list):
        """ Encode a list of observations """
        return [self.sp.EncodeAsIds(o) for o in obs_list]

    def act(self, states, poss_acts, sample=True):
        """ Returns a string action from poss_acts. """
        idxs, values = self.network.act(states, poss_acts, sample)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, values

    def load(self, path, suffixStr=""):
        print("Loading agent from path: " + str(path))
        try:
            sys.modules['model'] = model
            self.network = torch.load(pjoin(path, "model" + suffixStr + ".pt"))
        except Exception as e:
            print("Error loading model.")
            logging.error(traceback.format_exc())
