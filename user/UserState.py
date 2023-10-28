from recsim import user
import numpy as np
from gym import spaces

class UserState(user.AbstractUserState):
  def __init__(self, num_candidates, time_budget):
    self._cards = num_candidates
    self._history = np.zeros((num_candidates, 3))
    self._last_review = np.repeat(-1.0, num_candidates)
    self._time_budget = time_budget
    self._time = 0
    self._W = np.zeros((num_candidates, 3))
    super(UserState, self).__init__()
  def create_observation(self):
    return {'history': self._history, 'last_review': self._last_review, 'time': self._time, 'time_budget': self._time_budget}

  def observation_space(self): # can this work?
    return spaces.Dict({
        'history': spaces.Box(shape=(self._cards, 3), low=0, high=np.inf, dtype=int),
        'last_review': spaces.Box(shape=(self._cards,), low=0, high=np.inf, dtype=int),
        'time': spaces.Box(shape=(1,), low=0, high=np.inf, dtype=int),
        'time_budget': spaces.Box(shape=(1,), low=0, high=np.inf, dtype=int),
    })

  def score_document(self, doc_obs):
    return 1