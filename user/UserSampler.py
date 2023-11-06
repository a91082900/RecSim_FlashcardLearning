from .UserState import UserState
from recsim import user
import numpy as np

class UserSampler(user.AbstractUserSampler):
  def __init__(self,
               user_ctor=UserState,
               num_candidates=10,
               time_budget=60,
               **kwargs):
    super(UserSampler, self).__init__(user_ctor, **kwargs)
    W = np.zeros((num_candidates, 3))
    W[:, 0] = 1
    W[:, 1] = self._rng.uniform(1.5, 5, num_candidates)
    W[:, 2] = self._rng.uniform(0.75, 2.5, num_candidates)
    self._state_parameters = {
      'num_candidates': num_candidates, 
      'time_budget': time_budget,
      'W': W
    }

  def sample_user(self):
    return self._user_ctor(**self._state_parameters)