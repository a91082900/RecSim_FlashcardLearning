from .UserState import UserState
from recsim import user

class UserSampler(user.AbstractUserSampler):
  def __init__(self,
               user_ctor=UserState,
               num_candidates=10,
               time_budget=60,
               **kwargs):
    super(UserSampler, self).__init__(user_ctor, **kwargs)
    doc_error = self._rng.uniform(0.5, 1.5, (num_candidates, 3))
    self._state_parameters = {
      'num_candidates': num_candidates, 
      'time_budget': time_budget,
      'doc_error': doc_error
    }

  def sample_user(self):
    return self._user_ctor(**self._state_parameters)