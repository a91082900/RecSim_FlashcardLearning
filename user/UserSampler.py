from .UserState import UserState
from recsim import user

class UserSampler(user.AbstractUserSampler):
  def __init__(self,
               user_ctor=UserState,
               num_candidates=10,
               time_budget=60,
               **kwargs):
    self._state_parameters = {'num_candidates': num_candidates, 'time_budget': time_budget}
    super(UserSampler, self).__init__(user_ctor, **kwargs)


  def sample_user(self):
    return self._user_ctor(**self._state_parameters)