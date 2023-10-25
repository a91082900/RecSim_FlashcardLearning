from recsim import user
from gym import spaces

class UserResponse(user.AbstractResponse):
  def __init__(self, recall=False, pr=0):
    self._recall = recall
    self._pr = pr

  def create_observation(self):
    return {'recall': int(self._recall), 'pr': self._pr}

  @classmethod
  def response_space(cls):
    # return spaces.Discrete(2)
    return spaces.Dict({'recall': spaces.Discrete(2), 'pr': spaces.Box(low=0.0, high=1.0)})
