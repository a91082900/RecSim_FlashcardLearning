from recsim import document
from gym import spaces
import numpy as np

class FlashcardDocument(document.AbstractDocument):
  def __init__(self, doc_id, difficulty):
    self.base_difficulty = difficulty
    # doc_id is an integer representing the unique ID of this document
    super(FlashcardDocument, self).__init__(doc_id)

  def create_observation(self):
    return np.array(self.base_difficulty)

  @staticmethod
  def observation_space():
    return spaces.Box(shape=(1,3), dtype=np.float32, low=0.0, high=1.0)

  def __str__(self):
    return "Flashcard {} with difficulty {}.".format(self._doc_id, self.base_difficulty)
