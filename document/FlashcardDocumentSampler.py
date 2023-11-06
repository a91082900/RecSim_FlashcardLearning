from .FlashcardDocument import FlashcardDocument
from recsim import document
import numpy as np

class FlashcardDocumentSampler(document.AbstractDocumentSampler):
  EASY_CARD = np.array([1.0, 5.0, 2.5])
  HARD_CARD = np.array([1.0, 1.5, 0.75])
  def __init__(self, doc_ctor=FlashcardDocument, seed=0, **kwargs):
    super(FlashcardDocumentSampler, self).__init__(doc_ctor, seed, **kwargs)
    self._doc_count = 0

  def sample_document(self):
    doc_features = {}
    doc_features['doc_id'] = self._doc_count
    is_hard = self._rng.binomial(1, 0.5)
    difficulty = (FlashcardDocumentSampler.HARD_CARD if is_hard 
                  else FlashcardDocumentSampler.EASY_CARD) + self._rng.uniform(-0.5, 0.5, (1, 3))
    doc_features['difficulty'] = np.array(difficulty)
    self._doc_count += 1
    return self._doc_ctor(**doc_features)