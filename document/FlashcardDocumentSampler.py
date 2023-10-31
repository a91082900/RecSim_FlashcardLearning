from .FlashcardDocument import FlashcardDocument
from recsim import document
import numpy as np

class FlashcardDocumentSampler(document.AbstractDocumentSampler):
  def __init__(self, doc_ctor=FlashcardDocument, seed=0, **kwargs):
    super(FlashcardDocumentSampler, self).__init__(doc_ctor, seed, **kwargs)
    self._doc_count = 0

  def sample_document(self):
    doc_features = {}
    doc_features['doc_id'] = self._doc_count
    difficulty = [1, self._rng.uniform(1.5, 5), self._rng.uniform(0.75, 2.5)]
    doc_features['difficulty'] = np.array(difficulty)
    self._doc_count += 1
    return self._doc_ctor(**doc_features)