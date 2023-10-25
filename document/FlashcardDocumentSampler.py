from .FlashcardDocument import FlashcardDocument
from recsim import document

class FlashcardDocumentSampler(document.AbstractDocumentSampler):
  def __init__(self, doc_ctor=FlashcardDocument, **kwargs):
    super(FlashcardDocumentSampler, self).__init__(doc_ctor, **kwargs)
    self._doc_count = 0

  def sample_document(self):
    doc_features = {}
    doc_features['doc_id'] = self._doc_count
    doc_features['difficulty'] = self._rng.random_sample((1, 3))
    self._doc_count += 1
    return self._doc_ctor(**doc_features)