import tensorflow_datasets as tfds


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info = True, as_supervised = True)