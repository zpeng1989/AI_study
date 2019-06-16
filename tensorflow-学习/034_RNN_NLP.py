import tensorflow_datasets as tfds


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info = True, as_supervised = True)

train_dataset, test_dataset = dataset['train'], dataset['test']

tokenizer = info.features['text'].encoder

print(tokenizer.vocab_size)


sample_string = 'Hello world, Tensorflow, new world'
tokenized_string = tokenizer.encoder(sample_string)
print(tokenized_string)

sample_string = 'Hello word , Tensorflow'
tokenized_string = tokenizer.encode(sample_string)
print('tokened id: ', tokenized_string)

# 解码会原字符串
src_string = tokenizer.decode(tokenized_string)
print('original string: ', src_string)

for t in tokenized_string:
    print(str(t) + '->', tokenizer.encoder([t]))


BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BATCH_SIZE)
train_dataset = train_dataset.padding_batch(BATCH_SIZE, train_dataset.output_shapes)
test_dataset = test_dataset.padding_batch(BATCH_SIZE, test_dataset.output_shapes)


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation ='relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    return model

model = get_model()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(train_dataset, epochs = 10, validation_data = test_dataset)
test_loss, test_acc = model.evaluate(test_dataset)

print(test_loss)
print(test_acc)

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad = False):
    tokened_sent = tokenizer.encoder(sentence)
    if pad:
        tokened_sent = pad_to_size(tokened_sent, 64)
    pred = model.predict(tf.expand_dims(tokened_sent, 0))
    return pred


sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)







