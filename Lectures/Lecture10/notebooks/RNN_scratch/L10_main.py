'''
Title: RNN "Hello World"
The original code used in this project was adapted from 
“A Recurrent Neural Network (RNN) From Scratch”, 
available on GitHub: https://github.com/vzhou842
'''

# import libraries
import numpy as np
import random

from rnn import RNN
from data import train_data, test_data

# Create the vocabulary.
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('-' * 42)
print('%d unique words found' % vocab_size)
print('-' * 42)

# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }

for idx in sorted(idx_to_word.keys()):
    print(f'{idx} -> {idx_to_word[idx]}')

def createInputs(text):
  '''
  Returns an array of one-hot vectors representing the words in the input text string.
  - text is a string
  - Each one-hot vector has shape (vocab_size, 1)
  '''
  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs

def softmax(xs):
  # Applies the Softmax Function to the input array.
  return np.exp(xs) / sum(np.exp(xs))


def show_one_hot_table(vocab, word_to_idx, num_words=10):
    '''
    Shows one hot encoding table
    '''
    print('-' * 60)
    print(f'{"Word":<15} {"Idx":<5} {"One-Hot (first 10 positions)":<30}')
    print('-' * 60)

    for word in vocab[:num_words]:
        idx = word_to_idx[word]
        one_hot = np.zeros(vocab_size)
        one_hot[idx] = 1

        # showthe first 10 values 
        preview = ' '.join(map(str, one_hot[:10].astype(int)))

        print(f'{word:<15} {idx:<5} [{preview} ...]')

    print('-' * 60)

# call table
show_one_hot_table(vocab, word_to_idx, num_words=10)


# Initialize RNN!
rnn = RNN(vocab_size, 2)

def processData(data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping text to True or False.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = createInputs(x)
    target = int(y)

    # Forward
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Calculate loss / accuracy
    loss -= np.log(probs[target].item())
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      rnn.backprop(d_L_d_y)

  return loss / len(data), num_correct / len(data)

# Training loop
for epoch in range(1000):
  train_loss, train_acc = processData(train_data)

  if epoch % 100 == 99:
    print('--- Epoch %d' % (epoch + 1))
    print("train_loss:", type(train_loss), np.shape(train_loss))
    print("train_acc :", type(train_acc), np.shape(train_acc))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

    test_loss, test_acc = processData(test_data, backprop=False)
    print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))


def predict(text):
    inputs = createInputs(text)
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    pred = np.argmax(probs)
    confidence = probs[pred].item()

    print('-' * 50)
    print(f'Text: "{text}"')
    print(f'Probabilities: {probs.ravel()}')
    print(f'Predicted class: {pred}')
    print(f'Confidence: {confidence:.4f}')

    if pred == 0:
        print('Sentiment: Negative')
    else:
        print('Sentiment: Positive')
    print('-' * 50)


predict("this is good")
predict("this is bad")
predict("i love this movie")
predict("i hate this")