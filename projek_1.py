import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Contoh dataset pertanyaan dan jawaban
questions = ["What's your name?",
             "How are you?",
             "What can you do?"
             ,"Your name?",
             "Hello",
             "How are you",
             "Are you Chatbot",
             "Your name Chatbot?"
             ]

answers = ["I'm a chatbot.",
           "I'm fine, thank you.",
           "I can answer questions.",
           "Im a Chatbot",
           "Hello too",
           "I'm just a Chatbot",
           "Yes i'm a Chatbot",
           "Yes"
           ]

# Gabungkan pertanyaan dan jawaban menjadi satu teks
corpus = questions + answers

# Tokenisasi teks
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Membuat dataset pelatihan
input_sequences = []
for question in questions:
    sequence = tokenizer.texts_to_sequences([question])[0]
    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i+1])

# Padding sequences
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Pisahkan input dan output
X = input_sequences[:, :-1]
Y = input_sequences[:, -1]
Y = tf.keras.utils.to_categorical(Y, num_classes=total_words)

# Buat model
model = keras.Sequential([
    keras.layers.Embedding(total_words, 100, input_length=max_sequence_length-1),
    keras.layers.LSTM(150),
    keras.layers.Dense(total_words, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Latih model
model.fit(X, Y, epochs=200, verbose=1)

# Fungsi untuk menghasilkan jawaban
def generate_response(seed_text):
    seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    for _ in range(10):  # Prediksi 10 kata berikutnya
        padded_sequence = pad_sequences([seed_sequence], maxlen=max_sequence_length-1, padding='pre')
        predicted_word_index = np.argmax(model.predict(padded_sequence), axis=-1)
        predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_word_index][0]
        seed_text += " " + predicted_word
        seed_sequence.append(predicted_word_index[0])
    return seed_text

# Input pengguna menjadi variabel
user_input = "What's your name?"

# Memanggil fungsi untuk menghasilkan respons chatbot
response = generate_response(user_input)
print(response)
