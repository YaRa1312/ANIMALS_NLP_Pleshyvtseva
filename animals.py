# import json
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier

# # read data from JSON file
# with open('D:\jsonformatter.json', encoding='utf-8') as f:
#     data = json.load(f)

# # convert JSON data to Pandas DataFrame
# df = pd.DataFrame(data)
# print (df)
# BEFORE = df.drop(columns=["ПІСЛЯ"])
# AFTER = df["ПІСЛЯ"]

# # train Decision Tree model
# animals_model = DecisionTreeClassifier()
# animals_model.fit(BEFORE, AFTER)

# # make predictions
# predictions = animals_model.predict([
#     ["Кі́нь сві́йський (Equus ferus caballus або Equus caballus) — підвид ссавців виду кінь дикий роду кінь (Equus) родини Коневих (Equidae) ряду Конеподібних (Equiformes), або непарнопалих (Perissodactyla)."],
#     ["Поширений на всіх континентах, крім Антарктиди, в більшості країн світу."]
# ])


###############################################################################################################################
# # # display predictions
# # print(predictions)
# import json
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction.text import CountVectorizer

# # read data from JSON file
# with open('C:\\ANIMALS PROJECT\\jsonformatter.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # convert JSON data to Pandas DataFrame
# df = pd.DataFrame(data)
# print (df)
# # use CountVectorizer to convert text data into numerical data
# vectorizer = CountVectorizer()
# BEFORE = vectorizer.fit_transform(df["ДО"])
# #print (df.drop(columns=["ПІСЛЯ"]))
# #AFTER = df.drop(columns=["ПІСЛЯ"])

# # get labels for the prediction task
# AFTER = df["ПІСЛЯ"]
# print (AFTER)
# # train Decision Tree model
# animals_model = DecisionTreeClassifier()
# print (animals_model.fit(BEFORE, AFTER))

# #print( animals_model.predict(['кінь', 'корова']))

# # make predictions
# new_texts = [
#     "Меду́зи (Medusozoa) — підтип вільноплавних морських тварин типу кнідарії, які мають драглисте тіло, що складається з шапки у формі парасолі та тягучих помацків.До підтипу належать класи Сцифоїдних (понад 200 видів), Ставромедуз (близько 50 видів), Кубомедуз (близько 20 видів) і Гідроїдних (близько 1000—1500 видів)."
# ]
# # new_texts = input("Введіть текст про тварин обсягом до 10 речень включно: ")
# new_texts_encoded = vectorizer.transform(new_texts)
# # print (new_texts_encoded)
# predictions = animals_model.predict(new_texts_encoded)

# print(predictions)
###############################################################################################################################







###############################################################################################################################

# !pip install transformers
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score

# Вбудований список стоп-слів
stop_words = [
    'а', 'і', 'в', 'на', 'з', 'до', 'як', 'про', 'також', 'та',
    'їх', 'його', 'але', 'або', 'у', 'від', 'по', 'не', 'за',
    'бути', 'це', 'все', 'що', 'якщо', 'коли', 'де', 'чи', 'ні',
    'той', 'цей', 'та', 'так', 'такий', 'така', 'таке', 'такі',
    'інший', 'інша', 'інше', 'інші', 'сам', 'сама', 'саме', 'самі',
    'свій', 'своя', 'своє', 'свої'
]

# Відкриття JSON-файлу та зчитування даних
with open('C:\\ANIMALS PROJECT\\jsonformatter.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

before_list = [row['ДО'] for row in data]
after_list = [row['ПІСЛЯ'] for row in data]

# Створення DataFrame зі списків
df = pd.DataFrame({'ДО': before_list, 'ПІСЛЯ': after_list})

# Виконуємо попередню обробку тексту (наприклад, перетворення на нижній регістр)
df['ДО'] = df['ДО'].str.lower()
df['ПІСЛЯ'] = df['ПІСЛЯ'].str.lower()

# Токенізація даних з колонки 'ДО'
sentences_before = df['ДО'].apply(sent_tokenize)

# Злиття вкладених списків у плоский список
sentences_before_flat = [sentence for sublist in sentences_before for sentence in sublist]

# Токенізація та видалення стоп-слів для кожного речення
tokenized_sentences_before = [word_tokenize(sentence) for sentence in sentences_before_flat]
filtered_sentences_before = [[word for word in sentence if word.lower() not in stop_words] for sentence in tokenized_sentences_before]

# Токенізація даних з колонки 'ПІСЛЯ'
sentences_after = df['ПІСЛЯ'].apply(sent_tokenize)

# Злиття вкладених списків у плоский список
sentences_after_flat = [sentence for sublist in sentences_after for sentence in sublist]

# Токенізація та видалення стоп-слів для кожного речення
tokenized_sentences_after = [word_tokenize(sentence) for sentence in sentences_after_flat]
filtered_sentences_after = [[word for word in sentence if word.lower() not in stop_words] for sentence in tokenized_sentences_after]

# Створення пар даних (оригінальні речення та їх спрощені версії)
combined_data = list(zip(filtered_sentences_before, filtered_sentences_after))

# Підготовка даних для моделі BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Кодування тексту за допомогою токенізатора BERT
encoded_data = tokenizer.batch_encode_plus(
    combined_data,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512,
    return_tensors='pt'
)

# Перетворення міток класів в числовий формат
labels = LabelEncoder().fit_transform(df['ПІСЛЯ'])

input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
min_length = min(len(input_ids), len(labels), len(attention_masks))
input_ids = input_ids[:min_length]
labels = labels[:min_length]
attention_masks = attention_masks[:min_length]

# Розділення даних на тренувальну, тестову та валідаційну вибірки
train_inputs, val_test_inputs, train_labels, val_test_labels, train_masks, val_test_masks = train_test_split(
    input_ids, labels, attention_masks, random_state=42, test_size=0.37, shuffle=True
)

val_inputs, test_inputs, val_labels, test_labels, val_masks, test_masks = train_test_split(
    val_test_inputs, val_test_labels, val_test_masks, random_state=42, test_size=0.5, shuffle=True
)

# Перетворення даних у формат PyTorch
train_inputs = torch.tensor(train_inputs)
val_inputs = torch.tensor(val_inputs)
test_inputs = torch.tensor(test_inputs)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)
train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)
test_masks = torch.tensor(test_masks)

# Створення датасетів та даталоадерів для тренування, валідації та тестування
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=16)
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=16)

# Завантаження та налаштування моделі BERT для класифікації
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Використання GPU для прискорення навчання моделі (якщо доступно)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Налаштування оптимізатора та критерію
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Тренування моделі
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch
        optimizer.zero_grad()
        # outputs = model(inputs, attention_mask=masks, labels=labels)
        outputs = model(inputs, attention_mask=masks, labels=labels.float())
        loss = outputs.loss
        logits = outputs.logits
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    avg_train_loss = train_loss / len(train_dataloader)
    print(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss}')

# Тестування моделі
model.eval()
predictions = []
true_labels = []
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    inputs, masks, labels = batch
    with torch.no_grad():
        outputs = model(inputs, attention_mask=masks)
    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = labels.to('cpu').numpy()
    predictions.extend(np.argmax(logits, axis=1).flatten())
    true_labels.extend(label_ids.flatten())

# Обчислення точності моделі
accuracy = accuracy_score(true_labels, predictions)
print(f'Test Accuracy: {accuracy}')
