# Імпортування потрібних бібліотек та модулів
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
# import numpy as np
# from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Вбудований список стоп-слів
stop_words = [
    'а', 'і', 'в', 'на', 'з', 'до', 'як', 'про', 'також', 'та',
    'їх', 'його', 'але', 'або', 'у', 'від', 'по', 'не', 'за',
    'бути', 'це', 'все', 'що', 'якщо', 'коли', 'де', 'чи', 'ні',
    'той', 'цей', 'та', 'так', 'такий', 'така', 'таке', 'такі',
    'інший', 'інша', 'інше', 'інші', 'сам', 'сама', 'саме', 'самі',
    'свій', 'своя', 'своє', 'свої'
]

# print (torch.cuda.is_available())
# Використання GPU для прискорення навчання моделі (якщо доступно)
device = torch.device('cuda')

# Відкриття JSON-файлу та зчитування даних
with open('C:\\ANIMALS PROJECT\\jsonformatter.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Створення списків, де елементами є комірки таблиці
before_list = [row['ДО'] for row in data]
after_list = [row['ПІСЛЯ'] for row in data]

# Створення DataFrame зі списків
df = pd.DataFrame({'ДО': before_list, 'ПІСЛЯ': after_list})

# Виконання попередньої обробки тексту (наприклад, перетворення на нижній регістр)
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
model.to(device)

# Налаштування оптимізатора та критерію
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Тренування моделі

epochs = 6
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels.float())
        loss = outputs.loss
        logits = outputs.logits
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        progress_bar.set_postfix({"Training Loss": train_loss / (len(progress_bar) + 1)})

    # Тестування моделі
    model.eval()  # Увімкнення режиму оцінювання

    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(test_dataloader, desc="Testing")
    for batch in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        with torch.no_grad():
            outputs = model(inputs, attention_mask=masks, labels=labels.float())

        loss = outputs.loss
        logits = outputs.logits
        test_loss += loss.item()

        # Підрахунок кількости правильних передбачень
        predicted_labels = logits.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    average_test_loss = test_loss / len(test_dataloader)
    accuracy = correct_predictions / total_predictions

# Виведення результатів у вигляді прогрес-барів
    print(f"Epoch {epoch+1} - Training Loss: {train_loss / (len(progress_bar) + 1):.4f}")
    print(f"Epoch {epoch+1} - Test Loss: {average_test_loss:.4f}")
    print(f"Epoch {epoch+1} - Accuracy: {accuracy*100:.2f}%")
