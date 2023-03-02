import json
import random
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)['intents']

# Split dataset into training and validation sets
train_data, val_data = train_test_split(intents, test_size=0.2, random_state=42)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define training data
training_data = []
for intent in train_data:
    for pattern in intent['patterns']:
        training_data.append((pattern, intent['tag']))

# Shuffle training data
random.shuffle(training_data)

# Tokenize and encode training data
batch_size = 16
train_encodings = tokenizer([data[0] for data in training_data], truncation=True, padding=True, return_tensors='pt')

# Extract all unique intent tags from the dataset
intent_labels = list(set([data[1] for data in training_data]))

train_labels = torch.tensor([intent_labels.index(data[1]) for data in training_data])

# Load or create BERT model
try:
    model = BertForSequenceClassification.from_pretrained('bert-model')
except:
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_labels))

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Train the model
model.train()
for epoch in range(10):
    total_loss = 0
    for i in range(0, len(training_data), batch_size):
        inputs = {key: val[i:i+batch_size] for key, val in train_encodings.items()}
        labels = train_labels[i:i+batch_size]
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / (len(training_data) / batch_size)
    print(f"Epoch: {epoch+1} Average training loss: {avg_loss}")

# Save the trained model
model.save_pretrained('bert-model')

# Set the model to evaluation mode
model.eval()

#Define Confidence Threshold
confidence_threshold = 0.5

# Define a function to predict the intent of user input
def predict_intent(text):
    # Tokenize the input text and add special tokens
    inputs = tokenizer.batch_encode_plus([text], padding=True, truncation=True, return_tensors='pt')

    # Make a prediction
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(dim=1).item()

    # Get the predicted intent tag and the confidence score
    predicted_tag = intent_labels[predicted_class_idx]
    confidence_score = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()

    if confidence_score < confidence_threshold:
        predicted_tag = "unknown"

    return predicted_tag, confidence_score

# Define a function to get a random response based on intent tag
def get_response(intent):
    intent_data = next((item for item in intents if item["tag"] == intent), None)
    if intent_data:
        responses = intent_data['responses']
        return random.choice(responses)
    else:
        return "I'm sorry, I didn't understand what you meant."

# Start the chatbot
print("Bot: Hi, how can I help you today?")
while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        print("Bot: Goodbye! Get well soon :)")
        break
    else:
        intent, confidence_score = predict_intent(user_input)
        response = get_response(intent)
        print("Bot:", response)