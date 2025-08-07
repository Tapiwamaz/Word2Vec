import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

file = open("./Books/HP1.txt","r")
book = ""
for i in range(10):
    book += file.readline()
# print(book)
words = book.lower()
new_book =""

for char in words:
    ascii_val = ord(char)
    if ((48 <= ascii_val <= 57) or (97 <= ascii_val <= 122)) or char == " ":
        new_book+= char

words = new_book.split(" ")

# Remove words that contain characters outside ASCII ranges 48-57 (0-9) and 97-122 (a-z)
# filtered_words = []

unique_words = {}
index = 0
for word in words:
    if word not in unique_words.keys():
        unique_words[word] = index
        index += 1
    else:
        continue


# print(words[:20])
# print(unique_words)?


dataset = {}
window_size = 2

for i in range(len(words)):
    central_word = words[i]
    
    if not central_word:  # Skip empty words
        continue
    
    # Initialize array if key doesn't exist
    if central_word not in dataset:
        dataset[central_word] = []
    
    # Get context words (2 words on each side)
    start_idx = max(0, i - window_size)
    end_idx = min(len(words), i + window_size + 1)
    
    # Collect context words for this occurrence of central_word
    context_words = []
    for j in range(start_idx, end_idx):
        if j != i and words[j]:  # Skip the central word itself and empty words
            context_words.append(words[j])
    
    # Append all context words to the array for this central word
    dataset[central_word].extend(context_words)

# Convert dictionary dataset to list of (input_idx, context_idx) pairs for training
training_pairs = []
for central_word, context_words in dataset.items():
    central_idx = unique_words[central_word]
    for context_word in context_words:
        if context_word in unique_words:
            context_idx = unique_words[context_word]
            training_pairs.append((central_idx, context_idx))

print(training_pairs[:10])
print(f"Total training pairs: {len(training_pairs)}")
print(f"Vocabulary size: {len(unique_words)}")

# Define Word2Vec Skip-gram model
class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecSkipGram, self).__init__()
        # Input embedding layer (center word)
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output embedding layer (context word prediction)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, center_word, context_word):
        # Get embeddings
        center_embed = self.input_embeddings(center_word)  # [batch_size, embedding_dim]
        context_embed = self.output_embeddings(context_word)  # [batch_size, embedding_dim]
        
        # Calculate dot product (similarity score)
        score = torch.sum(center_embed * context_embed, dim=1)  # [batch_size]
        
        return score

# Custom Dataset class for Word2Vec
class Word2VecDataset(Dataset):
    def __init__(self, training_pairs):
        self.pairs = training_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center_idx, context_idx = self.pairs[idx]
        return torch.tensor(center_idx, dtype=torch.long), torch.tensor(context_idx, dtype=torch.long)

# Hyperparameters
vocab_size = len(unique_words)
embedding_dim = 100  # Size of word embeddings
learning_rate = 0.01
batch_size = 64
epochs = 10

# Create model, dataset, and dataloader
model = Word2VecSkipGram(vocab_size, embedding_dim)
word2vec_dataset = Word2VecDataset(training_pairs)
dataloader = DataLoader(word2vec_dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training...")

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (center_words, context_words) in enumerate(dataloader):
        
        # Forward pass - positive samples
        positive_scores = model(center_words, context_words)
        positive_labels = torch.ones_like(positive_scores)
        
        # Generate negative samples (random context words)
        number_of_negative = torch.Size([32])
        # print(context_words.shape)
        negative_context = torch.randint(0, vocab_size, context_words.shape, dtype=torch.long)
        # print(negative_context)
        negative_scores = model(center_words, negative_context)
        negative_labels = torch.zeros_like(negative_scores)
        
        # Combine positive and negative samples
        all_scores = torch.cat([positive_scores, negative_scores])
        all_labels = torch.cat([positive_labels, negative_labels])
        
        # Calculate loss
        loss = criterion(all_scores, all_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

print("Training completed!")

# Extract word embeddings
word_embeddings = model.input_embeddings.weight.data.numpy()
print(f"Word embeddings shape: {word_embeddings.shape}")

# Function to get word embedding
def get_word_embedding(word):
    if word in unique_words:
        idx = unique_words[word]
        return word_embeddings[idx]
    else:
        return None

# Test with some words
test_words = ["mr", "mrs", "and", "to"]
for word in test_words:
    embedding = get_word_embedding(word)
    if embedding is not None:
        print(f"Embedding for '{word}': {embedding[:5]}...")  # Show first 5 dimensions
    else:
        print(f"Word '{word}' not in vocabulary")

# print(dataset["the"])

mr = get_word_embedding("mr")
mrs = get_word_embedding("mrs")
# queen=  get_word_embedding("queen")
# queens = get_word_embedding("queens")
print(np.dot(mr,mrs))
# print(np.dot(harry,potter))

