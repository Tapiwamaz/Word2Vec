import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time 

file = open("./Books/HP1.txt","r")
book = ""
for i in range(153):
    if i >= 102:
        book += file.readline()
    else:   
        file.readline()
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


dataset = {}
window_size = 1

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
        
    def forward(self, center_word, target_words):
        # Get embeddings
        center_embed = self.input_embeddings(center_word)  # [batch_size, embedding_dim]
        target_embed = self.output_embeddings(target_words)  # [batch_size, num_targets, embedding_dim]
        
        # Calculate dot product scores
        # center_embed: [batch_size, embedding_dim] -> [batch_size, 1, embedding_dim]
        # target_embed: [batch_size, num_targets, embedding_dim]
        scores = torch.bmm(center_embed.unsqueeze(1), target_embed.transpose(1, 2))  # [batch_size, 1, num_targets]
        scores = scores.squeeze(1)  # [batch_size, num_targets]
        
        return scores

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
learning_rate = 0.005
batch_size = 64
num_negative_samples = 1  # Number of negative samples per positive sample

# Create model, dataset, and dataloader
model = Word2VecSkipGram(vocab_size, embedding_dim)
word2vec_dataset = Word2VecDataset(training_pairs)
dataloader = DataLoader(word2vec_dataset, batch_size=batch_size, shuffle=True)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training...")


epoch =0
loss = float('inf')
start_time = time.time()
while loss > 0.5:
    total_loss = 0

    for batch_idx, (center_words, context_words) in enumerate(dataloader):
        batch_size_actual = center_words.size(0)
        
        # Generate negative samples
        negative_samples = torch.randint(0, vocab_size, (batch_size_actual, num_negative_samples), dtype=torch.long)
        
        # Combine positive and negative samples
        # positive: [batch_size, 1], negative: [batch_size, num_negative_samples]
        all_targets = torch.cat([context_words.unsqueeze(1), negative_samples], dim=1)  # [batch_size, 1 + num_negative_samples]
        
        # Forward pass
        scores = model(center_words, all_targets)  # [batch_size, 1 + num_negative_samples]
        
        # Apply log softmax for numerical stability
        log_probs = torch.log_softmax(scores, dim=1)  # [batch_size, 1 + num_negative_samples]
        
        # The positive sample is always at index 0, so we want to maximize its probability
        # Negative log likelihood: we want to minimize -log(P(positive))
        loss_epoch = -log_probs[:, 0].mean()  # Take negative log prob of positive samples
        
        # Backward pass
        optimizer.zero_grad()
        loss_epoch.backward()
        optimizer.step()
        
        total_loss += loss_epoch.item()
        
        
    loss = total_loss / len(dataloader)

    epoch +=1

end_time = time.time()  # End timing
total_training_time = end_time - start_time

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


# ron = get_word_embedding("ron")
# harry = get_word_embedding("harry")
# # queen=  get_word_embedding("queen")
# # queens = get_word_embedding("queens")
# print(np.dot(ron,harry))
# # hermione, ron, potter, magic, hogwarts, dumbledore


word_pairs = [
    # ("hermione", "ron"),
    # ("harry", "potter"),
    ("magic", "hogwarts"),
    # ("ron", "harry"),
    # ("dumbledore", "hogwarts"),
    # ("hermione", "potter")
]

dot_products = {}

for word1, word2 in word_pairs:
    emb1 = get_word_embedding(word1)
    emb2 = get_word_embedding(word2)
    
    if emb1 is not None and emb2 is not None:
        dot_prod = np.dot(emb1, emb2)
        dot_products[f"{word1}_{word2}"] = dot_prod


# Save results to CSV with dot products
file = open("Similarity vs Samples4.csv", "w+")
file.write("Negative_Samples,Similarity\n")


# for pair_name in dot_products.keys():
#     file.write(f"{pair_name}, ")

# file.write("\n")

# Write the data
file.write(f"{num_negative_samples}")

# Add dot product values
for dot_prod in dot_products.values():
    file.write(f" ,{dot_prod:.4f}")

file.write("\n")
file.close()
