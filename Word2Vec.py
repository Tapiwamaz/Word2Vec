import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time 

# ===================================================================================================================
# Function and classes
def get_word_embedding(word,word_embeddings):
    if word in unique_words:
        idx = unique_words[word]
        return word_embeddings[idx]
    else:
        return None
    
class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecSkipGram, self).__init__()
        # Input embedding layer (center word)
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output embedding layer (context word prediction)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, center_word, target_words):
        center_embed = self.input_embeddings(center_word)  
        target_embed = self.output_embeddings(target_words)  
        
        # Calculate dot product scores
        scores = torch.bmm(center_embed.unsqueeze(1), target_embed.transpose(1, 2)) 
        scores = scores.squeeze(1) 
        
        return scores

class Word2VecDataset(Dataset):
    def __init__(self, training_pairs):
        self.pairs = training_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center_idx, context_idx = self.pairs[idx]
        return torch.tensor(center_idx, dtype=torch.long), torch.tensor(context_idx, dtype=torch.long)    

# ===================================================================================================================


# ===================================================================================================================

# Preprocessing
# We decided to look at 50 pages from the 1st harry potter book
# Pages 102 to 152 (inclusive)

file = open("./Books/HP1.txt","r")
book = ""
for i in range(153):
    if i >= 102:
        book += file.readline()
    else:   
        file.readline()

words = book.lower()
new_book =""

# remove all punctuation from the text
for char in words:
    ascii_val = ord(char)
    if ((48 <= ascii_val <= 57) or (97 <= ascii_val <= 122)) or char == " ":
        new_book+= char


words = new_book.split(" ")



# We use a dictonary to get the unique words and give them unique indexes in the order they appear 
# in our subset of the book
unique_words = {}
index = 0
for word in words:
    if word and word not in unique_words.keys():
        unique_words[word] = index
        index += 1
    else:
        continue


dataset = {}
window_size = 1

for i in range(len(words)):
    central_word = words[i]
    
    # Potentially a few empty spaces in the set of non unique words
    if not central_word:  
        continue
    
    # Initialize array if key doesn't exist
    if central_word not in dataset:
        dataset[central_word] = []
    
    # Get context words and deal with bounds
    start_idx = max(0, i - window_size)
    end_idx = min(len(words), i + window_size + 1)
    
    # Collect all context words for this occurrence of central_word
    context_words = []
    for j in range(start_idx, end_idx):
        if j != i and words[j]: 
            context_words.append(words[j])
    
    
    dataset[central_word].extend(context_words)
    dataset[central_word] = list(set(dataset[central_word]))

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

# ===================================================================================================================


# Hyperparameters
vocab_size = len(unique_words)
embedding_dim = 100  # Size of word embeddings
learning_rate = 0.005
batch_size = 64
num_negative_samples = 1  # Number of negative samples per positive sample (What we change)

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
    #Convergence is at an average loss of < 0.5 for an epoch  
    total_loss = 0

    for batch_idx, (center_words, context_words) in enumerate(dataloader):
        batch_size_actual = center_words.size(0)
        
        # Generate negative samples (indexes)
        negative_samples = torch.randint(0, vocab_size, (batch_size_actual, num_negative_samples), dtype=torch.long)
        
        # Combine positive and negative samples
        # positive: [batch_size, 1], negative: [batch_size, num_negative_samples]
        all_targets = torch.cat([context_words.unsqueeze(1), negative_samples], dim=1)  
        scores = model(center_words, all_targets)  
        
        # Apply log softmax 
        log_probs = torch.log_softmax(scores, dim=1)  
        
        # The positive sample is always at index 0, so we want to maximize its probability
        # Negative log likelihood means we want to minimize -log(P(positive))
        loss_epoch = -log_probs[:, 0].mean()  # Take negative log prob of positive samples
        
        # Backward pass
        optimizer.zero_grad()
        loss_epoch.backward()
        optimizer.step()
        
        total_loss += loss_epoch.item()
        
        
    loss = total_loss / len(dataloader)

    epoch +=1

end_time = time.time() 
total_training_time = end_time - start_time

print("Training completed!")

word_embeddings = model.input_embeddings.weight.data.numpy()



# These are our chosen pairs with which we will determine how the embedding space is changing with different 
# number of negative samples
word_pairs = [
    # ("hermione", "ron"),
    # ("harry", "potter"),
    # ("magic", "hogwarts"),
    ("ron", "harry"),
    # ("dumbledore", "hogwarts"),
    # ("hermione", "potter")
]

word1, word2 = word_pairs[0]
embedding1, embedding2 = get_word_embedding(word1,word_embeddings) , get_word_embedding(word2,word_embeddings)

dot_prod = np.dot(embedding1, embedding2)
print(dot_prod)


# # Save results to CSV with dot products
# file = open("Similarity vs Samples4.csv", "w+")
# file.write("Negative_Samples,Similarity\n")

# file.write(f"{num_negative_samples},{dot_prod:.4f}")

# file.write("\n")
# file.close()
