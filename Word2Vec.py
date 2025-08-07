import 

file = open("./Books/HP1.txt","r")
book = ""
for i in range(5):
    book += file.readline()
print(book)
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


print(words[:20])
print(unique_words)


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



print(dataset["the"])