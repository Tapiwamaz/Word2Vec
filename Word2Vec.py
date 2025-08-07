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
dataset  = {}

for word in words:
    dataset[word]
