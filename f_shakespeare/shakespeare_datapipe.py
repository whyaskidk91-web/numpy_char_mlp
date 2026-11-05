import numpy as np

text = open("shakespeare.txt").read() #1Million characters of shakespeare text
chars = sorted(set(text)) #used characters
vocab_size = len(chars)
stoi = {c: a for a,c in enumerate(chars)} #string to int
itos = {a: c for a,c in enumerate(chars)} #int to string
ctx = 16 #context size
data = np.array([stoi[c] for c in text]) #better computable
cut = int(len(text) * 0.9)
#10% are used for test data
train,test = data[:cut],data[cut:]
print("Import successful\n")





