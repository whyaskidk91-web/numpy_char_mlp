import random as r

names = open("names.txt").read().split()
#32k names, mostly ordered alphabetically
r.seed(42)
r.shuffle(names)
chars = ['.'] + sorted(set(''.join(names)))
cut = int(len(names) * 0.9)
#10% of data are for test
names,test_samples = names[:cut],names[cut:]
print("Import successful\n")
stoi = {c: a for a,c in enumerate(chars)} #string to int
itos = {a: c for a,c in enumerate(chars)} #int to string
ctx = 7 #context size

def create_windows(sample):
  cX,cY = [],[]
  for name in sample:
    my_name = "." * ctx + name + "."
    i_name = [stoi[s] for s in my_name]
    for c in range(len(my_name) - ctx):
      cX.append(i_name[c:c + ctx])#WINDOW
      cY.append(i_name[c + ctx])#TARGET
  return cX,cY

train_X,train_Y = create_windows(names)
test_X,test_Y = create_windows(test_samples)




