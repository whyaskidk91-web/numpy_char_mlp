from names_datapipe import train_X,train_Y,test_X,test_Y,names,itos,stoi,ctx
##pipeline with the names,vocabs and context size
import numpy as np

rng = np.random.default_rng(5)
#random seed

emb_size = 10 #embedding size
train_steps = 30000 #iterations
hidden_layer_size = 64
vocab_size = len(stoi) #number of letters used to generate names
batch_size = 32 #batch size
lr_base,lr = 0.01,0.01
input_size = ctx * emb_size #flattened embedding

layers = [input_size,hidden_layer_size,hidden_layer_size,vocab_size]
weights = [rng.normal(0.0,np.sqrt(2 / a),size=(a,c)) for a,c in zip(layers,layers[1:])] #includes He init sqrt(2 / a)
biases = [np.zeros(c) for c in layers[1:]]
embeddings = rng.normal(0,1,size=(vocab_size,emb_size))
mW = [np.zeros_like(W) for W in weights] #momentum for weights
vW = [np.zeros_like(W) for W in weights] #average of squared gradients for weights
mb = [np.zeros_like(bb) for bb in biases]
vb = [np.zeros_like(bb) for bb in biases]
mC = np.zeros_like(embeddings)
vC = np.zeros_like(embeddings)
beta1,beta2,eps = 0.9,0.999,1e-8
train_loss = 0
print_gap = 1000 #loss print for every 1000 runs

MY_X = np.array(train_X)
MY_Y = np.array(train_Y)
AC_X = np.array(test_X)
AC_Y = np.array(test_Y)

def softmax(z):
  z = z - z.max(axis=1, keepdims=True)
  e = np.exp(z)
  return e / e.sum(axis=1, keepdims=True)

def relu(z):
  return np.maximum(0,z)

def forward(xin):
  layer_output = [xin] #input inserted
  for i,(W,bb) in enumerate(zip(weights,biases)): 
    z = layer_output[-1] @ W + bb
    #compute neuron score
    layer_output.append(softmax(z) if i == len(weights) - 1 else relu(z))
  return layer_output
    

def backward(layer_output,target,t,win):
  global embeddings,mC,vC
  #so global variables get used without error
  gradient = target - layer_output[-1] #!negative gradient of softmax and crossentropy
  for i in reversed(range(len(weights))):
    gW = layer_output[i].T @gradient /batch_size #gradient for weights
    gb = gradient.mean(axis=0)
    if i > 0:
      gradient = (gradient @ weights[i].T)*(layer_output[i] > 0)
    #ReLU derivative: only activated neurons pass gradient back
    else: #first layer
      dx = gradient @ weights[0].T / batch_size
    mW[i] = beta1 * mW[i] + (1 - beta1) * gW
    vW[i] = beta2 * vW[i] + (1 - beta2) * gW ** 2
    mb[i] = beta1 * mb[i] + (1 - beta1) * gb
    vb[i] = beta2 * vb[i] + (1 - beta2) * gb ** 2
    weights[i] += lr * (mW[i] / (1 - beta1 ** t)) / (np.sqrt(vW[i] / (1 - beta2 ** t)) + eps) #adjust weights
    biases[i] += lr * (mb[i] / (1 - beta1 ** t)) / (np.sqrt(vb[i] / (1 - beta2 ** t)) + eps) #adjust biases
  gC = np.zeros_like(embeddings)
  np.add.at(gC,win, dx.reshape(batch_size,ctx,emb_size))
  #written differently so multiple same values get computed
  mC = beta1 * mC + (1 - beta1) * gC
  vC = beta2 * vC + (1 - beta2) * gC ** 2
  embeddings += lr * (mC / (1 - beta1 ** t)) / (np.sqrt(vC / (1 - beta2 ** t)) + eps)
    
    
    

#ANN training loop
for i in range(train_steps):
  idx = rng.integers(0.0,len(MY_X), size=batch_size)
  win = MY_X[idx]
  xin = embeddings[win].reshape(batch_size,input_size)
  target = np.eye(vocab_size)[MY_Y[idx]]
  layer_output = forward(xin)
  prob_choice = layer_output[-1][np.arange(batch_size),MY_Y[idx]]
  train_loss += -np.log(prob_choice + 1e-9).mean()
  if (i+1) % print_gap == 0:
    train_loss /= print_gap
    print(f"L:{train_loss:.2f} ", end="")
    train_loss = 0
  lr = lr_base * 0.5 * (1 + np.cos(np.pi * i / train_steps))
  #cosine learn rate
  backward(layer_output,target,i + 1,win)

def accuracy(aX,aY):
  correct = 0
  for _ in range(1000):
    idx = rng.integers(0.0,len(aX),size=batch_size)
    target = aY[idx]
    xin = embeddings[aX[idx]].reshape(batch_size,input_size)
    acc_output = forward(xin)[-1]
    correct += (target == np.argmax(acc_output,axis=1)).sum()
  return correct / (batch_size * 10)
  #prints in percentage

def loss(lX,lY):
  total_loss = 0
  for _ in range(100):
    idx = rng.integers(0.0,len(lX),size=batch_size)
    xin = embeddings[lX[idx]].reshape(batch_size,input_size)
    output_loss = forward(xin)[-1]
    likely = output_loss[np.arange(batch_size),lY[idx]]
    total_loss += -np.log(likely + 1e-9).mean()
  return total_loss / 100

#Print accuracy and loss:
print("\n\nAC train: \n", accuracy(MY_X,MY_Y))
print("AC test: \n", accuracy(AC_X,AC_Y))
print("LOSS train: \n", loss(MY_X,MY_Y))
print("LOSS test: \n", loss(AC_X,AC_Y))
print("\nNAME##########COPY###")

#generate names
for _ in range(10):
  window = [0] * ctx
  name = ""
  while True:
    probs = forward(embeddings[window].reshape(1,input_size))[-1][0]
    letter = rng.choice(vocab_size,p=probs)
    if letter == 0:
      break
    else:
      name += itos[letter]
      window = window[1:] + [letter]
    
  print(name, " " * (12 - len(name)),"•" if name in names else "=") #checks if name is remembered

#completes names with beginnings given by user
while True:
  word_input = input().lower()
  safety_check = [c for c in word_input if c not in stoi]
  if word_input == "/q":
    print("\a#COMMAND#\nLoop ended")
    break
  elif len(word_input) > ctx:
    print("\a!ERROR!\nToo long, max size:",ctx)
    continue
  elif len(safety_check) > 0:
    print("\a!ERROR!\nUnsupported symbol:",safety_check)
    continue
  window = [int(stoi[e]) for e in word_input]
  name = "".join([itos[e] for e in window])
  print(name)
  window = [0] * (ctx - len(window)) + window
  while True:
    probs = forward(embeddings[window].reshape(1,input_size))[-1][0]
    letter = rng.choice(vocab_size,p=probs)
    if letter == 0:
      break
    else:
      name += itos[letter]
      window = window[1:] + [letter]
    
  print(name, " " * (12 - len(name)),"•" if name in names else "=") #check if name got memorized


    
    







    
    





