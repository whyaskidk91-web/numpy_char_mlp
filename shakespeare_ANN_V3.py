from shakespeare_datapipe import train,test,stoi,itos,ctx,vocab_size
#import data,vocabs and context size
import numpy as np

train_steps = 30000 #iterations
emb_size = 10 #embedding size
batch_size = 32
hidden_layer_size = 128
lr_base,lr = 0.01,0.01
input_size = ctx * emb_size
#flattened embedding

rng = np.random.default_rng(5)
#random seed

layers = [input_size,hidden_layer_size,hidden_layer_size,vocab_size]
weights = [rng.normal(0,np.sqrt(2 / a),size=(a,c)) for a,c in zip(layers,layers[1:])] #includes He init sqrt(2 / a)
biases = [np.zeros(c) for c in layers[1:]]
embeddings = rng.normal(0,1,size=(vocab_size,emb_size))
mW = [np.zeros_like(W) for W in weights] #momentum for weights
vW = [np.zeros_like(W) for W in weights] #average of squared gradient for weights
mb = [np.zeros_like(bb) for bb in biases]
vb = [np.zeros_like(bb) for bb in biases]
mC = np.zeros_like(embeddings)
vC = np.zeros_like(embeddings)
beta1,beta2,eps = 0.9,0.999,1e-8
train_loss,print_gap = 0,1000

def batch(bD,batch_size):
  idx = rng.integers(0,len(bD) - ctx,size=batch_size)
  X = bD[idx[:,None] + np.arange(ctx)]
  Y = bD[idx + ctx]
  return X,Y

def relu(z):
  return np.maximum(0,z)

def softmax(z):
  z = z - z.max(axis=1,keepdims=True)
  e = np.exp(z)
  return e / e.sum(axis=1,keepdims=True)

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
  gradient = target - layer_output[-1] #!negative gradient softmax and crossentropy
  for i in reversed(range(len(weights))):
    gW = layer_output[i].T @ gradient / batch_size
    gb = gradient.mean(axis=0)
    if i > 0:
      gradient = (gradient @ weights[i].T) * (layer_output[i] > 0) #ReLU derivative: only activated neurons pass gradient back
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
  win,y = batch(train,batch_size)
  xin = embeddings[win].reshape(batch_size,input_size)
  target = np.eye(vocab_size)[y]
  layer_output = forward(xin)
  prob_choice = layer_output[-1][np.arange(batch_size),y]
  train_loss += -np.log(prob_choice + 1e-9).mean()
  if (i + 1) % print_gap == 0:
    print(train_loss / print_gap)
    train_loss = 0
  lr = lr_base * 0.5 * (1 + np.cos(np.pi * i / train_steps))
  #cosine learn rate
  backward(layer_output,target,i + 1,win)

def accuracy(aD):
  correct = 0
  for _ in range(100):
    w,y = batch(aD,batch_size)
    xin = embeddings[w].reshape(batch_size,input_size)
    value = forward(xin)[-1]
    correct += (y == np.argmax(value,axis=1)).sum()
  return correct / batch_size
  #prints percentage

def loss(lD):
  total_loss = 0
  for _ in range(100):
    w,y = batch(lD,batch_size)
    xin = embeddings[w].reshape(batch_size,input_size)
    value = forward(xin)[-1]
    likely = value[np.arange(batch_size),y]
    total_loss += -np.log(likely + 1e-9).mean()
  return total_loss / 100

#Print accuracy and loss:
print("\n\nAC train: \n", accuracy(train))
print("AC test: \n", 
accuracy(test))
print("LOSS train: \n", loss(train))
print("LOSS test: \n", loss(test))
print("\n#######TEXT#######\n")

#generate random text
window = [stoi['\n']] * ctx #empty window as start
out = ""
for _ in range(300):
  probs = forward(embeddings[window].reshape(1,input_size))[-1][0]
  letter = rng.choice(vocab_size,p=probs)   #random choice of letters with p probabilities
  out += itos[letter]
  window = window[1:] + [letter]
print(out)