import torch
import torch.nn as nn
import torch.functional as F
import random
from LM import Transformer

# Parameters
VOCAB_SIZE = 12 # 0-9,+,=
DROPOUT = .2
EMB_SIZE = 384
BLOCK_SIZE = len('abc+def=ghij')
NUM_BLOCKS = 6
NUM_HEADS = 6
NUM_LEN = 3
BATCH_SIZE = 64
TR_SIZE = 100_000
LEARNING_RATE = 3e-4
NUM_ITERS = 5000
INTERVAL = 200
device = device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gen_data(n: int) -> list:
    """
    Generates Addition problems where we are assuming the numbers are reverse. 
    Ex: "324 + 534 = 858" goes 1's, 10's then 100's instead of 100's 10's 1's. This improves model
    performance, as with addition, we want to calculate the 1's place first

    Args:
        n (int): Number of examples you want to generate
    
    Returns:
        list: Python list of encoded data
    """

    nums1 = [random.randint(0,999) for _ in range(n)]
    nums2 = [random.randint(0,999) for _ in range(n)]
    X = [f"{i}+{j}=" for i,j in zip(nums1,nums2)] 
    Y = [{str(i+j)[::-1]} for i,j in zip(nums1,nums2)] # Reverse the answer to improve performance
    return X,Y

def encode_data(s: str) -> list:
    if len(s) < len('abc+def=ghij'):
        nums1, remainder = s.split('+')
        nums2, nums3 = remainder.split('=')
        print(nums1)
        while len(nums1) < NUM_LEN:
            nums1 = '0' + nums1
        while len(nums2) < NUM_LEN:
            nums2 = '0' + nums2
        while len(nums3) < NUM_LEN:
            nums3 = '0' + nums3

    out = []
    for c in nums1 + '+' + nums2 + '=' + nums3:
        if c == '+':
            out.append(-1)  # Map + to -1
        elif c == '=':
            out.append(-2)  # Map = to -2
        else:
            out.append(int(c))  # Else map to val
    return out

def decode_data(l: list) -> str:
    return ''.join(l)

def create_batch(bs):
    x,y = gen_data(bs)
    x = torch.tensor(x,dtype=torch.long)
    y = torch.tensor(y,dtype=torch.long)
    return x,y

@torch.no_grad()
def estimate_loss(model,interval,bs):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros()
        for k in range(interval):
            X, Y = create_batch(bs)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


X = create_batch(BATCH_SIZE)

model = Transformer(VOCAB_SIZE,EMB_SIZE,BLOCK_SIZE,NUM_BLOCKS,NUM_HEADS,DROPOUT)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr = )

for iter in range(NUM_ITERS):

    # every once in a while evaluate the loss on train and val sets
    if iter % INTERVAL == 0 or iter == NUM_ITERS - 1:
        losses = estimate_loss(model,INTERVAL,BATCH_SIZE)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = create_batch(BATCH_SIZE)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()