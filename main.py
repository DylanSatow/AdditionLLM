import torch
import torch.nn as nn
import torch.functional as F
import random

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
    X = [f"{i}+{j}={str(i+j)[::-1]}" for i,j in zip(nums1,nums2)] # Reverse the answer to improve performance

    return X

def encode_data(s: str) -> list:
    out = []
    for c in s:
        if c == '+':
            out.append(-1) # Map + to -1
        elif c=='=':
            out.append(-2) # Map = to -2
        else:
            out.append(c) # Else map to val
    return out

X = gen_data(1000)
print(X[:5])
X = [encode_data(s) for s in X]


