import torch
import torch.nn as nn
import torch.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttentionHead(nn.Module):
    """Attention head of the transformer. Used to allow communication between times. Each character emits 
    a key showing what they contain, a query showing what information they find valuable for their prediction,
    and a value showing what they have to actually say. 

    Attributes:
        head_size (int): Dimensionality of the head elements
        emb_size (int): Dimensionality of the embedding space for each characer
        dropout (float): Percentage of neurons to execute dropout on
    """

    def __init__(self, head_size: int, emb_size: int, dropout: float):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass through the attention head. We project the queries onto the keys defining the affinity,
        and this shows how significant the value will be.

        Args:
            x (torch.tensor): The input to be forwarded through the attention head
        
        Returns:
            torch.tensor: Output layer of the attention head
        """
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        aff = q @ k.transpose(1, 2)  # Projection of the queries onto the
        aff = aff * self.head_size ** -.5  # Ensure unit variance for a gaussian input
        # Turn the aff logits into a prob distribution
        aff = F.softmax(aff, dim=2)
        aff = self.dropout(aff)  # Apply the dropout to the affinities

        att_out = aff @ v  # Provide weights to the values corresponding to their affinities
        return att_out


class Block(nn.Module):
    """One complete transformer Block. Combines a feedforward net with the multihead-attention layers. 
    First, we allow the times to communicate with times from the past, then we let a feedforward
    neural network process this information.
    Also uses a layer norm over a batch norm to avoid many of the annoyances associated with batch norms

    Attributes:
        emb_size (int): Embedding dimension for each character/position
        num_heads (int): Number of attention heads in the multihead layer
        ff_size (int): Size of the feedforward network
    """

    def __init__(self, emb_size: int, num_heads: int, ff_size: int, dropout: float):
        super().__init__()

        # Communication (Chain together attention heads in parallel)
        self.num_heads = num_heads
        self.head_size = emb_size // num_heads  # head_size * num_heads = emb_size
        self.head_list = nn.ModuleList(
            [AttentionHead(self.head_size,emb_size,dropout) for i in range(num_heads)])
        self.att_linear = nn.Linear(self.head_size * self.num_heads, emb_size)
        self.dropout = nn.Dropout(dropout)

        # Standard feedforward net
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, emb_size),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.attention_ln = nn.LayerNorm(emb_size)
        self.ff_ln = nn.LayerNorm(emb_size)

    def forward(self, x):
        """Forward pass through the self attention block

        Args:
            x (torch.tensor): Data to be forwarded through the block
        
        Returns:
            torch.tensor: Data outputted by a forward pass through the block
        """
        # Perform a self-attention
        attention_ln_out = self.attention_ln(x)
        # Forward pass through each head individually
        heads = [head(attention_ln_out) for head in self.head_list]
        # Concatinate all the head outputs into one layer
        self_attention_out = torch.cat(heads)
        # Project heads back into emb_size
        self_attention_out = self.att_linear(self_attention_out)
        # Apply the dropout to the output
        self_attention_out = self.dropout(self_attention_out)

        # Layernorm and forwad through the feedforward network
        ff_ln_out = self.ff_ln(x)
        ff_out = self.ff(ff_ln_out)

        # Sum the ff, self_attention, and x. Uses recurrence connections, which function thanks to
        # the linearity of the derivative in backpropagation
        out = x + ff_out + self_attention_out

        return out


class Transformer(nn.Module):

    def __init__(self, vocab_size: int, emb_size: int, block_size: int, num_blocks: int, num_heads: int,dropout:float):

        self.token_emb_transform = nn.Embedding(vocab_size, emb_size)
        self.pos_emb_transform = nn.Embedding(vocab_size, block_size)

        # Generate Blocks
        self.blocks_list = [Block(emb_size, num_heads) for i in range(num_blocks)]
        self.blocks_transform = nn.Sequential(*self.blocks_list) # Unpack blocks_list to be funcitonal arg
        self.final_layer_norm = nn.LayerNorm(emb_size) # Apply the layer norm
        self.final_transform = nn.Linear(emb_size, vocab_size) # Create logits

        self.apply(self._init_weights)

    # Weight initializations taken from Andrej Karpathy's ChatGPT tutorial
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ex, targets):
        token_embs = self.token_emb_transform(ex)
        pos_embs = self.pos_emb_transform(ex)
        x = token_embs + pos_embs
        x = self.blocks_transform(x)
        x = self.final_layer_norm(x)
        logits = self.final_transform(x)

        if targets:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss