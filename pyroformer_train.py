import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import matplotlib.pyplot as plt
import random

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 108 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

def load_protein_sequences_from_txt(filepath):
    """
    Lê o arquivo .txt, filtra APENAS pelas sequências 'Thermophilic'
    e retorna uma LISTA de sequências individuais.
    """
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) > 3 and parts[3] == 'Thermophilic':
                sequence = parts[1]
                sequences.append(sequence)
    return sequences

def encode(s, add_special_tokens=True):
    # Primeiro, adicionamos os espaços entre os aminoácidos, como no treinamento
    s_spaced = " ".join(list(s))
    
    if add_special_tokens:
        # Adiciona os tokens especiais, também com espaços, para consistência
        full_sequence_str = f"<sos> {s_spaced} <eos>"
    else:
        full_sequence_str = s_spaced
        
    return tokenizer.encode(full_sequence_str).ids

def decode(l, skip_special_tokens=True):
    decoded_text = tokenizer.decode(l, skip_special_tokens=skip_special_tokens)
    return decoded_text.replace(" ", "") # Remove os espaços adicionados

# Carregue suas sequências
sequences = load_protein_sequences_from_txt('Sequences_data/MeltomeTrain_cls.txt')
len(sequences)
# Adicionamos tokens para caracteres não-canônicos que podem estar no seu dataset
extra_chars = ['U', 'X', 'Z', 'B', 'O'] 
sequences = [s for s in sequences if not any(char in s for char in extra_chars)]
sequences = [s for s in sequences if len(s) <= 512 and len(s) >= 100]  # Filtra sequências muito longas


tokenizer = Tokenizer.from_file("protein_tokenizer.json")
tokenizer.get_vocab()
vocab = tokenizer.get_vocab()
pad_token_id = vocab["<pad>"]
vocab_size = len(vocab)


n = int(0.9 * len(sequences)) # 90% para treino, 10% para validação
train_data = sequences[:n]
val_data = sequences[n:]


def get_batch(split):

    # Seleciona o conjunto de dados (treino ou validação)
    data_list = train_data if split == 'train' else val_data
    
    # 1. Sorteia 'batch_size' sequências aleatórias da lista de strings
    # Certifique-se de que `data_list` tem pelo menos `batch_size` elementos.
    # Se não tiver, pode ocorrer um erro ou o `random.sample` pode retornar menos elementos.
    # Para datasets menores, considere `random.choices` ou lidar com isso de outra forma.
    batch_sequences_raw = random.sample(data_list, batch_size)
    
    # 2. Tokeniza e codifica cada sequência para uma lista de IDs
    # A função `encode` já adiciona <sos>, <eos> e cuida dos espaços
    batch_tokenized_ids = [tokenizer.encode(s).ids for s in batch_sequences_raw]
    
    # Encontra o comprimento da sequência mais longa neste lote
    # Este será o `T` (tempo/comprimento) para o lote atual
    max_len = max(len(ids) for ids in batch_tokenized_ids)
    
    # Garante que `max_len` não exceda `block_size` do seu modelo.
    # Sequências mais longas que `block_size` serão truncadas.
    # Isso é crucial para o Transformer.
    max_len = min(max_len, block_size) 
    
    # --- 3. Padding e Truncamento ---
    x_batch = []
    y_batch = []
    
    for token_ids in batch_tokenized_ids:
        # Aplica truncamento se a sequência for maior que max_len
        # (ou block_size se max_len > block_size)
        truncated_ids = token_ids[:max_len]
        
        # Sequência de input (x): todos os tokens exceto o último, com padding
        # Ex: <sos> M S K <pad> <pad>
        x = truncated_ids[:-1] # Remove o último token, pois ele será o target do penúltimo
        num_padding_x = max_len - len(x)
        x_padded = x + [pad_token_id] * num_padding_x
        
        # Sequência de target (y): todos os tokens exceto o primeiro, com padding
        # Ex: M S K <eos> <pad> <pad>
        y = truncated_ids[1:] # Remove o primeiro token (<sos>), pois ele foi o input do primeiro target
        num_padding_y = max_len - len(y)
        y_padded = y + [pad_token_id] * num_padding_y
        
        # Adiciona ao lote final, truncando novamente para `max_len` se necessário
        # (o que deve ser o caso se `max_len` foi definido pelo block_size)
        x_batch.append(x_padded[:max_len])
        y_batch.append(y_padded[:max_len])
        
    # Converte as listas para tensores PyTorch
    x = torch.tensor(x_batch, dtype=torch.long) # Shape (batch_size, max_len)
    y = torch.tensor(y_batch, dtype=torch.long) # Shape (batch_size, max_len)
    
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ One head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)     # (B,T,C)
        q = self.query(x)     # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim = -1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) * (B,T,C) -> (B,T,C)
        return out
          
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(([Head(head_size) for _ in range(num_heads)]))
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out

class FeedFoward(nn.Module):
    
    """" a simple linear layer follower by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
            )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """" Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_tablet = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head = 4),
        #     Block(n_embd, n_head = 4),
        #     Block(n_embd, n_head = 4),
        #     nn.LayerNorm(n_embd),
        # )
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention
        #self.ffwd = FeedFoward(n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_tablet(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        #x = self.sa_heads(x) # apply one head of self_attention. (B,T,C)
        #x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # CORREÇÃO: Corta o contexto para os últimos 'block_size' tokens
            # O modelo só consegue processar sequências de até 'block_size'
            idx_cond = idx[:, -block_size:]
            
            # Passa o contexto cortado para o modelo
            logits, loss = self(idx_cond)
            
            # O resto do código permanece igual
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
torch.save(model.state_dict(), "pyroformer_model.pt")
# Recommendations for improvement:

@torch.no_grad() # Desliga o cálculo de gradientes para inferência (economiza memória e é mais rápido)
def generate_sequences(model, tokenizer, num_sequences=5, max_new_tokens=512, temperature=1.0, top_k=None, device='cpu'):
    """
    Gera novas sequências de proteínas usando o modelo treinado.

    Args:
        model: O modelo Transformer treinado (GPT).
        tokenizer: O tokenizador BPE carregado.
        num_sequences (int): Quantas sequências gerar.
        max_new_tokens (int): Comprimento máximo da sequência a ser gerada (incluindo <sos>/<eos>).
        temperature (float): Controla a aleatoriedade da geração. >1.0 mais aleatório, <1.0 mais focado.
        top_k (int, opcional): Limita a amostragem aos top K tokens mais prováveis.
        device (str): O dispositivo para onde mover os tensores ('cpu' ou 'cuda').

    Returns:
        list: Uma lista de strings, onde cada string é uma sequência de proteína gerada.
    """
    model.eval() # Coloca o modelo em modo de avaliação (desliga dropout, batchnorm etc.)
    
    # Obtenha o ID do token de início de sequência
    sos_token_id = tokenizer.token_to_id("<sos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    pad_token_id = tokenizer.token_to_id("<pad>") # Usaremos para padding se o prompt precisar
    
    # O prompt inicial é apenas o token <sos>
    # Ele precisa ter a dimensão (batch_size, sequence_length), onde sequence_length é 1
    # Note que aqui estamos gerando uma sequência por vez para simplificar.
    # Se você quiser gerar N sequências em paralelo, o 'idx' inicial seria (num_sequences, 1)
    # e as próximas linhas teriam que ser adaptadas para lidar com um tensor 2D.
    
    generated_sequences = []

    print(f"\nIniciando a geração de {num_sequences} sequências de proteínas...")

    for _ in range(num_sequences):
        # Crie o tensor do prompt inicial para UMA sequência
        # O shape deve ser (1, 1) para [sos_token_id]
        idx = torch.tensor([[sos_token_id]], dtype=torch.long, device=device)
        
        # Gerar até max_new_tokens ou encontrar <eos>
        for i in range(max_new_tokens):
            # O modelo pode ter um `block_size` fixo para o contexto de entrada.
            # Certifique-se de que `idx` não excede `block_size` antes de passar para o modelo.
            idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
            
            # Obtenha as previsões do modelo
            logits, _ = model(idx_cond) # O modelo espera (B, T, C), então idx_cond deve ser (1, current_length)
            
            # Focamos apenas na última previsão (o próximo token na sequência)
            logits = logits[:, -1, :] # Torna-se (1, C)
            
            # Aplique temperature para suavizar ou aguçar as probabilidades
            if temperature == 0: # Para amostragem determinística (greedy)
                probs = F.softmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
            
            # Aplique top_k sampling se especificado
            if top_k is not None:
                v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                probs[probs < v[:, -1]] = 0 # Zera probabilidades abaixo do K-ésimo maior
                probs = probs / probs.sum(dim=-1, keepdim=True) # Renormaliza

            # Amostre o próximo token
            next_token_id = torch.multinomial(probs, num_samples=1) # next_token_id será (1, 1)
            
            # Concatene o token gerado com a sequência atual
            idx = torch.cat((idx, next_token_id), dim=1)
            
            # Se gerou o token <eos>, pare a geração para esta sequência
            if next_token_id.item() == eos_token_id:
                break
        
        # Decodifique os IDs gerados em uma string de aminoácidos
        # Remove os tokens especiais (<sos>, <eos>, <pad>) na decodificação final
        full_decoded_sequence = tokenizer.decode(idx.squeeze().tolist(), skip_special_tokens=True)
        generated_sequences.append(full_decoded_sequence)
        
        print(f"Sequência {len(generated_sequences)} gerada (comprimento {len(full_decoded_sequence)}).")

    model.train() # Retorna o modelo ao modo de treinamento
    return generated_sequences

NUM_GENERATIONS = 5
MAX_SEQ_LENGTH = 200 # Limita o comprimento das sequências geradas
TEMPERATURE = 0.8 # Um pouco de aleatoriedade, mas focado
TOP_K_SAMPLING = 50 # Considera apenas os 50 tokens mais prováveis

print("\n--- INICIANDO GERAÇÃO DE NOVAS PROTEÍNAS ---")
new_proteins = generate_sequences(
    model=m, # Seu modelo treinado
    tokenizer=tokenizer,
    num_sequences=NUM_GENERATIONS,
    max_new_tokens=MAX_SEQ_LENGTH,
    temperature=TEMPERATURE,
    top_k=TOP_K_SAMPLING,
    device=device
)

print("\n--- PROTEÍNAS GERADAS ---")
for i, protein in enumerate(new_proteins):
    print(f"Proteína {i+1}: {protein}")
    if len(protein) == 0:
        print("   (AVISO: Sequência vazia. O modelo pode ter gerado <eos> imediatamente.)")
