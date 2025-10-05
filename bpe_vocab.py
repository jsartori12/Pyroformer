# train_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import matplotlib.pyplot as plt
import random
import torch

# 1. Reutilize a função que você já criou para carregar e filtrar os dados
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

# Carregue suas sequências
sequences = load_protein_sequences_from_txt('Sequences_data/MeltomeTrain_cls.txt')
len(sequences)
# Adicionamos tokens para caracteres não-canônicos que podem estar no seu dataset
extra_chars = ['U', 'X', 'Z', 'B', 'O'] 
sequences = [s for s in sequences if not any(char in s for char in extra_chars)]
sequences = [s for s in sequences if len(s) <= 512 and len(s) >= 100]  # Filtra sequências muito longas
sequences[0]
longest_element = max(sequences, key=len)
len(sequences)
# 2. Inicialize um tokenizador BPE vazio
# O BPE começa com um alfabeto de caracteres e aprende a juntar os pares mais frequentes
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

initial_alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'] 
len(initial_alphabet)
special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

# 3. Defina um trainer para o tokenizador
# vocab_size é um hiperparâmetro! Ele define o tamanho final do seu vocabulário.
# Inclui os caracteres básicos, os tokens especiais e todos os novos "motivos" aprendidos.
# Um valor entre 1000 e 5000 é um bom começo.
trainer = BpeTrainer(vocab_size=50256 , special_tokens=special_tokens, initial_alphabet=initial_alphabet)


# 5. Treine o tokenizador
# O pre_tokenizer Whitespace() vai dividir a sequência nos espaços que adicionamos
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(sequences, trainer=trainer)

# 6. Salve o tokenizador treinado em um arquivo
tokenizer.save("protein_tokenizer.json")
print("Tokenizador treinado e salvo em protein_tokenizer.json")
tokenizer = Tokenizer.from_file("protein_tokenizer.json")

test_sequence_spaced = " ".join(list("MFSGFNACDDFPAGVDPALGLVPPASSRD"))
# Opcional: Teste o tokenizador
encoded = tokenizer.encode("MFSGFNACDDFPAGVDPALGLVPPASSRDFSGFNACDDFPAGVDPALGLVPPASSRD")


print("Sequência de teste tokenizada:")
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)

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

# --- TESTE CORRETO DA FUNÇÃO ENCODE ---
print("\n--- Testando a função encode corrigida ---")
test_seq = "MFSGFNACDDFPAGVDPALGLVPPASSRD"
encoded_ids = encode(test_seq)
decoded_text = decode(encoded_ids)
