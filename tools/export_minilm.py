import torch
from transformers import AutoModel, AutoTokenizer
import struct
import os

# Format: HVL_MODEL (Binary)
# [MAGIC: 8 bytes "HVLMODEL"]
# [HEADER: 4x int32] (num_layers, hidden_dim, vocab_size, max_seq_len)
# [EMBEDDINGS: word_embeddings, pos_embeddings, type_embeddings]
# [LAYERS: 6x Transformer Blocks]

def align_to_byte(f, alignment=64):
    current = f.tell()
    pad = (alignment - (current % alignment)) % alignment
    if pad > 0:
        f.write(b'\0' * pad)

def export_model(model_name, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    config = model.config
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    max_seq_len = config.max_position_embeddings
    
    print(f"Config: L={num_layers}, H={hidden_dim}, V={vocab_size}, S={max_seq_len}")

    with open(output_path, "wb") as f:
        # 1. Header
        f.write(b"HVLMODEL")
        f.write(struct.pack("iiii", num_layers, hidden_dim, vocab_size, max_seq_len))
        align_to_byte(f)

        def write_tensor(t):
            data = t.detach().numpy().astype('float32').tobytes()
            f.write(data)
            align_to_byte(f)

        # 2. Embeddings
        print("Exporting Embeddings...")
        write_tensor(model.embeddings.word_embeddings.weight)
        write_tensor(model.embeddings.position_embeddings.weight)
        write_tensor(model.embeddings.token_type_embeddings.weight)
        write_tensor(model.embeddings.LayerNorm.weight)
        write_tensor(model.embeddings.LayerNorm.bias)

        # 3. Transformer Layers
        for i in range(num_layers):
            print(f"Exporting Layer {i}...")
            layer = model.encoder.layer[i]
            
            # Attention
            write_tensor(layer.attention.self.query.weight)
            write_tensor(layer.attention.self.query.bias)
            write_tensor(layer.attention.self.key.weight)
            write_tensor(layer.attention.self.key.bias)
            write_tensor(layer.attention.self.value.weight)
            write_tensor(layer.attention.self.value.bias)
            
            # Output
            write_tensor(layer.attention.output.dense.weight)
            write_tensor(layer.attention.output.dense.bias)
            write_tensor(layer.attention.output.LayerNorm.weight)
            write_tensor(layer.attention.output.LayerNorm.bias)

            # FFN
            write_tensor(layer.intermediate.dense.weight)
            write_tensor(layer.intermediate.dense.bias)
            write_tensor(layer.output.dense.weight)
            write_tensor(layer.output.dense.bias)
            write_tensor(layer.output.LayerNorm.weight)
            write_tensor(layer.output.LayerNorm.bias)

    vocab_path = os.path.join(os.path.dirname(output_path), "vocab.txt")
    print(f"Exporting vocab to {vocab_path}...")
    with open(vocab_path, "w", encoding="utf-8") as fv:
        sorted_vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
        for token, _ in sorted_vocab:
            fv.write(f"{token}\n")

    print("Success! Model exported with 64-byte alignment.")

if __name__ == "__main__":
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    output_path="models/paraphrase-multilingual-MiniLM-L12-v2.hvl_model"
    export_model(model_name, output_path)
