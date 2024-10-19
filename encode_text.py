import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Define batch size for reading and encoding text
BATCH_SIZE = 64  # Adjust batch size based on your system's memory capacity, e.g. 16 for slower speed

class TextEncoder:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _read_text_file_in_batches(self, file_path: str, chunk_size: int = 512):
        """
        Reads a large text file in chunks to process in batches.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Split text into chunks of size 'chunk_size'
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]

    def _encode_text_batch(self, text_batch: list):
        """
        Encode a batch of text using the transformer model.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            # Pass inputs through the model to get embeddings
            model_output = self.model(**inputs)
            # Pool the output to get sentence embeddings
            embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
        return embeddings.cpu().numpy()

    def _mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling to obtain sentence embeddings from token embeddings.
        """
        token_embeddings = model_output.last_hidden_state  # First element is the hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_and_store(self, file_path: str, output_file: str):
        """
        Encodes the text in batches and stores the embeddings into a file.
        """
        embeddings = []
        # Read and encode text in batches
        for batch, current_chunk, total_chunks in self._read_text_file_in_batches(file_path):
            batch_embeddings = self._encode_text_batch([batch])
            embeddings.append(batch_embeddings)
            chunk_count += 1
            
            # Print progress update
            if total_chunks > 0:
                print(f"Chunk {current_chunk}/{total_chunks} processed ({(current_chunk / total_chunks) * 100:.2f}% completed)")
            else:
                print(f"Chunk {current_chunk} processed")

        # Stack all the embeddings and save to a file
        embeddings = np.vstack(embeddings)
        np.save(output_file, embeddings)
        
        # Final message
        print(f"Document encoded successfully. Total chunks processed: {chunk_count}")

# Example usage:
if __name__ == "__main__":
    encoder = TextEncoder()
    input_file = "path\to\document.txt"  # Provide the path to your .txt file
    output_file = "path\to\store\encoded_embeddings.npy"  # Path to save the embeddings
    encoder.encode_and_store(input_file, output_file)

    print("Text encoded successfully.")
