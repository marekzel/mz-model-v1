from transformers import AutoTokenizer
import numpy as np
import os

class MzTokenizerWrapper:
    def __init__(self, path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
    def encode(self, text: str) -> np.ndarray:
        """Converts text directly into a NumPy array of 16-bit integers."""
        list_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return np.array(list_ids, dtype=np.uint16)
        
    def decode(self, token_ids: np.ndarray) -> str:
        """Converts a NumPy array or list of token IDs back into text."""
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)

    def tokenize_file_to_npy(self, input_txt_path: str, separator: str = "===DOCUMENT_BOUND==="):
        """Reads a text file, tokenizes it using encode(), and saves as .npy."""
        with open(input_txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        documents = full_text.split(separator)
        eos_token_id = self.tokenizer.eos_token_id
        all_token_ids = []

        num_of_docs = 0
        for document in documents:
            clean_document = document.strip()
            if not clean_document:
                continue
                
            document_tokens = self.tokenizer.encode(clean_document, add_special_tokens=False)
            all_token_ids.extend(document_tokens)
            all_token_ids.append(eos_token_id)
            num_of_docs += 1

        token_array = np.array(all_token_ids, dtype=np.uint16)

        base_path, _ = os.path.splitext(input_txt_path)
        output_npy_path = f"{base_path}.npy"

        np.save(output_npy_path, token_array)
        print(f"Successfully saved {len(token_array)} tokens to {output_npy_path} (documents processed: {num_of_docs})")
        
        return len(token_array)

    def load_npy_to_tokens(self, input_npy_path: str) -> np.ndarray:
        """Loads a .npy file and returns it as a NumPy array."""
        return np.load(input_npy_path)