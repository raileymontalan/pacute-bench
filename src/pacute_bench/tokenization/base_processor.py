"""
Base class for tokenizer processors with common utilities.
"""

import os
import json
from tqdm import tqdm


class TokenizerProcessor:
    """
    Base class for tokenizer processors that provides common utilities.
    
    Attributes:
        tokenizer: The HuggingFace tokenizer instance
        tokenizer_name: Sanitized tokenizer name for use in filenames
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the base processor.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
        self.tokenizer_name = self._get_tokenizer_name()
    
    def _get_tokenizer_name(self):
        """
        Get a safe filename from tokenizer name or config.
        
        Returns:
            str: Sanitized tokenizer name with / and \ replaced by -
        """
        # Try to get name from tokenizer attributes
        if hasattr(self.tokenizer, 'name_or_path'):
            name = self.tokenizer.name_or_path
        elif hasattr(self.tokenizer, 'model_name'):
            name = self.tokenizer.model_name
        else:
            name = self.tokenizer.__class__.__name__
        
        # Sanitize for filename (replace / with -)
        return name.replace('/', '-').replace('\\', '-')
    
    def get_mergeable_ranks(self):
        """
        Get mergeable ranks (token bytes -> token ID mapping) for the tokenizer.
        
        This function provides compatibility across different tokenizer implementations:
        - For tiktoken-based tokenizers (e.g., GPT-4), uses the `_mergeable_ranks` attribute
        - For other tokenizers (e.g., GemmaTokenizerFast), constructs the mapping from vocab
        
        Returns:
            dict: Mapping from token bytes to token IDs {bytes: int}
            
        Note:
            Some tokenizers like Gemma use byte-fallback BPE where not all tokens
            can be cleanly represented as UTF-8 strings. This function handles such cases.
        """
        # Check if tokenizer has tiktoken's _mergeable_ranks attribute
        if hasattr(self.tokenizer, '_mergeable_ranks'):
            return self.tokenizer._mergeable_ranks
        
        # Fall back to building from vocab for other tokenizers
        print("Building mergeable_ranks from tokenizer vocab (tokenizer doesn't have _mergeable_ranks)...")
        vocab = self.tokenizer.get_vocab()
        mergeable_ranks = {}
        
        for token_str, token_id in vocab.items():
            try:
                # Try to encode the token string to bytes
                token_bytes = token_str.encode('utf-8')
                mergeable_ranks[token_bytes] = token_id
            except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
                # Skip tokens that can't be encoded (e.g., special tokens, byte-fallback tokens)
                continue
        
        print(f"Built mergeable_ranks with {len(mergeable_ranks)} entries from vocab of size {len(vocab)}")
        return mergeable_ranks
    
    def get_cache_path(self, subdirectory, filename):
        """
        Get the full path for a cache file in the project root data directory.
        
        Args:
            subdirectory: Subdirectory under data/ (e.g., "expansions")
            filename: Name of the cache file
            
        Returns:
            str: Absolute path to the cache file
        """
        # Get project root (two levels up from this file: src/tokenization -> src -> root)
        current_file_path = os.path.abspath(__file__)
        src_tokenization_dir = os.path.dirname(current_file_path)
        src_dir = os.path.dirname(src_tokenization_dir)
        project_root = os.path.dirname(src_dir)
        
        # Build path to data directory in project root
        cache_path = os.path.join(project_root, "data", subdirectory, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        return cache_path
    
    def build_expansions(self):
        """
        Build token expansion dictionary from tokenizer vocabulary.
        
        Process:
        1. Create merges dict: {(token_id1, token_id2): merged_token_id}
        2. Invert to expansions dict: {token_id: [(id1, id2), ...]}
        
        Returns:
            Dict mapping token IDs to lists of possible expansions (2-tuples of token IDs)
        """
        # Build merges dict from tokenizer vocabulary
        mergeable_ranks = self.get_mergeable_ranks()  # bytes -> token_id mapping
        tokens_as_tuples = [tuple(token) for token in mergeable_ranks.keys()]
        merges = {}
        
        for token_as_bytes, token_id in tqdm(
            mergeable_ranks.items(),
            desc="Building tokenizer's merges"
        ):
            # Skip tokens that are too short to split
            if len(token_as_bytes) <= 1:
                continue
            
            # Split token at each possible point and find valid merges
            for j in range(1, len(token_as_bytes)):
                first_part = token_as_bytes[:j]
                second_part = token_as_bytes[j:]
                
                # Check if both parts exist in vocabulary
                if (tuple(first_part) in tokens_as_tuples and 
                    tuple(second_part) in tokens_as_tuples):
                    first_part_id = mergeable_ranks[first_part]
                    second_part_id = mergeable_ranks[second_part]
                    merges[(first_part_id, second_part_id)] = token_id
        
        # Invert merges to create expansions dict
        expansions = {}
        for (id1, id2), merged_id in merges.items():
            if merged_id in expansions:
                expansions[merged_id].append((id1, id2))
            else:
                expansions[merged_id] = [(id1, id2)]
        
        return expansions
    
    def set_expansions(self):
        """
        Loads expansions dict from file if it exists, otherwise builds and saves it to file.
        Uses consolidated data/expansions/ directory for all tokenizers.
        """
        filename = f"expansions_{self.tokenizer_name}.json"
        json_file_path = self.get_cache_path("expansions", filename)

        # Check if the file exists
        if os.path.exists(json_file_path):
            print(f"Found '{filename}' at: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                expansions_raw = json.load(f)
                assert isinstance(expansions_raw, dict), f"{filename} must be a dictionary."
                print(f"Successfully loaded {filename}.")
                
                # Convert string keys to integers and lists to tuples
                expansions = {}
                for key, value in expansions_raw.items():
                    int_key = int(key)
                    # Convert lists back to tuples (JSON serialization converts tuples to lists)
                    expansions[int_key] = [tuple(exp) if isinstance(exp, list) else exp 
                                          for exp in value]
        else:
            print(f"Building expansions for {self.tokenizer_name}...")
            expansions = self.build_expansions()
            # save the expansions to the file
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(expansions, f)
            print(f"Successfully saved {filename} to: {json_file_path}")
        
        self.expansions = expansions
        print(f"Successfully set self.expansions with {len(expansions)} expandable tokens.")
