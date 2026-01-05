import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
import torch


def create_gen_feature_extractor(model_name):
    """
    Creates a feature extractor pipeline for a given model.
    Compatible with: CL, Bert, Electra, SimSce, BGE, some GTE(thenlper), tbc
    """
    print(f"Creating feature extractor: {model_name}.")
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"Selected device: {device_name}")

    # If the model is compatible with SentenceTransformer (e.g., GTR models)
    if "gtr-t5-base" in model_name or "sentence-t5-base" in model_name.lower() or "modernbert-embed" in model_name.lower():
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model = model.to(f"cuda:{device}" if device == 0 else "cpu")

        def extractor(texts: list[str]):
            return model.encode(texts, convert_to_numpy=True)

        print("Loaded as SentenceTransformer model.")
        return extractor

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cuda:0" if device == 0 else "cpu")

    hf_pipeline = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer,
        device=device)
    print("Finished creating a feature extractor.")

    def extractor(texts: list[str]):
        outputs = hf_pipeline(texts)
        return [np.array(o) for o in outputs]

    # return extractor
    return hf_pipeline


def create_gte_feature_extractor(model_name):
    """
    Creates a feature extractor for a given model,
    Compatible with: some GTE (Alibaba), tbc.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def extract_features(texts):
        """
        Extracts helpers (embeddings) for a list of texts.

        Returns:
            A list of lists where each inner list is the token embeddings for a single input text.
            Each list has shape (seq_length, hidden_dim).
        """
        # Tokenize input texts
        batch_dict = tokenizer(
            texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        batch_dict = {key: val.to(device) for key, val in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)

        return outputs.last_hidden_state.cpu().numpy().tolist()

    return extract_features
