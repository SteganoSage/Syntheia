"""
Day 2: Hugging Face Fundamentals & BERT Exploration
This script demonstrates:
1. Tokenization with BERT WordPiece tokenizer ([CLS], [SEP], token_ids, attention_mask)
2. Extracting contextual embeddings from bert-base-uncased
3. Running off-the-shelf Zero-Shot Classification on financial software requirements
"""

import torch
from transformers import AutoTokenizer, AutoModel, pipeline

def step_1_tokenization_demo():
    print("=" * 60)
    print("STEP 1: Understanding Tokenization with BERT")
    print("=" * 60)
    
    model_name = "bert-base-uncased"
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sample_text = "All customer Aadhaar and PAN data must be encrypted using AES-256 at rest."
    print(f"\nOriginal Input: \n\"{sample_text}\"")
    
    # Tokenize text
    encoded = tokenizer(
        sample_text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    
    print("\n--- Tokens Breakdown ---")
    print(f"Tokens: {tokens}")
    print(f"Input IDs: {encoded['input_ids'][0].tolist()}")
    print(f"Attention Mask: {encoded['attention_mask'][0].tolist()}")
    print(f"\n💡 Notice:")
    print(" • [CLS] is added at the start (ID: 101) - used as the sentence representation.")
    print(" • [SEP] is added at the end (ID: 102) - sentence separator.")
    print(" • Subwords like 'aadhaar' or 'pan' might be broken into sub-tokens if not in vocab.")


def step_2_embeddings_demo():
    print("\n" + "=" * 60)
    print("STEP 2: Extracting BERT Contextual Embeddings")
    print("=" * 60)
    
    model_name = "bert-base-uncased"
    print(f"Loading base model: {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    sample_text = "The system shall process online loan applications within 30 seconds."
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(sample_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # last_hidden_state shape: (batch_size, sequence_length, hidden_dim=768)
    last_hidden_state = outputs.last_hidden_state
    cls_embedding = last_hidden_state[:, 0, :] # Extract [CLS] vector
    
    print(f"Input text: \"{sample_text}\"")
    print(f"Sequence length (tokens): {last_hidden_state.shape[1]}")
    print(f"Hidden representation shape: {last_hidden_state.shape}")
    print(f"[CLS] sentence embedding vector dimension: {cls_embedding.shape[-1]} (768 dimensions)")
    print(f"[CLS] sample values (first 5 numbers): {cls_embedding[0, :5].tolist()}")
    print("\n💡 This 768-dim [CLS] vector is what we feed into our classification head!")


def step_3_zero_shot_baseline_demo():
    print("\n" + "=" * 60)
    print("STEP 3: Testing Off-the-Shelf Zero-Shot Classification Baseline")
    print("=" * 60)
    
    # Using a fast zero-shot model (BART-large-MNLI or DistilBERT based)
    print("Loading zero-shot classification pipeline (type: zero-shot-classification)...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    # Our 13 project categories
    candidate_labels = [
        "Business Requirements",
        "Stakeholder Requirements",
        "Functional Requirements",
        "Security Requirements",
        "Privacy Requirements",
        "Regulatory Requirements",
        "Performance Requirements",
        "Availability and Reliability",
        "Usability Requirements",
        "Data-management Requirements",
        "Integration Requirements",
        "Audit and Reporting",
        "Operational and Maintenance"
    ]
    
    test_requirements = [
        "All PII and cardholder data must be tokenized and encrypted according to PCI-DSS standards.",
        "The system must handle 5,000 loan applications per minute with a response time under 200ms.",
        "Monthly audit logs of all approved loans must be exported in PDF for RBI compliance checks."
    ]
    
    print("\nTesting 3 sample requirements against all 13 categories (Multi-label=True):\n")
    
    for req in test_requirements:
        result = classifier(req, candidate_labels, multi_label=True)
        print(f"Requirement: \"{req}\"")
        print("Top 3 Predicted Labels:")
        for label, score in zip(result["labels"][:3], result["scores"][:3]):
            print(f"  • {label}: {score:.2%}")
        print("-" * 50)
        
    print("\n💡 Key Observation:")
    print("While zero-shot works reasonably, it is slow and general.")
    print("Fine-tuning our own BERT will be 100x faster, lightweight, and tailored specifically to our 13 categories!")


if __name__ == "__main__":
    step_1_tokenization_demo()
    step_2_embeddings_demo()
    step_3_zero_shot_baseline_demo()
