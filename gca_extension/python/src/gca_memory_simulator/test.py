"""Test script for computing cosine similarity between contradictory phrases using Gemini embeddings.

This module demonstrates how to use the Gemini embedding model to compute
cosine similarity between two phrases that contradict each other.
"""

import numpy as np
from google import genai
import bisect
import os


# API key for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.
    
    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.
        
    Returns:
        Cosine similarity score between -1 and 1, where 1 indicates identical
        vectors, 0 indicates orthogonal vectors, and -1 indicates opposite vectors.
    """
    # Normalize the vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    cosine_sim = dot_product / (norm1 * norm2)
    
    return float(cosine_sim)


def get_embedding(client: genai.Client, text: str) -> np.ndarray:
    """Get embedding for a given text using Gemini embedding model.
    
    Args:
        client: Initialized Gemini client.
        text: Text to get embedding for.
        
    Returns:
        Numpy array containing the embedding vector.
    """
    result = client.models.embed_content(
        model="gemini-embedding-001",  # Current stable Gemini embedding model
        contents=text
    )
    
    # Extract the embedding values and convert to numpy array
    embedding_values = result.embeddings[0].values
    return np.array(embedding_values, dtype=np.float32)


def quantile(vs, alpha):
    """Return the α-quantile of nonempty list vs."""
    xs = sorted(vs)
    idx = int(alpha * (len(xs)-1))
    return xs[idx]


def strength_pure(sims, t_neg=-0.4, t_pos=0.4,
                  alpha_lo=0.25, alpha_hi=0.75):
    # 1. Build P1 and P2
    P1 = [s for s in sims if s >= t_pos]
    if    P1:
        P = P1
    else:
        P2 = [s for s in sims if s > 0]
        P  = P2

    # 2. Anchors
    if P:
        Ql = quantile(P, alpha_lo)
        Qu = quantile(P, alpha_hi)
    else:
        Ql, Qu = 0.0, -t_neg  # = 0.4

    # 3. Map
    out = []
    for s in sims:
        if s > 0:
            out.append(s)
        elif s > t_neg:
            out.append(0.0)
        else:
            r = (abs(s) - abs(t_neg)) / (1 - abs(t_neg))
            out.append(Ql + r * (Qu - Ql))
    return out


def transform(s, L=0.4, U=0.6):
    if s <= L:
        return 0.0
    if s >= U:
        return 1.0
    return (s - L) / (U - L)

def combine_threshold_fusion(sims_matrix, L=0.4, U=0.6, alpha=0.5):
    """
    sims_matrix: list of m lists of length N,
    sims_matrix[j][i] = s_j(d_i).
    Returns list of length N with fused scores C(d_i).
    """
    m = len(sims_matrix)
    if m == 0:
        return []
    N = len(sims_matrix[0])
    out = []
    for i in range(N):
        tvals = [transform(sims_matrix[j][i], L, U) for j in range(m)]
        mean_t = sum(tvals) / m
        max_t  = max(tvals)
        out.append(alpha * mean_t + (1 - alpha) * max_t)
    return out



def main():
    """Main function to demonstrate cosine similarity between contradictory phrases."""
    # Initialize the Gemini client
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Define two contradictory phrases
    phrase1 = "The weather is absolutely beautiful and sunny today."
    phrase2 = "The weather is terrible and stormy today."
    
    print("Computing cosine similarity between contradictory phrases:")
    print(f"Phrase 1: '{phrase1}'")
    print(f"Phrase 2: '{phrase2}'")
    print()
    
    # Get embeddings for both phrases
    print("Generating embeddings...")
    embedding1 = get_embedding(genai_client, phrase1)
    embedding2 = get_embedding(genai_client, phrase2)
    
    # Compute cosine similarity
    similarity = compute_cosine_similarity(embedding1, embedding2)
    
    print(f"Embedding 1 shape: {embedding1.shape}")
    print(f"Embedding 2 shape: {embedding2.shape}")
    print(f"Cosine similarity: {similarity:.4f}")
    print()
    
    # Interpret the result
    if similarity > 0.8:
        interpretation = "Very similar"
    elif similarity > 0.5:
        interpretation = "Somewhat similar"
    elif similarity > 0.2:
        interpretation = "Slightly similar"
    elif similarity > -0.2:
        interpretation = "Neutral/unrelated"
    elif similarity > -0.5:
        interpretation = "Somewhat dissimilar"
    else:
        interpretation = "Very dissimilar"
    
    print(f"Interpretation: {interpretation}")
    
    # Test with more similar phrases for comparison
    print("\n" + "="*60)
    print("For comparison, testing with similar phrases:")
    
    similar_phrase1 = "The weather is absolutely beautiful and sunny today."
    similar_phrase2 = "Today has gorgeous weather with bright sunshine."
    
    print(f"Phrase 1: '{similar_phrase1}'")
    print(f"Phrase 2: '{similar_phrase2}'")
    
    embedding3 = get_embedding(genai_client, similar_phrase1)
    embedding4 = get_embedding(genai_client, similar_phrase2)
    
    similar_similarity = compute_cosine_similarity(embedding3, embedding4)
    print(f"Cosine similarity: {similar_similarity:.4f}")
    
    test_negatives = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4]
    print(f"Contradiction strength: {strength_pure(test_negatives)}")
    print(f"Contradiction strength: {strength_pure([-0.9, -0.8])}")
    print(f"Contradiction strength: {strength_pure([-0.6, -0.5])}")
    test_negatives_positives = test_negatives + [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"Contradiction strength: {strength_pure(test_negatives_positives)}")
    test_negatives_small_positives = test_negatives + [0.0, 0.1, 0.2, 0.3, 0.4]
    print(f"Contradiction strength: {strength_pure(test_negatives_small_positives)}")
    test_positives = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"Contradiction strength: {strength_pure(test_positives)}")
    test_combine_scores = combine_threshold_fusion([[0.8, 0.77], [0.7, 0.77], [0.6, 0.77]])
    print(f"Combined scores: {test_combine_scores}")



if __name__ == "__main__":
    main()

