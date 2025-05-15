import torch
import torch.nn.functional as F
import logging

# Setup logging
logger = logging.getLogger(__name__)


def normalize_features(features, dim=-1, eps=1e-8):
    """
    Normalize feature vectors along specified dimension with numerical stability.
    Args:
        features: Input tensor of any shape
        dim: Dimension along which to normalize
        eps: Small constant for numerical stability
    Returns:
        Normalized features with same shape as input
    """
    norm = features.norm(dim=dim, keepdim=True).clamp(min=eps)
    return features / norm


def validate_shape(tensor, expected_dims, name="tensor"):
    """
    Validate tensor shape has expected number of dimensions.
    Args:
        tensor: Input tensor to validate
        expected_dims: List of valid dimension counts
        name: Name of tensor for error messages
    """
    if tensor.dim() not in expected_dims:
        expected_str = ", ".join([str(d) for d in expected_dims])
        raise ValueError(f"Expected {name} to have {expected_str} dimensions, got {tensor.dim()}")


def refine_prompts(support_feats, text_feats, gamma=None):
    """
    TIMO's IGT module: Image‐Guided Text prompt refinement.
    Args:
      support_feats: Tensor[C, N, D] or Tensor[C, D] of few‐shot image features
      text_feats:    Tensor[C, P, D] of raw prompt embeddings
      gamma:         float temperature (if None, defaults per-dataset)
    Returns:
      Tensor[C, D] refined class text features
    """
    try:
        # Input validation
        validate_shape(text_feats, [3], "text_feats")
        validate_shape(support_feats, [2, 3], "support_feats")
        
        # Ensure consistent dtype
        dtype = text_feats.dtype
        device = text_feats.device
        support_feats = support_feats.to(dtype).to(device)
        
        # Normalize features
        text_feats = normalize_features(text_feats)
        
        # Collapse support if needed and normalize
        if support_feats.dim() == 3:
            # Average across shots dimension (N)
            support_feats = support_feats.mean(dim=1)
        support_feats = normalize_features(support_feats)
        
        C, P, D = text_feats.shape
        if support_feats.shape != (C, D):
            raise ValueError(f"Support features shape mismatch. Expected [C={C}, D={D}], got {support_feats.shape}")

        # Set gamma with reasonable bounds for numerical stability
        if gamma is None:
            gamma = 50.0
        # Ensure gamma is a scalar with reasonable bounds
        gamma = float(max(1.0, min(100.0, gamma)))

        # Compute per-prompt matching scores & weights with improved numerical stability
        # Use more explicit dimensions in einsum for better readability
        weights = torch.einsum('cd,cpd->cp', support_feats, text_feats)  # [C, P]
        
        # Improve numerical stability of softmax by scaling
        # Scaling before softmax helps prevent overflow/underflow
        # Correct temperature scaling - divide by gamma instead of multiply
        scaled_weights = weights / gamma
        
        # Further improve numerical stability by subtracting the max value
        scaled_weights = scaled_weights - scaled_weights.max(dim=1, keepdim=True)[0]
        
        # Apply softmax with increased numerical stability
        weights = F.softmax(scaled_weights, dim=1)  # [C, P]

        # Weighted sum of prompts → one refined vector per class
        refined = torch.einsum('cp,cpd->cd', weights, text_feats)  # [C, D]
        refined = normalize_features(refined)
        
        return refined
    except Exception as e:
        logger.error(f"Error in refine_prompts: {str(e)}")
        raise


def refine_images(image_feats, refined_text_feats, temperature=0.1):
    """
    TIMO's TGI module: Text‐Guided Image feature adaptation.
    Args:
      image_feats:        Tensor[N, D] raw support image features
      refined_text_feats: Tensor[D] corresponding refined text feature
      temperature:        Scaling factor for attention temperature
    Returns:
      Tensor[N, D] adapted image features
    """
    try:
        # Input validation
        validate_shape(image_feats, [2], "image_feats")
        validate_shape(refined_text_feats, [1], "refined_text_feats")
        
        N, D = image_feats.shape
        if refined_text_feats.shape != (D,):
            raise ValueError(f"Text features shape mismatch. Expected [D={D}], got {refined_text_feats.shape}")
        
        # Ensure consistent dtype and device
        dtype = image_feats.dtype
        device = image_feats.device
        refined_text_feats = refined_text_feats.to(dtype).to(device)
        
        # Normalize input features for consistent similarity computation
        image_feats = normalize_features(image_feats)
        refined_text_feats = normalize_features(refined_text_feats)

        # Compute attention scores with improved numerical stability
        # First reshape text features for matrix multiplication
        refined_text_feats_expanded = refined_text_feats.unsqueeze(-1)  # [D, 1]
        
        # Compute similarity scores with stable implementation
        sim = torch.matmul(image_feats, refined_text_feats_expanded).squeeze(-1)  # [N]
        
        # Scale similarity scores by temperature parameter for numerical stability
        # Lower temperature makes distribution more peaked, higher makes it more uniform
        # Use torch.clamp for better tensor type handling
        temperature = torch.clamp(torch.tensor(temperature, dtype=dtype, device=device), 
                                 min=0.01, max=100.0)
        sim_scaled = sim / temperature
        
        # Subtract max value for numerical stability before softmax
        sim_scaled = sim_scaled - sim_scaled.max()
        
        # Apply softmax for attention weights
        attn = F.softmax(sim_scaled, dim=0).unsqueeze(-1)  # [N, 1]
        
        # Weight image features by attention
        adapted = image_feats * attn  # [N, D]
        
        # Re-normalize the output features
        adapted = normalize_features(adapted)
        
        return adapted
    except Exception as e:
        logger.error(f"Error in refine_images: {str(e)}")
        raise





