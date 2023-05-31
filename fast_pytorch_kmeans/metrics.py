from torch.nn.functional import normalize

def cos_sim(a, b):
    """
      Compute cosine similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return normalize(a, dim=-1) @ normalize(b, dim=-1).transpose(-2, -1)


def euc_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return 2 * a @ b.transpose(-2, -1) - (a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]



def dot_sim(a, b):
    """
      Compute "dot product" similarity of 2 sets of vectors. 
      With previously normalized vectors, it speedup the computation (normalized vectors dot product is consine similarity)
      
      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return a @ b.transpose(-2, -1)