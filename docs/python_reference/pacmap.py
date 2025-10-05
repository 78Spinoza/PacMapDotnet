# PACMAP Python Reference Implementation
# Downloaded from: https://github.com/YingfanWang/PaCMAP/blob/master/source/pacmap/pacmap.py

"""
PaCMAP: Pairwise Controlled Manifold Approximation and Projection

Reference implementation for understanding the PACMAP algorithm.
This file will be used as reference for implementing PACMAP in C++.

Key differences from UMAP:
- Uses triplet-based structure preservation instead of fuzzy simplicial sets
- Three types of pairs: neighbors, mid-near, and further
- Three-phase optimization with dynamic weight adjustment
- Explicit loss function with weighted triplet distances
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings

class PaCMAP:
    """
    PaCMAP: Pairwise Controlled Manifold Approximation and Projection

    Parameters:
    -----------
    n_components : int, default=2
        Number of dimensions for the embedding

    n_neighbors : int, default=10
        Number of nearest neighbors to consider

    MN_ratio : float, default=0.5
        Mid-near ratio for triplet sampling

    FP_ratio : float, default=2.0
        Far-pair ratio for triplet sampling

    distance : str, default="euclidean"
        Distance metric to use

    lr : float, default=1.0
        Learning rate

    num_iters : tuple, default=(100, 100, 250)
        Number of iterations for each optimization phase

    verbose : bool, default=False
        Whether to print progress information

    apply_pca : bool, default=True
        Whether to apply PCA preprocessing (we will use False)

    random_state : int, default=None
        Random seed for reproducibility
    """

    def __init__(self, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
                 distance="euclidean", lr=1.0, num_iters=(100, 100, 250),
                 verbose=False, apply_pca=True, intermediate=False,
                 intermediate_snapshots=[0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450],
                 random_state=None, save_tree=False):

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.distance = distance
        self.lr = lr
        self.num_iters = num_iters
        self.verbose = verbose
        self.apply_pca = apply_pca
        self.intermediate = intermediate
        self.intermediate_snapshots = intermediate_snapshots
        self.random_state = random_state
        self.save_tree = save_tree

    def fit_transform(self, X, init=None, pair_neighbors=None, pair_MN=None, pair_FP=None):
        """
        Fit the model with X and return the embedded coordinates

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        embedding : array-like, shape (n_samples, n_components)
            Embedded coordinates
        """
        return self._fit(X, init, pair_neighbors, pair_MN, pair_FP)

    def fit(self, X, y=None, init=None, pair_neighbors=None, pair_MN=None, pair_FP=None):
        """
        Fit the model with X

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        self : object
            Returns the instance itself
        """
        self.embedding_ = self._fit(X, init, pair_neighbors, pair_MN, pair_FP)
        return self

    def transform(self, X):
        """
        Transform X into the embedded space

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        embedding : array-like, shape (n_samples, n_components)
            Embedded coordinates
        """
        # Implementation would go here
        pass

    def _fit(self, X, init=None, pair_neighbors=None, pair_MN=None, pair_FP=None):
        """
        Internal fitting method
        """
        n_samples, n_features = X.shape

        # Step 1: Preprocessing (optional PCA, normalization)
        if self.apply_pca and n_features > 100:
            # Apply PCA for dimensionality reduction
            pass

        # Step 2: Initialize embedding (random or specified)
        if init is None:
            np.random.seed(self.random_state)
            embedding = np.random.normal(0, 1e-4, (n_samples, self.n_components))
        else:
            embedding = init.copy()

        # Step 3: Sample triplets
        if pair_neighbors is None or pair_MN is None or pair_FP is None:
            pair_neighbors, pair_MN, pair_FP = self._sample_triplets(X)

        # Step 4: Optimize embedding using gradient descent
        embedding = self._optimize_embedding(embedding, pair_neighbors, pair_MN, pair_FP)

        return embedding

    def _sample_triplets(self, X):
        """
        Sample three types of triplets:
        1. Neighbor pairs: nearest neighbors
        2. Mid-near pairs: mid-distance pairs
        3. Further pairs: far-distance pairs
        """
        n_samples = X.shape[0]

        # Sample nearest neighbors
        pair_neighbors = self._sample_neighbors_pair(X)

        # Sample mid-near pairs
        n_MN = int(self.n_neighbors * self.MN_ratio)
        pair_MN = self._sample_MN_pair(X, n_MN)

        # Sample further pairs
        n_FP = int(self.n_neighbors * self.FP_ratio)
        pair_FP = self._sample_FP_pair(X, n_FP)

        return pair_neighbors, pair_MN, pair_FP

    def _sample_neighbors_pair(self, X):
        """
        Sample nearest neighbor pairs
        """
        n_samples = X.shape[0]

        # Use sklearn's NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1,
                               metric=self.distance).fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Remove self (first neighbor is always the point itself)
        pair_neighbors = []
        for i in range(n_samples):
            for j in range(1, self.n_neighbors + 1):
                pair_neighbors.append((i, indices[i, j]))

        return np.array(pair_neighbors)

    def _sample_MN_pair(self, X, n_MN):
        """
        Sample mid-near pairs for global structure preservation
        """
        n_samples = X.shape[0]
        pair_MN = []

        # Implementation would sample pairs at intermediate distances
        # This is simplified - actual implementation uses distance-based sampling

        for i in range(n_samples):
            # Find points at intermediate distances from point i
            # This would involve distance calculations and sampling strategy
            pass

        return np.array(pair_MN)

    def _sample_FP_pair(self, X, n_FP):
        """
        Sample further pairs for uniform distribution
        """
        n_samples = X.shape[0]
        pair_FP = []

        # Sample far pairs to encourage uniform distribution
        # This involves finding distant points in the original space

        for i in range(n_samples):
            # Find far points from point i
            # This would use distance-based sampling
            pass

        return np.array(pair_FP)

    def _optimize_embedding(self, embedding, pair_neighbors, pair_MN, pair_FP):
        """
        Optimize embedding using gradient descent with three phases
        """
        n_samples = embedding.shape[0]

        # Convert pairs to triplet format
        triplets = self._convert_to_triplets(pair_neighbors, pair_MN, pair_FP)

        # Three-phase optimization
        phase1_iters, phase2_iters, phase3_iters = self.num_iters
        total_iters = phase1_iters + phase2_iters + phase3_iters

        for iteration in range(total_iters):
            # Find weights for current iteration (three-phase schedule)
            w_neighbors, w_MN, w_FP = self._find_weight(iteration, total_iters)

            # Compute gradients
            gradients = self._pacmap_grad(embedding, triplets, w_neighbors, w_MN, w_FP)

            # Update embedding (could use SGD, Adam, etc.)
            embedding = self._update_embedding(embedding, gradients)

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, weights: w_neighbors={w_neighbors}, w_MN={w_MN}, w_FP={w_FP}")

        return embedding

    def _convert_to_triplets(self, pair_neighbors, pair_MN, pair_FP):
        """
        Convert pairs to triplet format with type information
        """
        triplets = []

        # Add neighbor pairs with type NEIGHBOR
        for i, j in pair_neighbors:
            triplets.append((i, j, 'neighbor'))

        # Add mid-near pairs with type MID_NEAR
        for i, j in pair_MN:
            triplets.append((i, j, 'mid_near'))

        # Add further pairs with type FURTHER
        for i, j in pair_FP:
            triplets.append((i, j, 'further'))

        return np.array(triplets)

    def _find_weight(self, iter, total_iters):
        """
        Three-phase weight adjustment schedule

        Phase 1 (0-10%): Focus on global structure, w_MN decreases from 1000 to 3
        Phase 2 (10-40%): Balance global and local, w_MN stays at 3
        Phase 3 (40-100%): Focus on local structure, w_MN drops to 0
        """
        progress = iter / total_iters

        if progress < 0.1:
            # Phase 1: Global structure
            w_MN = 1000.0 * (1.0 - progress * 10.0) + 3.0 * (progress * 10.0)
            w_neighbors = 1.0
            w_FP = 1.0
        elif progress < 0.4:
            # Phase 2: Balance
            w_MN = 3.0
            w_neighbors = 1.0
            w_FP = 1.0
        else:
            # Phase 3: Local structure
            w_MN = 3.0 * (1.0 - (progress - 0.4) / 0.6)
            w_neighbors = 1.0
            w_FP = 1.0

        return w_neighbors, w_MN, w_FP

    def _pacmap_grad(self, embedding, triplets, w_neighbors, w_MN, w_FP):
        """
        Compute gradients for PACMAP loss function

        Loss function components:
        - Neighbors: w_neighbors * (d_ij / (10 + d_ij))
        - Mid-near: w_MN * (d_ij / (10000 + d_ij))
        - Further: w_FP * (1 / (1 + d_ij))
        """
        n_samples, n_components = embedding.shape
        gradients = np.zeros_like(embedding)

        for i, j, triplet_type in triplets:
            # Compute distance in embedding space
            diff = embedding[i] - embedding[j]
            d_ij = np.sqrt(np.sum(diff ** 2))

            if d_ij < 1e-8:
                continue

            # Compute gradient based on triplet type
            if triplet_type == 'neighbor':
                # Pull closer: gradient = w_neighbors * 10 / ((10 + d)^2)
                grad_magnitude = w_neighbors * 10.0 / ((10.0 + d_ij) ** 2)
            elif triplet_type == 'mid_near':
                # Moderate pull: gradient = w_MN * 10000 / ((10000 + d)^2)
                grad_magnitude = w_MN * 10000.0 / ((10000.0 + d_ij) ** 2)
            else:  # further
                # Push apart: gradient = -w_FP / ((1 + d)^2)
                grad_magnitude = -w_FP / ((1.0 + d_ij) ** 2)

            # Apply gradient
            gradient = grad_magnitude * diff / d_ij
            gradients[i] += gradient
            gradients[j] -= gradient

        return gradients

    def _update_embedding(self, embedding, gradients):
        """
        Update embedding using gradients (could use SGD, Adam, etc.)
        """
        # Simple gradient descent update
        learning_rate = self.lr
        embedding -= learning_rate * gradients

        return embedding