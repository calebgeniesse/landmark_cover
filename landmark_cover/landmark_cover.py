"""
LandmarkCover for landmark-based binning with KeplerMapper.

Authors: Caleb Geniesse, geniesse@stanford.edu
         Samir Chowdhury, samirc@stanford.ed
"""
import numpy as np
from kmapper.cover import Cover
from sklearn.neighbors import NearestNeighbors
from torch_cluster import fps
import torch

__all__ = [
    'LandmarkCover'
]


class LandmarkCover(Cover):
    """
    Explicit definition of a landmark-based bin (or disk) cover.

    Parameters
    ============

    n_landmarks: int (default = None)
        Number landmarks to use (instead of individual data points)
        If None, all points will be used.

    
    Example
    ---------

    ::

        >>> import numpy as np
        >>> from kmapper.cover import LandmarkCover
        >>> data = np.random.random((100,2))
        >>> cov = LandmarkCover(radius=100)
        >>> centers = cov.fit(data)
        >>> cov.transform_single(data, centers[0])
        TODO: show output
        >>> hyper_cubes = cov.transform(data, centers)
    """
    def __init__(self, 
                 n_landmarks=None, 
                 perc_overlap=0.5, 
                 metric='minkowski',
                 p=1, 
                 metric_params=None,
                 n_jobs=None, 
                 verbose=0
        ):
        # self.centers_ = None
        
        self.n_landmarks = n_landmarks
        self.perc_overlap = perc_overlap
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose

        # deprecated
        self.n_cubes = self.n_landmarks


    def __repr__(self):
        return "LandmarkCover(n_landmarks=%s, perc_overlap=%s, metric=%s, p=%s, metric_params=%s, n_jobs=%s)" % (
            self.n_landmarks,
            self.perc_overlap,
            self.metric,
            self.p,
            self.metric_params,
            self.n_jobs,
        )

   
    def fit(self, data):
        """ Fit a cover on the data. This method constructs centers and radii in each dimension given the `perc_overlap` and `n_cube`.

        Parameters
        ============

        data: array-like
            Data to apply the cover to. Warning: First column must be an index column.

        Returns
        ========

        centers: list of arrays
            A list of centers for each bin

        """

        # TODO: support indexing into any columns
        di = np.array(range(1, data.shape[1]))
        indexless_data = data[:, di]
        n_dims = indexless_data.shape[1]
        n_samples = indexless_data.shape[0]

        # TODO: implement landmarking
        # TODO: should we run fps on geodesic distances?
        # TODO: does this work if data is a distance matrix?
        landmark_ratio = self.n_landmarks / n_samples
        landmarks = fps(
            torch.Tensor(indexless_data),
            batch=None,
            ratio=landmark_ratio,
            random_start=False # first node will be used as starting node
        )
        landmarks = landmarks.numpy()
        landmarks.sort()


        # extract centers (just set to input data points)
        centers = indexless_data[landmarks, :]

        
        # set radius
        landmark_nbrs = NearestNeighbors(
            metric=self.metric, p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
        landmark_nbrs.fit(centers)
        landmark_distances = landmark_nbrs.kneighbors_graph(
            X=data[:, di], n_neighbors=2, mode='distance',
        )
        
        # TODO: compute radius based on perc_overlap
        # TODO: accept fixed eps as a parameter?
        nonzeros = landmark_distances.A
        nonzeros[nonzeros < 1e-8] = np.nan
        epsilon = np.nanmax(np.nanmin(nonzeros, axis=1))
        radius = self.perc_overlap * (4. * epsilon)

        
        self.centers_ = centers
        self.radius_ = radius
        self.epsilon_ = epsilon

        self.landmark_ratio_ = landmark_ratio
        self.landmarks_ = landmarks
        self.landmark_nbrs_ = landmark_nbrs
        
        # TODO: kmapper expects n_cubes, perc_overlap, so set these for now
        # self.n_cubes = self.n_landmarks
        # self.perc_overlap = self.perc_overlap

        #self.nbrs_matrix_ = A
        self.di_ = di

        if self.verbose > 0:
            print(
                " - LandmarkCover - landmarks: %s\ncenters: %s"
                % (self.landmarks_, self.centers_)
            )

        return centers

    
    def transform_single(self, data, center, i=0):
        """ Compute entries of `data` in bin centered at `center`

        Parameters
        ===========

        data: array-like
            Data to find in entries in cube. Warning: first column must be index column.
        center: array-like
            Center points for the bin (i.e., corresponding to a landmark)
        i: int, default 0
            Optional counter to aid in verbose debugging.
        """
        
        self.landmark_nbrs_.fit(center.reshape(1,-1))
        hyperbin = self.landmark_nbrs_.radius_neighbors_graph(
            X=data[:,self.di_], radius=self.radius_,
            mode='connectivity',
        )
        hyperbin = data[hyperbin.nonzero()[0]]
        
        if self.verbose > 1:
            print(
                "There are %s points in landmark bin %s/%s"
                % (hyperbin.shape[0], i + 1, len(self.centers_))
            )

        return hyperbin

    def transform(self, data, centers=None):
        """ Find entries of all hyperbins. If `centers=None`, then use `self.centers_` as computed in `self.fit`.
            
            Empty hyperbins are removed from the result

        Parameters
        ===========

        data: array-like
            Data to find in entries in bin. Warning: first column must be index column.
        centers: list of array-like
            Center points for all bin centers as returned by `self.fit`. Default is to use `self.centers_`.

        Returns
        =========
        hyperbins: list of array-like
            list of entries in each hyperbin in `data`.

        """

        centers = centers or self.centers_
        hyperbins = [
            self.transform_single(data, center, i) for i, center in enumerate(centers)
        ]

        # Clean out any empty cubes (common in high dimensions)
        hyperbins = [bin for bin in hyperbins if len(bin)]
        return hyperbins

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


    
