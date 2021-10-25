"""
LandmarkCover for landmark-based binning with KeplerMapper.

Authors: Caleb Geniesse, geniesse@stanford.edu
         Samir Chowdhury, samirc@stanford.ed
"""
import numpy as np
from kmapper.cover import Cover
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse
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
                 radius=None,
                 metric='minkowski',
                 p=1, 
                 metric_params=None,
                 n_jobs=None, 
                 verbose=0
        ):        
        self.n_landmarks = n_landmarks
        self.perc_overlap = perc_overlap
        self.radius = radius 
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose

        # deprecated
        self.n_cubes = self.n_landmarks


    def __repr__(self):
        return "LandmarkCover(n_landmarks=%s, perc_overlap=%s, radius=%s, metric=%s, p=%s, metric_params=%s, n_jobs=%s)" % (
            self.n_landmarks,
            self.perc_overlap,
            self.radius,
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
        
        # create indexless copy of data
        di = np.arange(1, data.shape[1])
        indexless_data = data[:, di]
        n_samples, n_dims = indexless_data.shape

        # furthest point sampling of landmark points
        landmark_ratio = self.n_landmarks / n_samples
        landmarks = fps(
            torch.Tensor(indexless_data),
            batch=None,
            ratio=landmark_ratio,
            random_start=False # first node will be used as starting node
        )
        landmarks = landmarks.numpy() 

        # extract centers (just set to input data points)
        centers = indexless_data[sorted(landmarks)]

        # setup neighbors, compute landmark to data distances
        neighbors_estimator = NearestNeighbors(
            metric=self.metric, p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
        neighbors_estimator.fit(indexless_data)

        # compute radius (i.e., max min distance to a landmark)
        radius = self.radius
        if radius is None:
            landmark_distances = neighbors_estimator.kneighbors_graph(
                X=centers, n_neighbors=n_samples, mode='distance',
            )
            nonzeros = landmark_distances.A
            nonzeros[nonzeros < 1e-8] = np.nan
            eps = np.nanmax(np.nanmin(nonzeros, axis=0))
            radius = self.perc_overlap * (4. * eps)

        # store computed variables       
        self.di_ = di
        self.landmarks_ = landmarks
        self.centers_ = centers
        self.radius_ = radius
        self.neighbors_estimator_ = neighbors_estimator

        # display landmarks, centers
        if self.verbose > 0:
            print(
                " - LandmarkCover\nradius: %s\nlandmarks: %s\ncenters:%s"
                % (self.radius_, self.landmarks_, self.centers_)
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

        # create indexless copy of data
        indexless_data = data[:, self.di_]
        n_samples, n_dims = indexless_data.shape

        # make sure data used during fit matches data to be transformed
        if ((n_samples != self.neighbors_estimator_.n_samples_fit_) or 
               (n_dims != self.neighbors_estimator_.n_features_in_)):
            self.neighbors_estimator_.fit(indexless_data) 

        # compute landmark bin membership
        hyperbin = self.neighbors_estimator_.radius_neighbors_graph(
            X=center[np.newaxis, :], radius=self.radius_, mode='connectivity',
        )
        hyperbin = data[sorted(hyperbin.nonzero()[1])]
        
        # display counts?
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

        # check for new centers
        centers = centers or self.centers_
        
        # transform centers
        hyperbins = [
             self.transform_single(data, center, i) for i, center in enumerate(centers)
        ]
        
        # clean out any empty cubes (common in high dimensions)
        hyperbins = [hyperbin for hyperbin in hyperbins if len(hyperbin)]

        return hyperbins


    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


    
