# Landmark Cover

[Landmark Cover](https://calebgeniesse.github.io/landmark_cover) is a Python implementation of [NeuMapper](https://braindynamicslab.github.io/neumapper/)'s landmark-based cover.

The `LandmarkCover` transformer was designed for use with [KeplerMapper](https://kepler-mapper.scikit-tda.org/en/latest/), but rather than dividing an *extrinsic* space (e.g., low-dimensional projection) into overlapping hypercubes, the landmark-based approach directly partitions data points into overlapping subsets based on their *intrinsic* distances from pre-selected landmark points.

Unlike [KeplerMapper](https://kepler-mapper.scikit-tda.org/en/latest/)'s `CubicalCover`, which scales exponentially with the dimensionality of the lens (i.e., `O(n_cubes ** lens.shape[1])`), `LandmarkCover` can be used to efficiently partition higher dimensional embeddings, or even the input data itself. Below we show examples using an entire pairwise geodesic distance matrix as a high dimensional lens. 



## Related Projects

- [NeuMapper](https://braindynamicslab.github.io/neumapper/) is a scalable Mapper algorithm for neuroimaging data analysis. The Matlab implementation was designed specifically for working with complex, high-dimensional neuroimaging data and produces a shape graph representation that can be annotated with meta-information and further examined using network science tools.

- [Reciprocal Isomap](https://calebgeniesse.github.io/reciprocal_isomap) is a reciprocal variant of Isomap for robust non-linear dimensionality reduction in Python. `ReciprocalIsomap` was inspired by scikit-learn's implementation of Isomap, but the reciprocal variant enforces shared connectivity in the underlying *k*-nearest neighbors graph (i.e., two points are only considered neighbors if each is a neighbor of the other).



## Setup

### Dependencies

#### [Python 3.6+](https://www.python.org/)

#### Required Python Packages
* [numpy](https://www.numpy.org)
* [scipy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org)
* [kmapper](https://scikit-tda.org/kepler-mapper)
* [torch](https://pytorch.org)
* [torch-cluster](https://github.com/rusty1s/pytorch_cluster)


### Install using pip

Assuming you have the required dependencies, you should be able to install using pip.
```bash
pip install git+https://github.com/calebgeniesse/landmark_cover.git
```

Alternatively, you can also clone the repository and build from source. 
```bash
git clone git@github.com:calebgeniesse/landmark_cover.git
cd landmark_cover

pip install -r requirements.txt
pip install -e .
```





## Usage


### `LandmarkCover` with a 2D lens

In the examples below, we look at a set of data points sampled from a trefoil knot. To generate this data, we use the `make_trefoil` tool provided by [dyneusr](https://braindynamicslab.github.io/dyneusr). Note, `dyneusr` is otherwise not required to use the `LandmarkCover`.

```python
from dyneusr.datasets import make_trefoil, draw_trefoil3d

# sample 100 points from a trefoil knot
trefoil = make_trefoil(100, noise=0.0)
X = trefoil.data

# visualize the data
draw_trefoil3d(X[:,0], X[:,1], X[:,2])

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/trefoil_knot.png?raw=true" width="80%">
</a>
</p>



Here, we show a simple example using `LandmarkCover` with `KeplerMapper`. 

```python
from kmapper import KeplerMapper
from landmark_cover import LandmarkCover
from sklearn.cluster import AgglomerativeClustering

# setup KeplerMapper object
mapper = KeplerMapper(verbose=1)

# run KeplerMapper using the LandmarkCover
graph = mapper.map(
    lens=X, X=X,
    cover=LandmarkCover(n_landmarks=30, perc_overlap=0.35, metric='euclidean'),
    clusterer=AgglomerativeClustering(n_clusters=3, linkage='single'),
    remove_duplicate_nodes=True,
)

# visualize the graph
html = mapper.visualize(graph, path_html=f'kmapper_landmark_cover.html')

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/kmapper_landmark_cover.png?raw=true">
</a>
</p>



### `LandmarkCover` and higher dimensional lenses

Now, let's explore a high dimensional lens based on pairwise distances (i.e., since `X` is already low dimensional). 

First, we can compute geodesic distances on a reciprocal neighbor graph using the [reciprocal_isomap](https://github.com/calebgeniesse/reciprocal_isomap) package. Note, in the example below, we aren't fitting the `ReciprocalIsomap` model, just using the internal `_reciprocal_distances` method to compute reciprocal geodesic distances.

```python
from reciprocal_isomap import ReciprocalIsomap

# compute geodesic distances on a reciprocal neighbor graph
r_isomap = ReciprocalIsomap(n_neighbors=8, neighbors_mode='connectivity')
geodesic_distances = r_isomap._reciprocal_distances(X).toarray()

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/geodesic_lens.png?raw=true" width="50%">
</a>
</p>



Now, let's use the geodesic distances as a lens for `KeplerMapper`. 

```python
import numpy as np 
from kmapper import KeplerMapper
from landmark_cover import LandmarkCover
from sklearn.cluster import AgglomerativeClustering

# setup KeplerMapper object
mapper = KeplerMapper(verbose=1)

# run KeplerMapper using the LandmarkCover
graph = mapper.map(
    lens=geodesic_distances, X=X,
    cover=LandmarkCover(n_landmarks=30, perc_overlap=0.35, metric='precomputed'),
    clusterer=AgglomerativeClustering(n_clusters=3, linkage='single'),
    remove_duplicate_nodes=True,
)

# visualize the graph
html = mapper.visualize(graph, path_html=f'kmapper_landmark_cover_geodesic_lens.html')

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/kmapper_landmark_cover_geodesic_lens.png?raw=true">
</a>
</p>







### Comparison with `CubicalCover`

Below, we compare graphs obtained using the `CubicalCover` and `LandmarkCover`. 

#### `CubicalCover` with a 2D lens

```python
from kmapper import KeplerMapper
from kmapper.cover import CubicalCover
from sklearn.cluster import AgglomerativeClustering

# setup KeplerMapper object
mapper = KeplerMapper(verbose=1)

# create low dimensional lens 
lens = mapper.fit_transform(X, projection=[0,1])

# run KeplerMapper using the CubicalCover
graph = mapper.map(
    lens=lens, X=X,
    cover=CubicalCover(n_cubes=8, perc_overlap=0.67),
    clusterer=AgglomerativeClustering(n_clusters=2, linkage='single'),
    remove_duplicate_nodes=True,
)

# visualize the graph
html = mapper.visualize(graph, path_html=f'kmapper_cubical_cover.html')

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/kmapper_cubical_cover.png?raw=true">
</a>
</p>




#### `LandmarkCover` with geodesic distances

```python
from kmapper import KeplerMapper
from landmark_cover import LandmarkCover
from sklearn.cluster import AgglomerativeClustering
from reciprocal_isomap import ReciprocalIsomap


# setup KeplerMapper object
mapper = KeplerMapper(verbose=1)

# compute geodesic distances on a reciprocal neighbor graph
r_isomap = ReciprocalIsomap(n_neighbors=8, neighbors_mode='connectivity')
geodesic_distances = r_isomap._reciprocal_distances(X).toarray()

# run KeplerMapper using the LandmarkCover
graph = mapper.map(
    lens=geodesic_distances, X=X,
    cover=LandmarkCover(n_landmarks=30, perc_overlap=0.35, metric='precomputed'),
    clusterer=AgglomerativeClustering(n_clusters=3, linkage='single'),
    remove_duplicate_nodes=True,
)

# visualize the graph
html = mapper.visualize(graph, path_html=f'kmapper_landmark_cover_geodesic_lens.html')

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/kmapper_landmark_cover_geodesic_lens.png?raw=true">
</a>
</p>





## **Citation**

If you find Landmark Cover useful, please consider citing:
> Geniesse, C., Chowdhury, S., & Saggar, M. (2022). [NeuMapper: A Scalable Computational Framework for Multiscale Exploration of the Brain's Dynamical Organization](https://doi.org/10.1162/netn_a_00229). *Network Neuroscience*, Advance publication. doi:10.1162/netn_a_00229








