# landmark_cover

A landmark-based cover for KeplerMapper.

`LandmarkCover` was designed to work with [KeplerMapper](https://scikit-tda.org/kepler-mapper). The class is derived from the `kmapper.cover.Cover` base class, and departs from the existing `kmapper.cover.CubicalCover` class by adding an additional landmark selection step during the fitting process, and then computing an "intrinsic" cover based on distances between the data and these landmarks during the transform process.



## Setup

### Dependencies

#### [Python 3.6+](https://www.python.org/)

#### Required Python Packages
* [numpy](https://www.numpy.org)
* [scipy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org)
* [kmapper](https://scikit-tda.org/kepler-mapper)


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
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/trefoil_knot.png?raw=true">
</a>
</p>



### The `LandmarkCover` object

Here, we will walk through a simple example using `LandmarkCover` with `KeplerMapper`. 

```python
from kmapper import KeplerMapper
from landmark_cover import LandmarkCover
from sklearn.cluster import AgglomerativeClustering

# setup KeplerMapper object
mapper = KeplerMapper(verbose=1)

# create low dimensional lens 
lens = mapper.fit_transform(X, projection=[0,1])

# run KeplerMapper using the LandmarkCover
graph = mapper.map(
    lens=lens, X=X,
    cover=LandmarkCover(n_landmarks=30, perc_overlap=0.33),
    clusterer=AgglomerativeClustering(n_clusters=3, linkage='single'),
    remove_duplicate_nodes=True,
)

# visualize the graph
html = mapper.visualize(graph, path_html=f'kmapper_landmark_cover.html')

```


### `LandmarkCover` and higher dimensional lenses

Unlike `CubicalCover`, the computational complexity of `LandmarkCover` does not depend on the dimensionality of the lens. This means we can use higher dimensional embeddings as lenses for `KeplerMapper` without suffering the exponential cost (i.e., `O(n_cubes ** lens.shape[1])`) of the more traditional grid-based cover. For example, we can fit the `LandmarkCover` to the entire data matrix, or any subset thereof. Since the data we are using is already relatively low dimensional, below we use geodesic distances as an example of a *higher* dimensional lens. 

First, let's compute geodesic distances on a reciprocal neighbor graph. For simplicity, we utilize the [reciprocal_isomap](https://github.com/calebgeniesse/reciprocal_isomap) package, which implements a variant of the `Isomap` algorithm that uses a reciprocal neighbor matrix under the hood. Note, in the example below, we aren't fitting the `ReciprocalIsomap` model, just using the internal `_reciprocal_distances` method to compute reciprocal geodesic distances.

```python
from reciprocal_isomap import ReciprocalIsomap

# compute geodesic distances on a reciprocal neighbor graph
r_isomap = ReciprocalIsomap(n_neighbors=8, neighbors_mode='connectivity')
geodesic_distances = r_isomap._reciprocal_distances(X).toarray()

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/geodesic_lens.png?raw=true">
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
    cover=LandmarkCover(n_landmarks=30, perc_overlap=0.33),
    clusterer=AgglomerativeClustering(n_clusters=3, linkage='single'),
    remove_duplicate_nodes=True,
)

# visualize the graph
html = mapper.visualize(graph, path_html=f'kmapper_landmark_cover_geodesic_lens.html')

```
https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/geodesic_lens.png
<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/kmapper_landmark_cover_geodesic_lens_with_summary.png?raw=true">
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
    cover=CubicalCover(n_cubes=8, perc_overlap=0.33),
    clusterer=AgglomerativeClustering(n_clusters=3, linkage='single'),
    remove_duplicate_nodes=True,
)

# visualize the graph
html = mapper.visualize(graph, path_html=f'kmapper_cubical_cover.html')

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/kmapper_cubical_cover_with_summary.png?raw=true">
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
    cover=LandmarkCover(n_landmarks=30, perc_overlap=0.33),
    clusterer=AgglomerativeClustering(n_clusters=3, linkage='single'),
    remove_duplicate_nodes=True,
)

# visualize the graph
html = mapper.visualize(graph, path_html=f'kmapper_landmark_cover_geodesic_lens.html')

```

<p align="center">
<a href="https://github.com/calebgeniesse/landmark_cover/tree/main/examples/trefoil_knot/">
<img src="https://github.com/calebgeniesse/landmark_cover/blob/main/examples/trefoil_knot/kmapper_landmark_cover_geodesic_lens_with_summary.png?raw=true">
</a>
</p>






