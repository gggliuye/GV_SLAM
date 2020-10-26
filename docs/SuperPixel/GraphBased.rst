Graph Based
=====================

Each pixel is treated as a node in a graph, and edge weight between nodes are set proportional ti the similarity
between pixels (normally, pixel distances and color differences.). Graph could be built based on grid sample, or KNN, etc.
Finally, the superpixel segments are extracted by effectively minimizing a cost function defined on the graph.

1. Felzenszwalb's
---------------------------

`My implementation <https://github.com/gggliuye/graph_based_image_segmentation>`_ based on
*Efficient graph-based image segmentation 2004*

`Paper <http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf>`_ and `Link <https://vio.readthedocs.io/zh_CN/latest/Other/PGM.html>`_ to my documentation about PGM segmentation.

.. image:: images/allresults.jpg
   :align: center
   :width: 60%

2. Entropy Rate
-------------------

`Entropy Rate Superpixel Segmentation <https://www.merl.com/publications/docs/TR2011-035.pdf>`_
guide the superpixel image segmentation using a clustering function consists of two terms :
(1) the entropy rate of a random walk in graph (that favors compact and homogeneous clusters),
(2) a balancing term in the cluster distribution (encourages similar sizes of clusters).

2.1 Information Theory
~~~~~~~~~~~~~~~~~~~

Entropy H and Entropy rate H':

.. math::
  H(X) = - \sum_{x\in X}p_{X}(x)\log(p_{X}(x))

.. math::
  H'(X) = \lim_{t\to \infty}H(X_{t}\mid X_{t-1}, X_{t-2},...,X_{1})

Giving the remaining uncertainty of a random variable :math:`X_{t}` given the values of the
history X known (which is to say, given the past trajectory). Since a stochastic process defined by a
Markov chain that is irreducible, aperiodic and positive recurrent has a stationary distribution,
the entropy rate is independent of the initial distribution.

2.2 Monotone & Submodularity
