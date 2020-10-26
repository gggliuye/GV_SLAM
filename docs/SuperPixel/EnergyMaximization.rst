Energy Maximization
==========================

Superpixel problem could also be seen as an **Energy Maximization** problem.
It has close relation to **Mumford-Shah Functional**, could refer to `my documentation <https://cvx-learning.readthedocs.io/en/latest/PaperRead/PrimalDualMumford.html>`_
for more details. And we could refer to `TV reconstruction <https://cvx-learning.readthedocs.io/en/latest/PaperRead/SolvingTVviaADMM.html>`_ for a solution based on convex optimization.

1. SEEDS
----------------

1.1 Energy function
~~~~~~~~~~~~~~~~

**Energy function** consists of a color-distribution term H and a boundaries term G :

.. math::
  E(s) = H(s) + \gamma G(s)

1.2 Color Distribution Term
~~~~~~~~~~~~~~~~~~~~

**Color Distribution**, sum of the evaluation of each superpixel:

.. math::
  H(x) = \sum_{k}\Phi(c_{A_{k}})

We want the pixel in each superpixel to concentrate a subset of colors (:math:`\mathcal{H}_{j}`).
Then we have :math:`c_{A_{k}}` the color histogram of pixels in :math:`A_{k}`, w.r.t the color subsets:

.. math::
  c_{A_{k}(j)} = \frac{1}{Z}\sum_{i\in A_{k}}\sigma (I(i)\in \mathcal{H}_{j})

Then we used the following measure (as it is more efficient for hill-climbing) :

.. math::
  \Phi(c_{A_{k}}) = \sum_{\{ \mathcal{H}_{j} \} } (c_{A_{k}(j)})^{2}

If all the colors are in one color bin, the upper function puts 1, while in other cases, the outputs are smaller.
But it does not take into account whether the colors are placed in bins far apart in the histogram or not.

1.3 Boundary Term
~~~~~~~~~~~~~~~~

It evaluates the shape of the superpixel, and it penalizes local irregualrities in the superpixel boundaries.

A image is divided in to N*N patches. The histogram (of K bins, each represents one superpixel)
of superpixel labels in an area is :

.. math::
  b_{N_{i}}(k) = \frac{1}{Z}\sum_{j\in N_{i}}\sigma (j\in A_{k})

The author considers that a superpixel has a better shape when most of the patches contain pixels from one unique superpixel.

.. math::
  G(s) = \sum_{i}\sum_{k} (b_{N_{i}}(k))^{2}

BT: This term is not directly pointing to the smoothness of the boundaries, while the experiments in
the paper show that it works.

1.4 Hill-Climbing
~~~~~~~~~~~~~

The update steps are done for different levels, from corase to fine.

The author developped two proposition to accelerate the update of the superpixels.

**Proposition 1.** Let the sizes of :math:`A_{k}` and :math:`A_{n}` (two superpixels) be similar .
i.e. :math:`\mid A_{k} \mid \approx \mid A_{n}\mid \gg \mid A_{k}^{l} \mid `. If the histogram
and :math:`A_{k}^{l}` (pixels in :math:`A_{k}` to move into :math:`A_{n}`) much smaller,
of :math:`A_{k}^{l}` is concentrated in a single bin, then :

.. math::
  int(c_{A_{n}}, c_{A_{k}^{l}}) \ge int(c_{A_{n}\ A_{k}^{l}}, c_{A_{k}^{l}}) 	\iff H(s) \ge H(s_{t})

**Proposition 2.** Let {:math:`b_{N_{i}}(k)`} be the histograms of the superpixel labellings computed
at the partitioning :math:`s_{t}`. :math:`A_{k}^{l}` is a pixel, and :math:`K_{A_{k}^{l}}` the set of pixels
whose patch intersects with that pixel, i.e. :math:`K_{A_{k}^{l}} = \{ i:A_{k}^{l} \in N_{i} \}`. If the
hill-climbing propose moving a pixel :math:`K_{A_{k}^{l}}` form superpixel k to superpixel n, then:

.. math::
  \sum_{i\in K_{A_{k}^{l}}} (b_{N_{i}}(n) + 1) \ge \sum_{i\in K_{A_{k}^{l}}} b_{N_{i}}(k) \iff G(s) \ge G(s_{t})
