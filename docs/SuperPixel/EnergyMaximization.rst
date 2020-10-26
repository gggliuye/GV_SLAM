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
  \Phi(c_{A_{k}}) = \sum_{\mathcal{H}_{j}} (c_{A_{k}(j)})^{2}

If all the colors are in one color bin, the upper function puts 1, while in other cases, the outputs are smaller.
But it does not take into account whether the colors are placed in bins far apart in the histogram or not.

1.3 Boundary Term
~~~~~~~~~~~~~~~~

It evaluates the shape of the superpixel, and it penalizes local irregualrities in the superpixel boundaries.

A image is divided in to N*N patches. The histogram of superpixel labels in an area is :

.. math::
  b_{N_{i}}(k) = \frac{1}{Z}\sum_{j\in N_{i}}\sigma (j\in A_{k})
