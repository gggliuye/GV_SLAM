Allan Variance
==================

`Statistics of Atomic Frequency Standards <http://tf.nist.gov/general/pdf/7.pdf>`_ by David W.Allan.

`Power Spectral Density of Brownian motion despite non-stationary <https://dsp.stackexchange.com/questions/45574/power-spectral-density-of-brownian-motion-despite-non-stationary>`_

`Mean and Covariance of Wiener Process <https://math.stackexchange.com/questions/568391/mean-and-covariance-of-wiener-process>`_


1. Power Spectral Density
-------------------------

**Definition** Power Spectral Density (PSD) of a time series x(t) describes the distribution of power into frequency components
composing that signal.

.. math::
  \hat{x}(f) = \int_{-\infty}^{\infty}e^{-2\pi i ft}x(t)dt

.. math::
  PSD_{x}(f) = S_{x}(f) = \mid \hat{x}(f) \mid^{2}

It represent the squared norm of the fourier transform (which represents the freqency space), it will reflect a distribution of the
magnitude of frequence. And we can further derivate the expression:

.. math::
  \begin{align}
  S_{x}(f) &= \int_{t_{1}=-\infty}^{\infty}e^{-2\pi i ft_{1}}x(t_{1})dt_{1}\int_{t_{2}=-\infty}^{\infty}e^{2\pi i ft_{2}}x(t_{2})dt_{2}\\
  &= \int_{t_{1}=-\infty}^{\infty}\int_{t_{2}=-\infty}^{\infty}e^{-2\pi i f(t_{1}-t_{2})}x(t_{1})x(t_{2})dt_{1}dt_{2}
  \end{align}
