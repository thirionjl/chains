======
Chains
======


.. image:: https://img.shields.io/pypi/v/ml-chains.svg
        :target: https://pypi.python.org/pypi/ml-chains

.. image:: https://img.shields.io/travis/thirionjl/chains.svg
        :target: https://travis-ci.org/thirionjl/chains

.. image:: https://readthedocs.org/projects/ml-chains/badge/?version=latest
        :target: https://ml-chains.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/thirionjl/chains/shield.svg
     :target: https://pyup.io/repos/github/thirionjl/chains/
     :alt: Updates

Mini machine learning library centered around optimizing a cost function
specified as a `computation
graph <http://colah.github.io/posts/2015-08-Backprop/>`__. Optimization
is done by computing partial derivatives by repetitive application of
the `chain rule <https://en.wikipedia.org/wiki/Chain_rule>`__. Hence the
name.

Originally created taking while following coursera's `deeplearning.ai
course <https://www.coursera.org/specializations/deep-learning>`__ but I
wanted a more "from scratch", "generic" and advanced implementation of
the course exercices.

Links
-----
* Free software: MIT license
* Documentation: https://chains.readthedocs.io.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
