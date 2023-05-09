.. _development:

===========
Development
===========

Whether you're a seasoned developer or new to Python, there are many ways you can contribute to *PedPy*.
If you're interested in improving the library's functionality, we encourage you to get involved in the development process.
Our code base is open source and available on GitHub, so you can easily contribute by submitting pull requests or reporting issues and bugs.
We also welcome feedback on new features or enhancements that you think would be useful.

If you're not comfortable contributing to the code base just yet, you can still help us by reporting any errors or bugs you encounter while using *PedPy*.
Your feedback is incredibly valuable and helps us to improve the library for everyone.

We're committed to creating a supportive and inclusive community that values collaboration and sharing knowledge.
So don't hesitate to reach out with any questions or ideas you may have. We look forward to seeing what you can accomplish with *PedPy*!

Reporting bugs
==============
At *PedPy*, we value feedback from our users, and we encourage you to report any issues or bugs that you encounter while using the library.
You can do this by visiting our `GitHub issues page <https://github.com/PedestrianDynamics/PedPy/issues>`_.
From there, you can submit a new issue, provide a detailed description of the problem, and even include code snippets or screenshots to help us better understand the issue.

By reporting bugs and issues, you're helping us to improve *PedPy* and make it more robust and reliable for everyone.
So don't hesitate to speak up - we appreciate your input and look forward to working with you to make *PedPy* the best it can be!

Enhancement requests
====================

You can submit feature requests on our `GitHub issues page <https://github.com/PedestrianDynamics/PedPy/issues>`_.
When submitting a feature request, please provide a clear and detailed description of the feature you would like to see added, along with any relevant use cases or examples.

We can't promise to implement every feature request, but we do carefully consider all requests and prioritize them based on their potential impact on the community and feasibility.
Your input is valuable to us and helps us to ensure that *PedPy* is meeting the needs of our users.

So if you have a great idea for a new feature or enhancement, don't hesitate to share it with us. We're excited to hear from you and look forward to continuing to improve *PedPy* together!

Contribute to the code
======================

Version control, Git, and GitHub
--------------------------------

Setting up development environment
----------------------------------

Structure of the code
---------------------

*PedPy* is organizied in the following manner:

::

    PedPy
    ├── docs
    ├── notebooks
    ├── pedpy
    │   ├── data
    │   ├── io
    │   ├── methods
    │   └── plotting
    ├── scripts
    └── tests


* ``PedPy`` is the folder we get when we issue a ``git pull/clone`` command.

* ``docs`` contains the everything related to our online documentation.

* ``notebooks`` holds Jupyter notebooks show casing how *PedPy* can be used.

* ``pedpy`` is the actual Python package directory, where our Python source files reside.

  * ``data`` contains all files related to internally used data structures.

  * ``io`` holds files responsible for reading and writing files into internal structures.

  * ``methods`` the place where the actual analyzing methods reside.
    The underlying Python files are structured in their compute domain, i.e., density, velocity, flow, or profiles.
    Some function may be used in different situations, these should be placed in ``method_utils.py``.

  * ``plotting`` contains everything which will help users to plot some of our results.

* ``scripts`` the place for some handy scripts.

* ``tests`` is the directory, where all our tests reside.


Tests
-----

We use unit and reference tests in our continuous integration (CI) process to ensure that our code is of high quality and that it behaves as expected.

Unit tests are used to test small, isolated parts of our code, such as individual functions or methods.
By testing each part of the code in isolation, we can quickly identify any issues or bugs in that specific area of the codebase.
This helps us catch issues early on, before they can propagate to other parts of the code and become more difficult to fix.
Currently we do not cover everything with unit-tests, this will hopefully change at some point.

Reference tests are used to test the behavior of our code against a known set of inputs and expected outputs.
By comparing the actual output of our code against the expected output, we can quickly identify any issues or bugs that might have been introduced during development.
As we see *PedPy* as successor of *JPSreport* we want to ensure, that we get the same results as with *JPSreport*.
Hence, we use results from *JPSreport* in different scenarios as reference.

As we do not only want to ensure that the results are correct, but also want to provide working examples for new user.
Thus, we check whether all notebooks in our repository work with the latest changes.
This ensures, that users can use these notebooks as reference when setting up their analyzes.

We use *GitHub Actions* to automate our testing, every Pull Request will be automatically trigger the workflow which then runs the tests.
Only Pull Request with succeeding pipelines will be allowed to be merged into the ``main`` branch.


Formatting/Linting
------------------

Except from the functional requirements (see :ref:`Tests`) for changes in the code base, we also have some non-functional requirements.
These will also be checked in our CI process for each Pull Request.

1. **Code formatting:**
To ensure that your Pull Request may get accepted, make sure that the code is formatted with ``black``.
We provide a helper script (``scripts/format.sh``) that will format every file in the correct manner.
To test it locally you can use ``scripts/check-format.sh``.

2. **Docstring style:**
Make sure to check whether every of your new functions has a docstring.
We decided to use Google-docstring style to be used in our project.
You can use `pydocstyle` to check if everything is correct locally.

3. **Type Hints:**
We decided that every function, parameter, return value, etc. should be annotated with type hints, as they make it clearer for users what to expect and what is needed.
For ensuring that no type hint is forgotten we use ``MyPy``.
This can be checked locally via ``python3 -m mypy --config-file mypy.ini pedpy/``

4. **Linting:**
Linting in Python is an important process that helps ensure that our code is consistent and adheres to best practices.
Linting tools like ``pylint`` analyze our code for potential errors, bad practices, and code smells.
This helps us catch issues early on and prevents them from becoming bigger problems down the line.


Update documentation
====================

The documentation is written in **reStructuredText**, which is almost like writing in plain English, and built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__.
The Sphinx Documentation has an excellent `introduction to reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__.
Review the Sphinx docs to perform more complex changes to the documentation as well.

Also important to know about the docs:


How to build the documentation
------------------------------

To build the documentation locally, you need to setup a development environment.
Then navigate to the ``docs/`` directory.
Then install all the needed requirements via:

.. code:: bash

    $ pip install -r requirements.txt

Afterwards you can build the documentation with:

.. code:: bash

    $ sphinx-build -a source build

It will create a new folder ``build/`` in which the websites are built.
To preview it locally, open ``build/index.html`` in any browser of your liking.


Preview changes
---------------

Once, the pull request is submitted, GitHub Actions will automatically build the documentation.
To view the built site:

1. Wait for CI to finish the `docs/readthedocs.org:pedpy` job
2. Click on `Details`
3. Click on the small `View docs` (**not** the large green box!)


Alternatively, you can find the documentation for a pull request, after the CI job has finished, under the following link.
As each pull request has a unique number, you need to substitute for ``<#PR>`` in the link:

.. code:: text

    https://pedpy--<#PR>.org.readthedocs.build/en/<#PR>/

Contact
=======

If you need any help, please feel free to open issues on GitHub or join us in our RocketChat Channel: https://juchat.fz-juelich.de/channel/pedpy-usergroup

