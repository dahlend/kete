Installation
============

These instructions are the method to install kete.

Requirements
------------

- A working installation of Python with a minimum version of `3.9`.

- Any Mac/Windows/Linux machine except Mac using Intel Processors.
  For these machines, installation from source is required (see below).

Steps
-----

Once Python is installed, then installation should just require opening a
terminal window and running the following command:

.. code-block:: console

    pip install kete


Installation from Source
========================

These instructions are the method to install kete from source, it assumes that you
have already downloaded a copy of the source code.

Requirements
------------

- A working installation of Python with a minimum version of `3.9`.

- Installation from the source code additionally requires an installation of the `Rust
  programming language <https://www.rust-lang.org/>`_. Installation instructions for this
  may be found here:

      https://www.rust-lang.org/learn/get-started

Steps
-----

Once Python and Rust are installed, then installation should just require opening a
terminal window to the folder where the source is located and running the following
command:

.. code-block:: console

    # cd /kete/source/directory
    pip install .