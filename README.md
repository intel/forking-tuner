# DISCONTINUATION OF PROJECT #
This project will no longer be maintained by Intel.
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.
Intel no longer accepts patches to this project.
# Forking Tuner:  The Forking Tuner

## What It Does

Forking Tuner is a pure python library with no dependencies that was originally
intended to optimize the number of inter- and intra-op threads for TensorFlow
models, but it can be used to minimise any other metrics.


## Installation

    git clone https://gitlab.devtools.intel.com/intelai/forking-tuner
    cd forking-tuner
    python setup.py install


## Usage

If you are trying to find the best threading configuration for some code like:

    res = ResNet50()
    res.predict(np.random.rand(32, 224, 224, 3), 32)

You could wrap it using the `forking_tuner` generator like so:
    
    import timeit
    from forking_tuner import nelder_mead
    from forking_tuner.tf import set_threading

    # the vertex here represents the initial thread counts
    for attempt in nelder_mead(vertex=[22, 2]):
      set_threading(attempt)
      # the model has to be created with the threading configuration, and therefore
      # has to be instantiated after the threading has been set
      res = ResNet50()
      print(attempt)  # will output the best vertex, i.e. inter- and intra- count
      # forking_tuner will use the last printed value as its objective for the attempt
      # you will only see the last, i.e. best, value printed
      print(timeit.timeit(lambda: res.predict(np.random.rand(32, 224, 224, 3),
                                              32), number=1))

Note that this particular example would perform poorly due to the timing
variance of predicting only one batch.

For more examples please consult `forking_tuner/examples/`


## How It Works

Forking Tuner implements the
[Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
method to find the local minimum for the
optimal threading configuration.  However, because the threading model for
TensorFlow cannot be modified once the first tensor in the application has been
created, Forking Tuner forks a child process for each iteration that then configures
threads, runs the workload, and reports the results back to the parent.

Though the `forking_tuner` invocation looks like a python generator that runs in a
loop, it actually ends up executing the inner code once per spawned process.


## Limitations

Forking Tuner does not work within Jupyter notebooks because the forking approach
would attempt to fork the entire notebook.


## Running Tests

    pip install tox
    tox
