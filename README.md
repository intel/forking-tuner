# Kamerton:  The Forking Tuner for TensorFlow

## What It Does

Kamerton is a pure python library with no dependencies that automatically finds
the optimal number of inter- and intra-op threads for a TensorFlow model.


## Installation

    git clone https://gitlab.devtools.intel.com/intelai/kamerton
    cd kamerton
    python setup.py install


## Usage

If you are trying to find the best threading configuration for some code like:

    res = ResNet50()
    res.predict(np.random.rand(32, 224, 224, 3), 32)

You could wrap it using the `kamerton` generator like so:
    
    import timeit
    from kamerton import kamerton, set_threading

    # `cb` is a callback that will set the inter- and intra-op thread count for
    # each iteration
    # the vertex represents the initial thread counts
    for attempt in kamerton(cb=set_threading, vertex=[22, 2]):
      # the model has to be created with the threading configuration, and therefore
      # has to be instantiated inside this loop
      res = ResNet50()
      print(attempt)  # will output the best vertex, i.e. inter- and intra- count
      # kamerton will use the last printed value as its objective for the attempt
      # you will only see the last, i.e. best, value printed
      print(timeit.timeit(lambda: res.predict(np.random.rand(32, 224, 224, 3),
                                              32), number=1))

Note that this particular example would perform poorly due to the timing
variance of predicting only one batch.

For more examples please consult `kamerton/examples/`


## How It Works

Kamerton implements the
[Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
method to find the local minimum for the
optimal threading configuration.  However, because the threading model for
TensorFlow cannot be modified once the first tensor in the application has been
created, Kamerton forks a child process for each iteration that then configures
threads, runs the workload, and reports the results back to the parent.

Though the `kamerton` invocation looks like a python generator that runs in a
loop, it actually ends up executing the inner code once per spawned process.
