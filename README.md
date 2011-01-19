tbb_parallel_stat
=================
An initial experiment using TBB to implement parallel statistics calculations.

hypothesis
----------
TBB provides high level threading primitives which may be used to trivially parallelize bulk data processing.
For data processing heavy tasks, this can lead to increased performance with a minimal increase in
programmer time, and without excessive increase in code complexity.
This could have applications in analytics, scientific computing, algorithmic trading, etc.

see source/main.cpp for more details

summary
-------
The parallel version is simple to implement, can leverage the serial versions stats class,
improves the performance of calculating various statistics (on my machine), and does not force the programmer
to write error-prone code for manual thread synchronization.

In conclusion TBB looks like a good tool to reach for first when attempting to parallelize a problem across multiple threads.

caveats
-------
This is a microbenchmark. It is not an indicator that TBB is good for any other problem than the one it was applied to here.

