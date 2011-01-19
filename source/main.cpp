
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <tbb/tbb.h>
#include <boost/bind.hpp>

using namespace std;
using namespace tbb;


// title:      Initial experiment with serial and parallel versions of running stat
// hypothesis: TBB provides primitives to trivially parallelize bulk data processing.
//             For data processing heavy tasks, this can lead to increased performance with a minimal increase in
//             programmer time, and without excessive increase in code complexity.
//             This could have applications in analytics, scientific computing, algorithmic trading, etc.
// method:     implement serial versions of a simple but useful data processing algorithm (running stats).
//             implement a parallel version, attempting to keep structure and readability as close to 
//             the serial version as possible.
//             Measure performance. (multiple runs. throwing out first run)
//             TBB will be considered worthwhile for further experiments if
//              * performance of parallel version is greater than serial version
//              * code is not radically different or unreadable
//              * programmer does not have to deal with error prone tasks (eg thread synchronization)
//             Ease of use is at least as important as performance in this experiment.
// result:
//             On my quad core Q6600 2.4ghz
//              serial
//               n:      10000000
//               mean:   1.0005e+006
//               var:    83333.7
//               stddev: 288.676
//               mean runtime: 0.203547s
//               runtime stddev: 0.00385597s
//              parallel
//               n:        10000000
//               mean:     1.0005e+006
//               var:      83333.7
//               stddev:   288.676
//               mean runtime: 0.0570485s
//               runtime stddev: 0.00592584s
//              basic loop
//               mean runtime: 0.016917s
//               runtime stddev: 0.00241816s
//
// conclusion: 
//             The code necessary to parallelize this simple bulk data operation is trivial, 
//             and does not deal make the programmer deal with error-prone tasks (like manual synchronization)
//             A speedup was achieved by the parallel version. 
//             I conclude that TBB is a worthy tool to become familiar with,
//             and should be one of the first tools to reach for when attempting to parallelize a task.
// caveats:    This is very much a micro benchmark, and as such it is almost certainly not a good indicator
//             of performance improvement (or otherwise) you can expect for other tasks. 
//             This particular task is sensitive to system load. the experiment was done on an otherwise idle system.
// extensions: Cuda+Thrust look like a nice way to easily exploit GPUs for similar tasks.
//             Would need to investigate potential latency and chunk size issues.
//             FastFlow, OpenMP, OMPTL could also be worth looking at. 


// number of times to run each test. we throw away the first runs timing.
static const int test_runs = 100;

static void generate_samples(vector<double>& v, int n) {
    v.clear();
    v.reserve(n);
    // sample data. dont really care what the values are or even the distribution.
    // produce random value between [1000000-1001000]
    srand(0);
    struct SampleGenerator {
        double operator()() const {
            double d = 1e6 + double(rand())/double(RAND_MAX)*1000.0;
            return d;
        }
    };
    generate_n( back_inserter(v), n, SampleGenerator() );
}


// uses walfords method
// borrowed (and altered) from http://www.johndcook.com/standard_deviation.html
// modified to support merging stats using technique from
// http://blog.cordiner.net/2010/06/16/calculating-variance-and-mean-with-mapreduce-python/
class RunningStats{
public:
    RunningStats(): m_n(0), m_mean(0.0), m_s(0.0) {}

    void push(double x) {
        m_n++;
        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (1 == m_n)
        {
            m_mean = x;
        }
        else
        {
            double old_m = m_mean;
            double old_s = m_s;
            m_mean = old_m + (x - old_m)/m_n;
            m_s = old_s + (x - old_m)*(x - m_mean);
        }
    }

    void merge(const RunningStats& b) {
        int n_a = n();
        int n_b = b.n();
        double mean_a = mean();
        double mean_b = b.mean();
        double variance_a = variance();
        double variance_b = b.variance();

        int n = n_a + n_b;
        double mean = (mean_a*n_a + mean_b*n_b) / n;
        double variance = (variance_a*n_a + variance_b*n_b)/n
            + n_a*n_b * pow((mean_b-mean_a)/n,2);

        m_n = n;
        m_mean = mean;
        m_s = variance * double(n-1);
    }

    int n() const { return m_n; }
    double mean() const { return (m_n>0)?m_mean:0.0; }
    double variance() const { return (m_n>1)?m_s/double(m_n-1):0.0; }
    double stddev() const { return sqrt(variance()); }
private:
    int m_n;
    double m_mean;
    double m_s;
};

static void test_serial_running_variance(const vector<double>& samples) {
    using namespace boost;
    RunningStats runtime_stats;
    RunningStats results;
    for (int run=0; run<test_runs; ++run)
    {
		results = RunningStats();	// reset stats for this run
        tick_count start = tick_count::now();
        for_each(samples.begin(), samples.end(), 
            bind(&RunningStats::push, &results, _1 )
            );
        tick_count stop = tick_count::now();
        tick_count::interval_t delta = stop-start;
        if (run>0)
            runtime_stats.push(delta.seconds());
    }
    cout << "serial" << std::endl;
    cout << " n:      " << results.n() << std::endl; 
    cout << " mean:   " << results.mean() << std::endl; 
    cout << " var:    " << results.variance() << std::endl; 
    cout << " stddev: " << results.stddev() << std::endl; 
    cout << " mean runtime: " << runtime_stats.mean() << "s" << std::endl; 
    cout << " runtime stddev: " << runtime_stats.stddev() << "s" << std::endl; 
}



// ParallelRunningStats is a model of the Parallel Reduce Body concept
// see http://www.threadingbuildingblocks.org/files/documentation/parallel_reduce_body_req.html
class ParallelRunningStats {
public:
    ParallelRunningStats(const vector<double>& samples):m_samples(&samples) {}
    // splitting constructor. 
    ParallelRunningStats(const ParallelRunningStats& b, split):m_samples(b.m_samples) {}

    ~ParallelRunningStats() {}

    // this is applied to subsections of our samples vector
    // (map step)
    void operator()(const blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i<r.end(); ++i) {
            m_stats.push((*m_samples)[i]);
        }
    }

    // join our sub results.
    // (reduce step)
    void join(const ParallelRunningStats& b) {
        m_stats.merge(b.stats());
    }

    // return value from parallel reduce body
    const RunningStats& stats() const { return m_stats; }

private:
    const vector<double>* m_samples;
    RunningStats m_stats;
};

static void test_parallel_running_variance(const vector<double>& samples) {
    RunningStats runtime_stats;
    ParallelRunningStats results(samples);
    for (int run=0; run<test_runs; ++run)
    {
		results = ParallelRunningStats(samples);	// reset stats
        tick_count start = tick_count::now();
        parallel_reduce( blocked_range<size_t>(0,samples.size()), results );
        tick_count stop = tick_count::now();
        tick_count::interval_t delta = stop-start;
        if (run>0)
            runtime_stats.push(delta.seconds());
    }
    cout << "parallel" << std::endl;
    RunningStats result_stats = results.stats();
    cout << " n:        " << result_stats.n() << std::endl; 
    cout << " mean:     " << result_stats.mean() << std::endl; 
    cout << " var:      " << result_stats.variance() << std::endl; 
    cout << " stddev:   " << result_stats.stddev() << std::endl; 
    cout << " mean runtime: " << runtime_stats.mean() << "s" << std::endl; 
    cout << " runtime stddev: " << runtime_stats.stddev() << "s" << std::endl; 
}

// loop through the array as fast as possible
// This gives us a lower bound on runtime.
static void test_basic_loop(const vector<double>& samples) {
    RunningStats runtime_stats;
    for (int run=0; run<test_runs; ++run)
    {
        tick_count start = tick_count::now();
        size_t s = samples.size();
        volatile const double* samples_ptr = &samples.at(0);
        // change stride to sizeof(cache line)/sizeof(double) to verify that you are mem bound.
        // (this reduces CPU work but still keeps number of cache misses the same)
        // if it has the same runtime as stride=1 then we are mem bound (as we expect to be).
        const int stride = 1;
        for(size_t i=0; i<s/stride; ++i){
            samples_ptr[i*stride];
        }
        tick_count stop = tick_count::now();
        tick_count::interval_t delta = stop-start;
        if (run>0)
            runtime_stats.push(delta.seconds());
    }
    cout << "basic loop" << std::endl;
    cout << " mean runtime: " << runtime_stats.mean() << "s" << std::endl; 
    cout << " runtime stddev: " << runtime_stats.stddev() << "s" << std::endl; 
}

int main() {
    vector<double> samples;
    srand(0);
    generate_samples(samples, 10000000);
	// can be interesting to play with the number of threads TBB uses to compare performance.
    //int num_threads = 2*tbb::task_scheduler_init::default_num_threads();
    //tbb::task_scheduler_init init( num_threads );
    test_serial_running_variance(samples);
    test_parallel_running_variance(samples);
    test_basic_loop(samples);
    return EXIT_SUCCESS;
}
