#ifndef KERNEL_HOOK_H
#define KERNEL_HOOK_H
#include <map>
#include <list>
#include <memory>
#include <pybind11/pybind11.h>
namespace kernel_cache {

struct LaunchConfigParams {
    /* data */
    int nclusters;
    int ncores;
    void* stream;
    // XPUStream stream;
    LaunchConfigParams() {}
    LaunchConfigParams(int cl, int co, void* s) : nclusters(cl), ncores(co), stream(s) {}
    LaunchConfigParams(const LaunchConfigParams& rhs);
};

struct LaunchArgSetParams {
    /* data */
    // const void* arg;
    char* arg;
    size_t size;
    size_t offset;
    LaunchArgSetParams() {}
    LaunchArgSetParams(char* a, size_t s, size_t o) : arg(a), size(s), offset(o) {}
    LaunchArgSetParams(const LaunchArgSetParams& rhs);
    ~LaunchArgSetParams() {
        if (arg != nullptr) {
            delete[] arg;
        }
    }
};

typedef std::list<LaunchArgSetParams> LaunchArgSetParamsList;

struct LaunchKernelParams {
    /* data */
    void* func;
    LaunchKernelParams() {}
    LaunchKernelParams(void* f) : func(f) {}
    LaunchKernelParams(const LaunchKernelParams& rhs);
};

struct KernelCacheEntry {
    LaunchConfigParams config_params_;
    LaunchArgSetParamsList arg_set_params_list_;
    LaunchKernelParams kernel_params_;
};

typedef std::list<KernelCacheEntry> KernelCacheEntryList;

class OpCacheEntry {
public:
    OpCacheEntry() {}
    OpCacheEntry(std::string name) : op_name_(name) {}

    /// execute the op
    void execute();
    /// add KernelCaccheEntry to list
    void add_entry(KernelCacheEntry& entry);
    /// the KernelCacheEntry size
    int64_t size();

private:
    std::string op_name_;
    std::shared_ptr<std::list<KernelCacheEntry>> kernel_cache_entry_list_;
};

class GraphCacheEntry {
public:
    GraphCacheEntry() {}
    /// execute the graph
    void execute();
    /// add entry to list
    void add_entry(OpCacheEntry& entry);
    /// the KernelCacheEntry size
    int64_t size();

private:
    // std::shared_ptr<std::list<KernelCacheEntry>> kernel_cache_entry_list_;
    std::shared_ptr<std::list<OpCacheEntry>> op_cache_entry_list_;
};

class KernelCache {
public:
    KernelCache() : capture_graph_(false), capture_op_(false) {}
    ~KernelCache() {}

    std::shared_ptr<GraphCacheEntry> get(int64_t key);

    bool enable_capture() {
        return capture_graph_;
    }

    /// capture launch params, and set the key for current graph
    void start_capture_launch_params(int64_t key);

    /// stop capture launch params
    void stop_capture_launch_params();

    /// start capture launch params for one op
    void start_capture_op(std::string name);

    /// stop capture launch params for one op
    void stop_capture_op();

    /// get current GraphCacheEntry
    GraphCacheEntry get_graph_cache_entry();

    OpCacheEntry get_op_cache_entry();

    void register_graph_cache_entry(int64_t key, GraphCacheEntry& entry);

private:
    // KernelCache() : capture_graph_(false), capture_op_(false) {}
    std::map<int64_t, std::shared_ptr<GraphCacheEntry>> graph_entry_map_;
    GraphCacheEntry graph_cache_entry_;
    OpCacheEntry op_cache_entry_;
    bool capture_graph_;
    bool capture_op_;
    int64_t key_;
    // std::string op_name_;
};

void init_kernel_cache(pybind11::module& m);

void install_hook();

}    // namespace kernel_cache

#endif