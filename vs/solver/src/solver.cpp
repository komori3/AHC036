#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#include <optional>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <omp.h>
#include <filesystem>
#include <intrin.h>
/* g++ functions */
int __builtin_clz(unsigned int n) { unsigned long index; _BitScanReverse(&index, n); return 31 - index; }
int __builtin_ctz(unsigned int n) { unsigned long index; _BitScanForward(&index, n); return index; }
namespace std { inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); } }
int __builtin_popcount(int bits) {
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >> 16 & 0x0000ffff);
}
/* enable __uint128_t in MSVC */
//#include <boost/multiprecision/cpp_int.hpp>
//using __uint128_t = boost::multiprecision::uint128_t;
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro io **/
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void output(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::output(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void output(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '{'; aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t); return os << '}'; } // tuple out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x); // container out (fwd decl)
template<class S, class T> std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) { return os << '{' << p.first << ", " << p.second << '}'; } // pair out
template<class S, class T> std::istream& operator>>(std::istream& is, std::pair<S, T>& p) { return is >> p.first >> p.second; } // pair in
std::ostream& operator<<(std::ostream& os, const std::vector<bool>::reference& v) { os << (v ? '1' : '0'); return os; } // bool (vector) out
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) { bool f = true; os << '{'; for (const auto& x : v) { os << (f ? "" : ", ") << x; f = false; } os << '}'; return os; } // vector<bool> out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) { bool f = true; os << '{'; for (auto& y : x) { os << (f ? "" : ", ") << y; f = false; } return os << '}'; } // container out
template<class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type> std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; } // container in
template<typename T> auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) { out << t.stringify(); return out; } // struct (has stringify() func) out
/** io setup **/
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } }
iosetup(true); // set false when solving interective problems
/** string formatter **/
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
/** dump **/
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<']'<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
/** timer **/
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 2.9e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 2.9e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};
/** rand **/
struct Xorshift {
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next_u64(); }
    inline uint64_t next_u64() { x ^= x << 7; return x ^= x >> 9; }
    inline uint32_t next_u32() { return next_u64() >> 32; }
    inline uint32_t next_u32(uint32_t mod) { return ((uint64_t)next_u32() * mod) >> 32; }
    inline uint32_t next_u32(uint32_t l, uint32_t r) { return l + next_u32(r - l + 1); }
    inline double next_double() { return next_u32() * e; }
    inline double next_double(double c) { return next_double() * c; }
    inline double next_double(double l, double r) { return next_double(r - l) + l; }
private:
    static constexpr uint32_t M = UINT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
};
/** shuffle **/
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { int r = rnd.next_u32(i); std::swap(v[i], v[r]); } }
/** split **/
std::vector<std::string> split(const std::string& str, const std::string& delim) {
    std::vector<std::string> res;
    std::string buf;
    for (const auto& c : str) {
        if (delim.find(c) != std::string::npos) {
            if (!buf.empty()) res.push_back(buf);
            buf.clear();
        }
        else buf += c;
    }
    if (!buf.empty()) res.push_back(buf);
    return res;
}
std::string join(const std::string& delim, const std::vector<std::string>& elems) {
    if (elems.empty()) return "";
    std::string res = elems[0];
    for (int i = 1; i < (int)elems.size(); i++) {
        res += delim + elems[i];
    }
    return res;
}
/** misc **/
template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); } // fill array
template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }
/** hash **/
namespace aux { template<typename T> inline void hash(std::size_t& s, const T& v) { s ^= std::hash<T>()(v) + 0x9e3779b9 + (s << 6) + (s >> 2); } }
namespace std { template<typename F, typename S> struct hash<std::pair<F, S>> { size_t operator()(const std::pair<F, S>& s) const noexcept { size_t seed = 0; aux::hash(seed, s.first); aux::hash(seed, s.second); return seed; } }; }

/* fast queue */
class FastQueue {
    int front = 0;
    int back = 0;
    int v[4096];
public:
    inline bool empty() { return front == back; }
    inline void push(int x) { v[front++] = x; }
    inline int pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
};

class RandomQueue {
    int sz = 0;
    int v[4096];
public:
    inline bool empty() const { return !sz; }
    inline int size() const { return sz; }
    inline void push(int x) { v[sz++] = x; }
    inline void reset() { sz = 0; }
    inline int pop(int i) {
        std::swap(v[i], v[sz - 1]);
        return v[--sz];
    }
    inline int pop(Xorshift& rnd) {
        return pop(rnd.next_u32(sz));
    }
};

#if 1
inline double get_temp(double stemp, double etemp, double t, double T) {
    return etemp + (stemp - etemp) * (T - t) / T;
};
#else
inline double get_temp(double stemp, double etemp, double t, double T) {
    return stemp * pow(etemp / stemp, t / T);
};
#endif

struct LogTable {
    static constexpr int M = 65536;
    static constexpr int mask = M - 1;
    double l[M];
    LogTable() : l() {
        unsigned long long x = 88172645463325252ULL;
        double log_u64max = log(2) * 64;
        for (int i = 0; i < M; i++) {
            x = x ^ (x << 7);
            x = x ^ (x >> 9);
            l[i] = log(double(x)) - log_u64max;
        }
    }
    inline double operator[](int i) const { return l[i & mask]; }
} log_table;

template <typename T = int>
T ipow(T x, T n) {
    T ret = 1;
    for (T i = 0; i < n; i++) ret *= x;
    return ret;
}



constexpr int N = 600;
constexpr int T = 600;

struct Input {

    const int M;
    const int LA;
    const int LB;
    const std::vector<std::pair<int, int>> uvs;
    const std::vector<int> ts;
    const std::vector<std::pair<int, int>> xys;

    static Input load(std::istream& in) {
        int M, LA, LB;
        in >> M >> M >> LA >> LA >> LB; // ignore N, T
        std::vector<std::pair<int, int>> uvs(M), xys(N);
        std::vector<int> ts(T);
        in >> uvs >> ts >> xys;
        return Input(M, LA, LB, uvs, ts, xys);
    }

    static Input load(int seed) {
        const auto path = format("../../tools_win/in/%04d.txt", seed);
        assert(std::filesystem::is_regular_file(path));
        std::ifstream ifs(path);
        return load(ifs);
    }

private:

    Input(
        const int M_,
        const int LA_,
        const int LB_,
        const std::vector<std::pair<int, int>>& uvs_,
        const std::vector<int>& ts_,
        const std::vector<std::pair<int, int>>& xys_
    ) : M(M_), LA(LA_), LB(LB_), uvs(uvs_), ts(ts_), xys(xys_) {}

};



struct UnionFind {

    std::vector<int> data;
    int set_sz;

    UnionFind() = default;

    explicit UnionFind(size_t sz) : data(sz, -1), set_sz(sz) {}

    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (data[x] > data[y]) std::swap(x, y);
        data[x] += data[y];
        data[y] = x;
        set_sz--;
        return true;
    }

    int find(int k) {
        if (data[k] < 0) return (k);
        return data[k] = find(data[k]);
    }

    int size(int k) {
        return -data[find(k)];
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }

    std::vector<std::vector<int>> groups() {
        int n = (int)data.size();
        std::vector<std::vector<int>> ret(n);
        for (int i = 0; i < n; i++) {
            ret[find(i)].emplace_back(i);
        }
        ret.erase(std::remove_if(ret.begin(), ret.end(), [&](const std::vector<int>& v) {
            return v.empty();
            }), ret.end());
        return ret;
    }
};



std::vector<std::vector<std::vector<int>>> compute_shortest_paths(const std::vector<std::vector<int>>& G) {
    auto paths = make_vector(std::vector<int>(), N, N);
    for (int s = 0; s < N; s++) {
        std::vector<int> dist(N, INT_MAX), prev(N, -1);
        std::queue<int> qu;
        qu.push(s);
        dist[s] = 0;
        while (!qu.empty()) {
            auto u = qu.front(); qu.pop();
            for (int v : G[u]) {
                if (dist[v] != INT_MAX) continue;
                dist[v] = dist[u] + 1;
                prev[v] = u;
                qu.push(v);
            }
        }
        for (int t = 0; t < N; t++) {
            std::vector<int> path;
            int u = t;
            while (true) {
                path.push_back(u);
                u = prev[u];
                if (u == -1) break;
            }
            std::reverse(path.begin(), path.end());
            paths[s][t] = path;
        }
    }
    return paths;
}

std::vector<std::pair<int, int>> compute_critical_edges(
    const Input& input,
    const std::vector<std::vector<std::vector<int>>>& paths,
    const bool tree
) {
    std::vector<std::pair<int, int>> nuvs;
    std::vector<bool> to_visit(N);
    to_visit[0] = true;
    for (int t : input.ts) to_visit[t] = true;
    //constexpr int margin = 50;
    //constexpr int width = 1000;
    //std::vector<cv::Point> points;
    UnionFind uf(N);
    //for (const auto& [x, y] : input.xys) {
    //    points.emplace_back(x + margin, y + margin);
    //}
    //cv::Mat3b img(width + margin * 2, width + margin * 2, cv::Vec3b(255, 255, 255));
    //for (int n = 0; n < N; n++) {
    //    const auto& p = points[n];
    //    if (to_visit[n]) {
    //        cv::circle(img, p, 3, cv::Scalar(0, 0, 255), cv::FILLED);
    //    }
    //    else {
    //        cv::circle(img, p, 2, cv::Scalar(0, 0, 255), cv::FILLED);
    //    }
    //}
    //for (const auto& p : points) {
    //    cv::circle(img, p, 2, cv::Scalar(0, 0, 255), cv::FILLED);
    //}
    for (const auto& [u, v] : input.uvs) {
        if (to_visit[u] && to_visit[v]) {
            if (tree) {
                if (!uf.same(u, v)) {
                    nuvs.emplace_back(u, v);
                    uf.unite(u, v);
                }
            }
            else {
                nuvs.emplace_back(u, v);
                uf.unite(u, v);
            }
            //cv::line(img, points[u], points[v], cv::Scalar(0, 0, 0));
        }
        else {
            //cv::line(img, points[u], points[v], cv::Scalar(224, 224, 224));
        }
    }
    while (true) {
        int min_dist = INT_MAX, min_u, min_v;
        for (int u = 0; u < N; u++) {
            if (!to_visit[u]) continue;
            for (int v = u + 1; v < N; v++) {
                if (!to_visit[v]) continue;
                if (uf.same(u, v)) continue;
                if (chmin(min_dist, (int)paths[u][v].size())) {
                    min_u = u;
                    min_v = v;
                }
            }
        }
        if (min_dist == INT_MAX) break;
        const auto& path = paths[min_u][min_v];
        //dump(min_dist, path);
        for (int i = 1; i < (int)path.size(); i++) {
            int u = path[i - 1], v = path[i];
            if (uf.same(u, v)) continue;
            uf.unite(u, v);
            nuvs.emplace_back(u, v);
            //cv::line(img, points[u], points[v], cv::Scalar(0, 0, 255));
        }
    }
    dump(nuvs.size());
    //cv::imshow("img", img);
    //cv::waitKey();
    return nuvs;
}

void dfs(const std::vector<std::vector<int>>& G, std::vector<bool>& visited, std::vector<int>& path, int p, int u) {
    visited[u] = true;
    path.push_back(u);
    for (int v : G[u]) {
        if (v == p) continue;
        if (visited[v]) continue;
        dfs(G, visited, path, u, v);
    }
}

std::vector<int> compute_initial_A(const Input& input, const std::vector<std::vector<int>>& G) {
    std::vector<bool> visited(N);
    std::vector<int> A;
    dfs(G, visited, A, -1, 0);
    assert(A.size() <= input.LA);
    return A;
}

std::vector<int> compute_tour(const Input& input, const std::vector<std::vector<std::vector<int>>>& paths) {
    std::vector<int> tour;
    int pos = 0;
    tour.push_back(pos);
    for (int npos : input.ts) {
        if (pos == npos) continue;
        auto path = paths[pos][npos];
        for (int i = 1; i < (int)path.size(); i++) {
            tour.push_back(path[i]);
        }
        pos = npos;
    }
    return tour;
}


std::tuple<int, std::vector<int>, std::vector<std::string>> solve(
    const Input& input,
    const std::vector<int>& tour,
    const std::vector<int>& initial_A,
    const int thresh
) {
    std::vector<int> A(initial_A);
    std::vector<std::string> ans;
    std::vector<int> signal(N);
    int idx = 0, score = 0;
    while (idx + 1 < (int)tour.size()) {
        int best_begin = -1;
        int best_nidx = -1;
        for (int begin = 0; begin + input.LB <= (int)A.size(); begin++) {
            for (int i = begin; i < begin + input.LB; i++) {
                signal[A[i]]++;
            }

            int nidx = idx;
            while (nidx + 1 < (int)tour.size() && signal[tour[nidx + 1]]) nidx++;
            if (chmax(best_nidx, nidx)) {
                best_begin = begin;
            }

            for (int i = begin; i < begin + input.LB; i++) {
                signal[A[i]]--;
            }
        }
        if (best_nidx - idx <= thresh && A.size() < input.LA) {
            for (int k = idx + 1; k < std::min((int)tour.size(), idx + 1 + input.LB); k++) {
                A.push_back(tour[k]);
                if (A.size() == input.LA) break;
            }

            for (int begin = 0; begin + input.LB <= (int)A.size(); begin++) {
                for (int i = begin; i < begin + input.LB; i++) {
                    signal[A[i]]++;
                }

                int nidx = idx;
                while (nidx + 1 < (int)tour.size() && signal[tour[nidx + 1]]) nidx++;
                if (chmax(best_nidx, nidx)) {
                    best_begin = begin;
                }

                for (int i = begin; i < begin + input.LB; i++) {
                    signal[A[i]]--;
                }
            }
        }
        score++;
        ans.push_back(format("s %d %d 0", input.LB, best_begin));
        //dump(best_nidx - idx);
        while (idx < best_nidx) {
            idx++;
            ans.push_back(format("m %d", tour[idx]));
        }
        //dump(best_begin, best_nidx);
    }
    //dump(score);
    while (A.size() < input.LA) A.push_back(0);
    return { score, A, ans };
}

std::tuple<int, std::vector<int>, std::vector<std::string>> solve(
    const Input& input,
    const std::vector<int>& tour,
    const std::vector<int>& initial_A
) {
    int best_score = INT_MAX;
    std::vector<int> best_A;
    std::vector<std::string> best_ans;

    int score;
    std::vector<int> A;
    std::vector<std::string> ans;

    for (int thresh = 1;; thresh++) {
        std::tie(score, A, ans) = solve(input, tour, initial_A, thresh);
        if (chmin(best_score, score)) {
            best_A = A;
            best_ans = ans;
            dump(thresh, best_score);
        }
        else {
            break;
        }
    }

    return { best_score, best_A, best_ans };
}


double compute_average_distance(
    const std::vector<int>& nodes,
    const std::vector<std::vector<std::vector<int>>>& paths
) {
    size_t ctr = 0, sum = 0;
    for (int i = 0; i < (int)nodes.size(); i++) {
        int u = nodes[i];
        for (int j = i + 1; j < (int)nodes.size(); j++) {
            int v = nodes[j];
            const auto& path = paths[u][v];
            ctr++;
            sum += path.empty() ? INT_MAX : path.size();
        }
    }
    return (double)sum / ctr;
}

std::vector<std::vector<int>> remove_node(const std::vector<std::vector<int>>& G, int u) {
    auto RG(G);
    for (int v : RG[u]) {
        auto& ws = RG[v];
        ws.erase(std::find(ws.begin(), ws.end(), u));
    }
    RG[u].clear();
    return RG;
}


namespace NGraph {

    struct Perf {
        Timer t;
        const char* const func;
        Perf(const char* func_) : func(func_) {}
        ~Perf() {
            fprintf(stderr, "%s: %.2f [ms]\n", func, t.elapsed_ms());
        }
    };

    std::vector<std::vector<double>> compute_apsp_wf(std::vector<std::vector<double>> G) {
        Perf perf(__FUNCTION__);
        constexpr double inf = 1e9;
        const int N = (int)G.size();
        for (int k = 0; k < N; k++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    chmin(G[i][j], G[i][k] + G[k][j]);
                }
            }
        }
        return G;
    }

    std::vector<std::vector<double>> compute_apsp_djk(
        const std::vector<std::vector<std::pair<double, int>>>& adj
    ) {
        Perf perf(__FUNCTION__);
        constexpr double inf = 1e9;
        const int N = (int)adj.size();
        auto dists = make_vector(inf, N, N);
        using E = std::pair<double, int>;
        using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;
        PQ pq;
        for (int s = 0; s < N; s++) {
            auto& dist = dists[s];
            dist[s] = 0;
            pq.emplace(0, s);
            while (!pq.empty()) {
                auto [cost, u] = pq.top(); pq.pop();
                if (dist[u] < cost) continue;
                for (const auto& [edge_len, v] : adj[u]) {
                    if (chmin(dist[v], dist[u] + edge_len)) {
                        pq.emplace(dist[v], v);
                    }
                }
            }
        }
        return dists;
    }

    std::vector<double> compute_sssp_djk(
        const std::vector<std::vector<std::tuple<double, int, int>>>& adj,
        int s
    ) {
        constexpr double inf = 1e9;
        using E = std::pair<double, int>;
        using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;
        PQ pq;
        auto dist = make_vector(inf, N);
        dist[s] = 0;
        pq.emplace(0, s);
        while (!pq.empty()) {
            auto [cost, u] = pq.top(); pq.pop();
            if (dist[u] < cost) continue;
            for (const auto& [edge_len, v, _] : adj[u]) {
                if (chmin(dist[v], dist[u] + edge_len)) {
                    pq.emplace(dist[v], v);
                }
            }
        }
        return dist;
    }

    std::vector<std::vector<double>> compute_apsp_djk(
        const std::vector<std::vector<std::tuple<double, int, int>>>& adj
    ) {
        Perf perf(__FUNCTION__);
        constexpr double inf = 1e9;
        const int N = (int)adj.size();
        std::vector<std::vector<double>> dists(N);
        for (int s = 0; s < N; s++) {
            dists[s] = compute_sssp_djk(adj, s);
        }
        return dists;
    }

    void add_edge(
        std::vector<std::vector<double>>& dists,
        const std::vector<std::vector<std::pair<double, int>>>& adj,
        int u, int v, double w
    ) {
        Perf perf(__FUNCTION__);
        using E = std::pair<double, int>;
        using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;
        PQ pq;
        for (int s = 0; s < N; s++) {
            auto& dist = dists[s];
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.emplace(dist[v], v);
            }
            else if (dist[v] + w < dist[u]) {
                dist[u] = dist[v] + w;
                pq.emplace(dist[u], u);
            }
            while (!pq.empty()) {
                auto [cost, x] = pq.top(); pq.pop();
                if (dist[x] < cost) continue;
                for (const auto& [edge_len, y] : adj[x]) {
                    if (chmin(dist[y], dist[x] + edge_len)) {
                        pq.emplace(dist[y], y);
                    }
                }
            }
        }
    }

    struct BFSTree {

        static constexpr int inf = 1000000000;

        struct Edge {
            int u, v;
            Edge(int u_ = -1, int v_ = -1) : u(u_), v(v_) {}
        };

        struct Neighbor {
            int v, e;
            Neighbor(int v_ = -1, int e_ = -1) : v(v_), e(e_) {}
        };

        const int N;
        const int root;
        std::vector<Edge> edges; // (u, v)
        std::vector<std::vector<Neighbor>> neighbors;
        std::vector<bool> used;

        std::vector<int> dist;
        std::vector<int> vpar;
        std::vector<int> efrom;

        std::vector<int> removed;
        FastQueue qu;

        BFSTree(const Input& input, int root_) : N((int)input.xys.size()), root(root_) {

            neighbors.resize(input.xys.size());
            for (int e = 0; e < input.M; e++) {
                const auto& [u, v] = input.uvs[e];
                edges.emplace_back(u, v);
                neighbors[u].emplace_back(v, e);
                neighbors[v].emplace_back(u, e);
            }

            used.resize(input.M, true);

            dist.resize(N, inf);
            vpar.resize(N, -1);
            efrom.resize(N, -1);

            std::queue<int> qu;
            dist[root] = 0;
            qu.push(root);
            while (!qu.empty()) {
                int from = qu.front(); qu.pop();
                for (const auto& [to, e] : neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (chmin(dist[to], dist[from] + 1)) {
                        vpar[to] = from;
                        efrom[to] = e;
                        qu.push(to);
                    }
                }
            }
        }

        void dfs(std::vector<int>& removed, int u) {
            for (const auto& [v, e] : neighbors[u]) {
                if (!used[e]) continue;
                if (vpar[v] == u) { // is child node
                    dfs(removed, v);
                }
            }
            vpar[u] = efrom[u] = -1;
            dist[u] = inf;
            removed.push_back(u);
        }

        void remove(int e) {
            assert(used[e]);
            used[e] = false;
            int subroot = -1;
            {
                const auto& [u, v] = edges[e];
                if (dist[u] == inf && dist[v] == inf) return;
                subroot = (dist[u] < dist[v]) ? v : u;
                int parent = (dist[u] < dist[v]) ? u : v;
                if (vpar[subroot] != parent) return; // 最短路ではない
            }
            // dfs で部分木を削除
            removed.clear();
            dfs(removed, subroot);

            // 削除した頂点に隣接していた頂点をキューに入れる
            qu.reset();
            for (int u : removed) {
                for (const auto& [v, e] : neighbors[u]) {
                    if (dist[v] != inf) {
                        qu.push(v);
                    }
                }
            }
            
            // bfs
            while (!qu.empty()) {
                int from = qu.pop();
                for (const auto& [to, e] : neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (chmin(dist[to], dist[from] + 1)) {
                        vpar[to] = from;
                        efrom[to] = e;
                        qu.push(to);
                    }
                }
            }
        }

        void add(int e) {
            assert(!used[e]);
            used[e] = true;
            const auto& [v0, v1] = edges[e];

            if (dist[v0] == inf && dist[v1] == inf) return;

            qu.reset();
            if (dist[v0] + 1 < dist[v1]) {
                dist[v1] = dist[v0] + 1;
                vpar[v1] = efrom[v1] = -1;
                qu.push(v1);
            }
            else if (dist[v1] + 1 < dist[v0]) {
                dist[v0] = dist[v1] + 1;
                vpar[v0] = efrom[v0] = -1;
                qu.push(v0);
            }
            else return;

            // bfs
            while (!qu.empty()) {
                int from = qu.pop();
                for (const auto& [to, e] : neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (chmin(dist[to], dist[from] + 1)) {
                        vpar[to] = from;
                        efrom[to] = e;
                        qu.push(to);
                    }
                }
            }
        }

    };

    struct ShortestPathTree2 {

        using E = std::pair<double, int>; // (cost, vid)
        using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;

        static constexpr double inf = 1e9;

        struct Edge {
            int u, v;
            double w;
            Edge(int u_ = -1, int v_ = -1, double w_ = -1.0) : u(u_), v(v_), w(w_) {}
        };

        struct Neighbor {
            int v, e;
            double w; // 高速化のため
            Neighbor(int v_ = -1, int e_ = -1, double w_ = -1.0) : v(v_), e(e_), w(w_) {}
        };

        const int N;
        const int root;
        std::vector<Edge> edges; // (w, u, v)
        std::vector<std::vector<Neighbor>> neighbors;
        std::vector<bool> used;

        std::vector<double> dist;
        std::vector<int> vpar;
        std::vector<int> efrom;

        ShortestPathTree2(const Input& input, int root_, bool unary = false) : N((int)input.xys.size()), root(root_) {

            neighbors.resize(input.xys.size());
            for (int e = 0; e < input.M; e++) {
                const auto& [u, v] = input.uvs[e];
                const auto& [ux, uy] = input.xys[u];
                const auto& [vx, vy] = input.xys[v];
                double w = unary ? 1.0 : sqrt((ux - vx) * (ux - vx) + (uy - vy) * (uy - vy));
                edges.emplace_back(u, v, w);
                neighbors[u].emplace_back(v, e, w);
                neighbors[v].emplace_back(u, e, w);
            }

            used.resize(input.M, true);

            dist.resize(N, inf);
            vpar.resize(N, -1);
            efrom.resize(N, -1);

            PQ pq;
            dist[root] = 0.0;
            pq.emplace(0.0, root);
            while (!pq.empty()) {
                auto [cost, from] = pq.top(); pq.pop();
                if (cost < dist[from]) continue; // <- いるか？
                for (const auto& [to, e, w] : neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (chmin(dist[to], cost + w)) {
                        vpar[to] = from;
                        efrom[to] = e;
                        pq.emplace(dist[to], to);
                    }
                }
            }
        }

        void dfs(std::vector<int>& removed, int u) {
            for (const auto& [v, e, w] : neighbors[u]) {
                if (!used[e]) continue;
                if (vpar[v] == u) { // is child node
                    dfs(removed, v);
                }
            }
            vpar[u] = efrom[u] = -1;
            dist[u] = inf;
            removed.push_back(u);
        }

        void remove(int e) {
            assert(used[e]);
            used[e] = false;
            int subroot = -1;
            {
                const auto& [u, v, w] = edges[e];
                if (dist[u] == inf && dist[v] == inf) return;
                subroot = (dist[u] < dist[v]) ? v : u;
                int parent = (dist[u] < dist[v]) ? u : v;
                if (vpar[subroot] != parent) return; // 最短路ではない
            }
            // dfs で部分木を削除
            std::vector<int> removed;
            dfs(removed, subroot);

            // 削除した頂点に隣接していた頂点をキューに入れる
            PQ pq;
            for (int u : removed) {
                for (const auto& [v, e, w] : neighbors[u]) {
                    if (dist[v] != inf) {
                        pq.emplace(dist[v], v);
                    }
                }
            }
            // dijkstra
            while (!pq.empty()) {
                auto [cost, from] = pq.top(); pq.pop();
                if (cost < dist[from]) continue; // <- いるか？
                for (const auto& [to, e, w] : neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (chmin(dist[to], cost + w)) {
                        vpar[to] = from;
                        efrom[to] = e;
                        pq.emplace(dist[to], to);
                    }
                }
            }
        }

        void add(int e) {
            assert(!used[e]);
            used[e] = true;
            const auto& [v0, v1, weight] = edges[e];

            if (dist[v0] == inf && dist[v1] == inf) return;

            PQ pq;

            if (dist[v0] + weight < dist[v1]) {
                dist[v1] = dist[v0] + weight;
                vpar[v1] = efrom[v1] = -1;
                pq.emplace(dist[v1], v1);
            }
            else if (dist[v1] + weight < dist[v0]) {
                dist[v0] = dist[v1] + weight;
                vpar[v0] = efrom[v0] = -1;
                pq.emplace(dist[v0], v0);
            }
            else return;

            // dijkstra
            while (!pq.empty()) {
                auto [cost, from] = pq.top(); pq.pop();
                if (cost < dist[from]) continue; // <- いるか？
                for (const auto& [to, e, w] : neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (chmin(dist[to], cost + w)) {
                        vpar[to] = from;
                        efrom[to] = e;
                        pq.emplace(dist[to], to);
                    }
                }
            }
        }

    };

    struct ShortestPathTree {

        struct UnorderedEdge {
            int eid;
            int v0, v1;
            double weight;
            UnorderedEdge(int eid_ = -1, int v0_ = -1, int v1_ = -1, double weight_ = -1.0)
                : eid(eid_), v0(v0_), v1(v1_), weight(weight_) {}
        };

        struct OrderedEdge {
            int eid;
            int s, t;
            double weight;
            OrderedEdge(int eid_ = -1, int s_ = -1, int t_ = -1, double weight_ = -1.0)
                : eid(eid_), s(s_), t(t_), weight(weight_) {}
        };

        struct Vertex {

            int vid;
            double distance_from_root;
            OrderedEdge parent;
            std::vector<OrderedEdge> children;
            std::vector<UnorderedEdge> neighbors;

            int remove(int eid) {
                int nerased = 0;
                if (parent.eid == eid) {
                    parent = { -1, -1, -1, -1.0 };
                    nerased++;
                }
                for (int i = 0; i < (int)children.size(); i++) {
                    if (children[i].eid == eid) {
                        children.erase(children.begin() + i);
                        nerased++;
                    }
                }
                for (int i = 0; i < (int)neighbors.size(); i++) {
                    if (neighbors[i].eid == eid) {
                        neighbors.erase(neighbors.begin() + i);
                        nerased++;
                    }
                }
                assert(nerased >= 1);
                return nerased;
            }

            void remove_child(int vid) {
                int eid = -1;
                for (int i = 0; i < (int)children.size(); i++) {
                    if (children[i].t == vid) {
                        eid = i;
                        break;
                    }
                }
                if (eid == -1) return;
                assert(eid != -1);
                children.erase(children.begin() + eid);
            }

        };

        static constexpr double inf = 1e9;

        const int root_index;
        std::vector<UnorderedEdge> edges; // 削除辺も含む
        std::vector<bool> used;
        std::vector<Vertex> vertices;


        void add(int eid) {
            assert(!used[eid]);
            used[eid] = true;

            auto [eid2, vid0, vid1, weight] = edges[eid];
            assert(eid2 == eid);
            auto& v0 = vertices[vid0];
            auto& v1 = vertices[vid1];
            
            // neighbors の更新
            v0.neighbors.emplace_back(eid, vid0, vid1, weight);
            v1.neighbors.emplace_back(eid, vid1, vid0, weight);

            using E = std::pair<double, int>; // (w, u)
            using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;

            PQ pq;

            //assert(v0.distance_from_root != inf || v1.distance_from_root != inf);
            if (v0.distance_from_root == inf && v1.distance_from_root == inf) {
                validate();
                return;
            }
            if (v0.distance_from_root + weight < v1.distance_from_root) {
                // v1 の仮の親を v0 として、キューに追加
                //assert(v1.parent.eid != -1);
                if (v1.parent.eid != -1) {
                    vertices[v1.parent.t].remove_child(vid1);
                }
                v1.distance_from_root = v0.distance_from_root + weight;
                v1.parent = OrderedEdge(eid, vid1, vid0, weight);
                v0.children.emplace_back(eid, vid0, vid1, weight);
                pq.emplace(v1.distance_from_root, vid1);
            }
            else if (v1.distance_from_root + weight < v0.distance_from_root) {
                //assert(v0.parent.eid != -1);
                if (v0.parent.eid != -1) {
                    vertices[v0.parent.t].remove_child(vid0);
                }
                v0.distance_from_root = v1.distance_from_root + weight;
                v0.parent = OrderedEdge(eid, vid0, vid1, weight);
                v1.children.emplace_back(eid, vid1, vid0, weight);
                pq.emplace(v0.distance_from_root, vid0);
            }
            else {
                validate();
                return;
            }

            // dijkstra と同様の操作を行う
            while (!pq.empty()) {
                auto [cost, uid] = pq.top(); pq.pop();
                auto& u = vertices[uid];
                if (cost < u.distance_from_root) continue;
                for (const auto& [eid, uid2, vid, weight] : u.neighbors) {
                    assert(uid == uid2);
                    auto& v = vertices[vid];
                    if (chmin(v.distance_from_root, cost + weight)) {
                        //assert(v.parent.eid != -1);
                        if (v.parent.eid != -1) {
                            vertices[v.parent.t].remove_child(vid);
                        }
                        v.parent = OrderedEdge(eid, vid, uid, weight);
                        u.children.emplace_back(eid, uid, vid, weight);
                        pq.emplace(v.distance_from_root, vid);
                    }
                }
            }

            validate();
        }

        ShortestPathTree(const Input& input, int root_index_, bool unary = false) : root_index(root_index_) {

            for (int eid = 0; eid < input.M; eid++) {
                const auto& [v0, v1] = input.uvs[eid];
                const auto& [v0x, v0y] = input.xys[v0];
                const auto& [v1x, v1y] = input.xys[v1];
                double weight = unary ? 1.0 : sqrt((v0x - v1x) * (v0x - v1x) + (v0y - v1y) * (v0y - v1y));
                edges.emplace_back(eid, v0, v1, weight);
            }

            used.resize(input.M, true);

            for (int vid = 0; vid < (int)input.xys.size(); vid++) {
                Vertex v;
                v.vid = vid;
                v.distance_from_root = inf;
                vertices.push_back(v);
            }
            for (const auto& [eid, v0, v1, weight] : edges) {
                vertices[v0].neighbors.emplace_back(eid, v0, v1, weight);
                vertices[v1].neighbors.emplace_back(eid, v1, v0, weight);
            }

            using E = std::pair<double, int>; // (w, u)
            using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;

            PQ pq;
            vertices[root_index].distance_from_root = 0.0;
            pq.emplace(0.0, root_index);
            while (!pq.empty()) {
                auto [cost, uid] = pq.top(); pq.pop();
                const auto& u = vertices[uid];
                if (cost < u.distance_from_root) continue;
                for (const auto& [eid, uid2, vid, weight] : u.neighbors) {
                    assert(uid == uid2);
                    auto& v = vertices[vid];
                    if (chmin(v.distance_from_root, cost + weight)) {
                        v.parent = OrderedEdge(eid, vid, uid, weight);
                        pq.emplace(v.distance_from_root, vid);
                    }
                }
            }

            for (int vid = 0; vid < (int)input.xys.size(); vid++) {
                auto& v = vertices[vid];
                if (v.parent.eid != -1) {
                    int uid = v.parent.t;
                    auto& u = vertices[uid];
                    u.children.emplace_back(v.parent.eid, uid, vid, v.parent.weight);
                }
            }
        }

        void dfs(std::vector<int>& removed, int uid) {
            auto& u = vertices[uid];
            for (const auto& child : u.children) {
                dfs(removed, child.t);
            }
            u.distance_from_root = inf;
            u.children.clear();
            u.parent = { -1, -1, -1, -1.0 };
            removed.push_back(uid);
        }

        void remove(int eid) {
            assert(used[eid]);

            used[eid] = false;
            int subroot_index = -1;
            {
                int uid = edges[eid].v0, vid = edges[eid].v1;
                if (vertices[uid].distance_from_root < vertices[vid].distance_from_root) {
                    subroot_index = vid;
                }
                else {
                    subroot_index = uid;
                }
                int u_erased = vertices[uid].remove(eid);
                int v_erased = vertices[vid].remove(eid);
                assert(u_erased == v_erased);
                // u_erased == v_erased == 1 の場合、最短路に辺 eid は含まれない
                if (u_erased == 1) {
                    validate();
                    return;
                }
            }
            
            using E = std::pair<double, int>; // (w, u)
            using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;

            PQ pq;

            // 部分木を削除する
            std::vector<int> removed;
            dfs(removed, subroot_index);

            // 削除した頂点に隣接していた頂点をキューに入れる
            for (int uid : removed) {
                auto& u = vertices[uid];
                for (const auto& neighbor : u.neighbors) {
                    assert(uid == neighbor.v0);
                    auto& v = vertices[neighbor.v1];
                    if (v.distance_from_root != inf) { // included
                        pq.emplace(v.distance_from_root, v.vid);
                    }
                }
            }

            // dijkstra と同様の操作を行う
            while (!pq.empty()) {
                auto [cost, uid] = pq.top(); pq.pop();
                const auto& u = vertices[uid];
                if (cost < u.distance_from_root) continue;
                for (const auto& [eid, uid2, vid, weight] : u.neighbors) {
                    assert(uid == uid2);
                    auto& v = vertices[vid];
                    if (chmin(v.distance_from_root, cost + weight)) {
                        v.parent = OrderedEdge(eid, vid, uid, weight);
                        pq.emplace(v.distance_from_root, vid);
                    }
                }
            }

            // children のアップデートを行う
            for (int uid : removed) {
                auto& u = vertices[uid];
                if (u.parent.eid != -1) {
                    auto& p = vertices[u.parent.t];
                    p.children.emplace_back(u.parent.eid, p.vid, u.vid, u.parent.weight);
                }
            }

            validate();
        }

        void validate() {
            return;
            std::vector<std::vector<std::tuple<double, int, int>>> adj(N);
            for (int eid = 0; eid < (int)edges.size(); eid++) {
                if (!used[eid]) continue;
                const auto& [_, u, v, w] = edges[eid];
                adj[u].emplace_back(w, v, eid);
                adj[v].emplace_back(w, u, eid);
            }

            using E = std::pair<double, int>;
            using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;
            PQ pq;
            auto dist = make_vector(inf, N);
            //auto prev_v = make_vector(-1, N);
            //auto prev_e = make_vector(-1, N);
            dist[root_index] = 0;
            pq.emplace(0, root_index);
            while (!pq.empty()) {
                auto [cost, u] = pq.top(); pq.pop();
                if (dist[u] < cost) continue;
                for (const auto& [edge_len, v, e] : adj[u]) {
                    if (chmin(dist[v], dist[u] + edge_len)) {
                        //prev_v[v] = u;
                        //prev_e[v] = e;
                        pq.emplace(dist[v], v);
                    }
                }
            }
            //auto fwd_vs = make_vector(std::vector<int>(), N);
            //for (int u = 0; u < N; u++) {
            //    if (prev_v[u] != -1) {
            //        fwd_vs[prev_v[u]].push_back(u);
            //    }
            //}
            //for (int u = 0; u < N; u++) {
            //    std::sort(fwd_vs[u].begin(), fwd_vs[u].end());
            //}

            for (int u = 0; u < N; u++) {
                assert(vertices[u].distance_from_root <= inf);
                assert(abs(dist[u] - vertices[u].distance_from_root) < 1e-8);
                //assert(prev_v[u] == vertices[u].parent.t);
                //assert(prev_e[u] == vertices[u].parent.eid);
                //std::vector<int> fwd;
                //for (const auto& child : vertices[u].children) {
                //    fwd.push_back(child.t);
                //}
                //std::sort(fwd.begin(), fwd.end());
                //if (fwd_vs[u] != fwd) {
                //    dump(u, fwd_vs[u], fwd, std::count(used.begin(), used.end(), true));
                //}
                //assert(fwd_vs[u] == fwd);
            }
        }

        std::vector<double> get_dist() const {
            std::vector<double> dist;
            for (int i = 0; i < N; i++) dist.push_back(vertices[i].distance_from_root);
            return dist;
        }

    };

    void compare(const std::vector<std::vector<double>>& d1, const std::vector<std::vector<double>>& d2) {
        const int N = (int)d1.size();
        for (const auto& v : d1) assert((int)v.size() == N);
        for (const auto& v : d2) assert((int)v.size() == N);
        for (int u = 0; u < N; u++) {
            for (int v = 0; v < N; v++) {
                assert(abs(d1[u][v] - d2[u][v]) < 1e-8);
            }
        }
    }

    void test_wf_djk(const Input& input) {

        auto G = make_vector(1e9, N, N);
        std::vector<std::vector<std::pair<double, int>>> adj(N);
        for (int u = 0; u < N; u++) {
            G[u][u] = 0.0;
        }

        for (int idx = 0; idx < (int)input.uvs.size(); idx++) {
            const auto& [u, v] = input.uvs[idx];
            const auto& [ux, uy] = input.xys[u];
            const auto& [vx, vy] = input.xys[v];
            double d = sqrt((ux - vx) * (ux - vx) + (uy - vy) * (uy - vy));
            G[u][v] = G[v][u] = d;
            adj[u].emplace_back(d, v);
            adj[v].emplace_back(d, u);
        }

        auto d1 = NGraph::compute_apsp_wf(G);
        auto d2 = NGraph::compute_apsp_djk(adj);
        NGraph::compare(d1, d2);

    }

    void test_add_edges(const Input& input) {

        auto G = make_vector(1e9, N, N);
        std::vector<std::vector<std::pair<double, int>>> adj(N);
        for (int u = 0; u < N; u++) {
            G[u][u] = 0.0;
        }

        Xorshift rnd;
        std::vector<int> remove_idxs(input.M);
        std::iota(remove_idxs.begin(), remove_idxs.end(), 0);
        shuffle_vector(remove_idxs, rnd);
        remove_idxs.erase(remove_idxs.begin() + 100, remove_idxs.end());
        assert(remove_idxs.size() == 100);

        for (int idx = 0; idx < (int)input.uvs.size(); idx++) {
            if (std::count(remove_idxs.begin(), remove_idxs.end(), idx)) continue;
            const auto& [u, v] = input.uvs[idx];
            const auto& [ux, uy] = input.xys[u];
            const auto& [vx, vy] = input.xys[v];
            double d = sqrt((ux - vx) * (ux - vx) + (uy - vy) * (uy - vy));
            G[u][v] = G[v][u] = d;
            adj[u].emplace_back(d, v);
            adj[v].emplace_back(d, u);
        }

        auto d1 = NGraph::compute_apsp_wf(G);
        auto d2 = NGraph::compute_apsp_djk(adj);
        NGraph::compare(d1, d2);

        for (int remove_idx : remove_idxs) {
            const auto& [u, v] = input.uvs[remove_idx];
            const auto& [ux, uy] = input.xys[u];
            const auto& [vx, vy] = input.xys[v];
            double d = sqrt((ux - vx) * (ux - vx) + (uy - vy) * (uy - vy));
            NGraph::add_edge(d2, adj, u, v, d);

            adj[u].emplace_back(d, v);
            adj[v].emplace_back(d, u);
            auto d3 = NGraph::compute_apsp_djk(adj);
            NGraph::compare(d2, d3);
        }
    }

    void test_shortest_path_naive(const Input& input) {
        Xorshift rnd;
        std::vector<int> eids(input.M);
        std::iota(eids.begin(), eids.end(), 0);
#pragma omp parallel for
        for (int s = 0; s < N; s++) {
#pragma omp critical(crit_sct)
            {
                dump(s);
                //shuffle_vector(eids, rnd);
            }
            NGraph::ShortestPathTree spt(input, s);
            for (int eid : eids) {
                spt.remove(eid);
                //spt.add(eid);
            }
            for (int eid : eids) {
                spt.add(eid);
            }
        }
    }

    template<typename T, typename U>
    bool near_equal(const std::vector<T>& d1, const std::vector<U>& d2) {
        if (d1.size() != d2.size()) return false;
        bool ok = true;
        for (int i = 0; i < (int)d1.size(); i++) {
            if (abs((double)d1[i] - (double)d2[i]) > 1e-8) {
                return false;
            }
        }
        return true;
    }

    void test_shortest_path(const Input& input, bool unary = false) {
        Xorshift rnd;
        std::vector<int> eids(input.M);
        std::iota(eids.begin(), eids.end(), 0);
        for (int s = 0; s < N; s++) {
            dump(s);
            NGraph::ShortestPathTree spt(input, s);
            NGraph::ShortestPathTree2 spt2(input, s);
            for (int eid : eids) {
                spt.remove(eid);
                spt2.remove(eid);
                assert(near_equal(spt.get_dist(), spt2.dist));
            }
            for (int eid : eids) {
                spt.add(eid);
                spt2.add(eid);
                assert(near_equal(spt.get_dist(), spt2.dist));
            }
        }
    }

    void test_bfs(const Input& input) {
        Xorshift rnd;
        std::vector<int> eids(input.M);
        std::iota(eids.begin(), eids.end(), 0);
        for (int s = 0; s < N; s++) {
            dump(s);
            NGraph::ShortestPathTree spt(input, s, true);
            NGraph::BFSTree bt(input, s);
            for (int eid : eids) {
                spt.remove(eid);
                bt.remove(eid);
                assert(near_equal(bt.dist, spt.get_dist()));
            }
            for (int eid : eids) {
                spt.add(eid);
                bt.add(eid);
                assert(near_equal(bt.dist, spt.get_dist()));
            }
        }
    }

    void check_runtime_shortest_path_1(const Input& input) {
        Perf perf(__FUNCTION__);
        Xorshift rnd;
        std::vector<int> eids(input.M);
        std::iota(eids.begin(), eids.end(), 0);
        for (int i = 0; i < 3; i++) {
            for (int s = 0; s < N; s++) {
                NGraph::ShortestPathTree spt(input, s);
                for (int eid : eids) {
                    spt.remove(eid);
                }
                for (int eid : eids) {
                    spt.add(eid);
                }
            }
        }
    }

    void check_runtime_shortest_path_2(const Input& input) {
        Perf perf(__FUNCTION__);
        Xorshift rnd;
        std::vector<int> eids(input.M);
        std::iota(eids.begin(), eids.end(), 0);
        for (int i = 0; i < 3; i++) {
            for (int s = 0; s < N; s++) {
                NGraph::ShortestPathTree2 spt2(input, s);
                for (int eid : eids) {
                    spt2.remove(eid);
                }
                for (int eid : eids) {
                    spt2.add(eid);
                }
            }
        }
    }

    void check_runtime_bfs(const Input& input) {
        Perf perf(__FUNCTION__);
        Xorshift rnd;
        std::vector<int> eids(input.M);
        std::iota(eids.begin(), eids.end(), 0);
        for (int i = 0; i < 3; i++) {
            for (int s = 0; s < N; s++) {
                NGraph::BFSTree bt(input, s);
                for (int eid : eids) {
                    bt.remove(eid);
                }
                for (int eid : eids) {
                    bt.add(eid);
                }
            }
        }
    }

}


int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

#if 0
    batch_execute();
    exit(1);
#endif

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#if _MSC_VER
    std::ifstream ifs("../../tools/in/0000.txt");
    std::istream& in = ifs;
    std::ofstream ofs("../../tools/out/0000.txt");
    std::ostream& out = ofs;
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
#endif

    const auto input = Input::load(in);
    dump(input.M, input.LB);
    
    {
        //NGraph::test_wf_djk(input);
        //NGraph::test_add_edges(input);
        //NGraph::test_shortest_path_naive(input);
        //NGraph::test_shortest_path(input, false);
        //NGraph::check_runtime_shortest_path_1(input);
        //NGraph::check_runtime_shortest_path_2(input);
        //NGraph::test_bfs(input);
        NGraph::check_runtime_bfs(input);

        exit(0);
    }

    std::vector<int> critical_nodes(input.ts.begin(), input.ts.end());
    critical_nodes.push_back(0);
    std::sort(critical_nodes.begin(), critical_nodes.end());
    critical_nodes.erase(std::unique(critical_nodes.begin(), critical_nodes.end()), critical_nodes.end());

    std::vector<std::vector<int>> G(N);
    for (const auto& [u, v] : input.uvs) {
        G[u].push_back(v);
        G[v].push_back(u);
    }

    //for (int u = 0; u < N; u++) {
    //    auto RG = remove_node(G, u);
    //    auto paths = compute_shortest_paths(RG);
    //    dump(u, compute_average_distance(critical_nodes, paths));
    //}

    auto paths = compute_shortest_paths(G);
    dump(compute_average_distance(critical_nodes, paths));

    auto edges2 = compute_critical_edges(input, paths, true);
    auto edges3 = compute_critical_edges(input, paths, false);

    std::vector<std::vector<int>> G2(N), G3(N);
    for (const auto& [u, v] : edges2) {
        G2[u].emplace_back(v);
        G2[v].emplace_back(u);
    }
    for (const auto& [u, v] : edges3) {
        G3[u].emplace_back(v);
        G3[v].emplace_back(u);
    }

    auto paths2 = compute_shortest_paths(G2);
    auto paths3 = compute_shortest_paths(G3);

    dump(compute_average_distance(critical_nodes, paths2));
    dump(compute_average_distance(critical_nodes, paths3));

    auto initial_A = compute_initial_A(input, G);
    auto initial_A2 = compute_initial_A(input, G2);
    auto initial_A3 = compute_initial_A(input, G3);
    dump(initial_A.size(), initial_A2.size(), initial_A3.size(), input.LA);

    auto tour = compute_tour(input, paths);
    auto tour2 = compute_tour(input, paths2);
    auto tour3 = compute_tour(input, paths3);
    dump(tour.size(), tour2.size(), tour3.size());

    int best_score = INT_MAX;
    std::vector<int> best_A;
    std::vector<std::string> best_ans;

    {
        auto [score, A, ans] = solve(input, tour, initial_A);
        if (chmin(best_score, score)) {
            best_A = A;
            best_ans = ans;
            dump(best_score);
        }
    }
    {
        auto [score, A, ans] = solve(input, tour2, initial_A2);
        if (chmin(best_score, score)) {
            best_A = A;
            best_ans = ans;
            dump(best_score);
        }
    }
    {
        auto [score, A, ans] = solve(input, tour3, initial_A3);
        if (chmin(best_score, score)) {
            best_A = A;
            best_ans = ans;
            dump(best_score);
        }
    }

    const auto& A = best_A;
    const auto& ans = best_ans;

    out << A[0];
    for (int i = 1; i < (int)A.size(); i++) out << ' ' << A[i];
    out << '\n';
    for (const auto& s : ans) out << s << '\n';

    dump(timer.elapsed_ms());

    return 0;
}