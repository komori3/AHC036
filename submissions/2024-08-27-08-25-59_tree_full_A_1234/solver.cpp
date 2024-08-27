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


namespace NGraph {

    constexpr int inf = 1000000000;

    struct Edge {
        int u, v;
        Edge(int u_ = -1, int v_ = -1) : u(u_), v(v_) {}
    };

    struct Neighbor {
        int v, e;
        Neighbor(int v_ = -1, int e_ = -1) : v(v_), e(e_) {}
    };

    struct Graph;
    using GraphPtr = std::shared_ptr<Graph>;
    struct Graph {
        std::vector<Edge> edges; // (u, v)
        std::vector<std::vector<Neighbor>> neighbors; // (v, e)
        Graph(const std::vector<Edge>& edges_, const std::vector<std::vector<Neighbor>>& neighbors_)
            : edges(edges_), neighbors(neighbors_) {}
    };

    GraphPtr create_graph(const Input& input) {
        std::vector<Edge> edges;
        std::vector<std::vector<Neighbor>> neighbors;
        neighbors.resize(input.xys.size());
        for (int e = 0; e < input.M; e++) {
            const auto& [u, v] = input.uvs[e];
            edges.emplace_back(u, v);
            neighbors[u].emplace_back(v, e);
            neighbors[v].emplace_back(u, e);
        }
        return std::make_shared<Graph>(edges, neighbors);
    }

    std::vector<Edge> create_edges(const Input& input) {
        std::vector<Edge> edges;
        for (int e = 0; e < input.M; e++) {
            const auto& [u, v] = input.uvs[e];
            edges.emplace_back(u, v);
        }
        return edges;
    }

    std::vector<std::vector<Neighbor>> create_neighbors(const Input& input) {
        std::vector<std::vector<Neighbor>> neighbors;
        neighbors.resize(input.xys.size());
        for (int e = 0; e < input.M; e++) {
            const auto& [u, v] = input.uvs[e];
            neighbors[u].emplace_back(v, e);
            neighbors[v].emplace_back(u, e);
        }
        return neighbors;
    }

    struct BFSTree {

        int root;
        GraphPtr g;

        std::bitset<3 * N> used;
        std::array<int, N> dist;
        std::array<int, N> vpar; // 実は不要説
        std::array<int, N> efrom;
        uint64_t total_dist;

        std::vector<int> removed;
        FastQueue qu;

        BFSTree() {}
        BFSTree(int root_, GraphPtr g_) : root(root_), g(g_) {
            used.set();
            dist.fill(inf);
            vpar.fill(-1);
            efrom.fill(-1);

            dist[root] = 0;
            qu.reset();
            qu.push(root);
            while (!qu.empty()) {
                int from = qu.pop();
                for (const auto& [to, e] : g->neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (chmin(dist[to], dist[from] + 1)) {
                        vpar[to] = from;
                        efrom[to] = e;
                        qu.push(to);
                    }
                }
            }
            total_dist = std::accumulate(dist.begin(), dist.end(), int64_t(0));
        }

        inline void change_dist(int u, int d) {
            total_dist += d - dist[u];
            dist[u] = d;
        }

        void dfs(std::vector<int>& removed, int u) {
            for (const auto& [v, e] : g->neighbors[u]) {
                if (!used[e]) continue;
                if (vpar[v] == u) { // is child node
                    dfs(removed, v);
                }
            }
            vpar[u] = efrom[u] = -1;
            change_dist(u, inf);
            removed.push_back(u);
        }

        void remove(int e) {
            assert(used[e]);
            used[e] = false;
            int subroot = -1;
            {
                const auto& [u, v] = g->edges[e];
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
                for (const auto& [v, e] : g->neighbors[u]) {
                    if (dist[v] != inf) {
                        qu.push(v);
                    }
                }
            }

            // bfs
            while (!qu.empty()) {
                int from = qu.pop();
                for (const auto& [to, e] : g->neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (dist[from] + 1 < dist[to]) {
                        change_dist(to, dist[from] + 1);
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
            const auto& [v0, v1] = g->edges[e];

            if (dist[v0] == inf && dist[v1] == inf) return;

            qu.reset();
            if (dist[v0] + 1 < dist[v1]) {
                change_dist(v1, dist[v0] + 1);
                vpar[v1] = v0;
                efrom[v1] = e;
                qu.push(v1);
            }
            else if (dist[v1] + 1 < dist[v0]) {
                change_dist(v0, dist[v1] + 1);
                vpar[v0] = v1;
                efrom[v0] = e;
                qu.push(v0);
            }
            else return;

            // bfs
            while (!qu.empty()) {
                int from = qu.pop();
                for (const auto& [to, e] : g->neighbors[from]) {
                    if (!used[e]) continue; // 削除したほうがいい？
                    if (dist[from] + 1 < dist[to]) {
                        change_dist(to, dist[from] + 1);
                        vpar[to] = from;
                        efrom[to] = e;
                        qu.push(to);
                    }
                }
            }
        }

        std::vector<std::vector<int>> to_graph() const {
            std::vector<std::vector<int>> G(N);
            for (int u = 0; u < N; u++) {
                if (efrom[u] != -1) {
                    const auto& [x, y] = g->edges[efrom[u]];
                    G[x].push_back(y);
                    G[y].push_back(x);
                }
            }
            return G;
        }
        
        std::vector<std::vector<int>> to_graph(const std::vector<int>& critical_nodes) const {
            std::bitset<N> critical;
            for (int u : critical_nodes) critical[u] = true;
            std::vector<int> indeg(N);
            for (int u = 0; u < N; u++) {
                if (vpar[u] != -1) {
                    indeg[vpar[u]]++;
                }
            }
            std::vector<int> stk;
            for (int u = 0; u < N; u++) {
                if (!indeg[u]) {
                    stk.push_back(u);
                }
            }
            std::bitset<N> removable;
            while (!stk.empty()) {
                int u = stk.back(); stk.pop_back();
                if (!critical[u]) {
                    removable[u] = true;
                    if (vpar[u] != -1) {
                        indeg[vpar[u]]--;
                        if (!indeg[vpar[u]]) {
                            stk.push_back(vpar[u]);
                        }
                    }
                }
            }
            std::vector<std::vector<int>> G(N);
            for (int u = 0; u < N; u++) {
                if (efrom[u] == -1) continue;
                const auto& [x, y] = g->edges[efrom[u]];
                if (removable[x] && removable[y]) continue;
                G[x].push_back(y);
                G[y].push_back(x);
            }
            //dump(indeg);
            return G;
        }

    };

    struct BFSTrees {

        GraphPtr g;
        std::array<BFSTree, N> trees;
        std::bitset<3 * N> used;
        int64_t entire_dist;

        BFSTrees(const Input& input) {
            g = create_graph(input);
            for (int u = 0; u < N; u++) {
                trees[u] = BFSTree(u, g);
            }
            used.set();
            entire_dist = 0;
            for (const auto& tree : trees) {
                entire_dist += tree.total_dist;
            }
        }

        int64_t add(int e) {
            assert(!used[e]);
            used[e] = true;
            auto prev_entire_dist = entire_dist;
            for (auto& tree : trees) {
                entire_dist -= tree.total_dist;
                tree.add(e);
                entire_dist += tree.total_dist;
            }
            return entire_dist - prev_entire_dist;
        }

        int64_t remove(int e) {
            assert(used[e]);
            used[e] = false;
            auto prev_entire_dist = entire_dist;
            for (auto& tree : trees) {
                entire_dist -= tree.total_dist;
                tree.remove(e);
                entire_dist += tree.total_dist;
            }
            return entire_dist - prev_entire_dist;
        }

    };

}

namespace NTree {

    using Graph = std::vector<std::vector<int>>;
    std::pair<int, int> dfs(const Graph& G, int u, int par) {  // 最遠点間距離と最遠点を求める
        std::pair<int, int> ret = std::make_pair(0, u);
        for (auto e : G[u]) {
            if (e == par) continue;
            auto next = dfs(G, e, u);
            next.first += 1;
            ret = std::max(ret, next);
        }
        return ret;
    }
    int tree_diameter(const Graph& G) {
        std::pair<int, int> p = dfs(G, 0, -1);
        std::pair<int, int> q = dfs(G, p.second, -1);
        return q.first;
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

    std::vector<int> critical_nodes(input.ts.begin(), input.ts.end());
    critical_nodes.push_back(0);
    std::sort(critical_nodes.begin(), critical_nodes.end());
    critical_nodes.erase(std::unique(critical_nodes.begin(), critical_nodes.end()), critical_nodes.end());

    std::vector<std::vector<int>> G(N);
    for (const auto& [u, v] : input.uvs) {
        G[u].push_back(v);
        G[v].push_back(u);
    }

    auto paths = compute_shortest_paths(G);

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

    std::vector<std::vector<int>> G4;
    {
        int min_diam = INT_MAX, min_root = -1;
        NGraph::BFSTrees trees(input);
        for (const auto& tree : trees.trees) {
            //auto g = tree.to_graph(critical_nodes);
            auto g = tree.to_graph();
            //exit(1);
            int diam = NTree::tree_diameter(g);
            if (chmin(min_diam, diam)) {
                G4 = g;
                min_root = tree.root;
            }
        }
        dump(min_diam);
    }

    auto paths2 = compute_shortest_paths(G2);
    auto paths3 = compute_shortest_paths(G3);
    auto paths4 = compute_shortest_paths(G4);

    auto initial_A = compute_initial_A(input, G);
    auto initial_A2 = compute_initial_A(input, G2);
    auto initial_A3 = compute_initial_A(input, G3);
    auto initial_A4 = compute_initial_A(input, G4);
    dump(initial_A.size(), initial_A2.size(), initial_A3.size(), initial_A4.size(), input.LA);

    auto tour = compute_tour(input, paths);
    auto tour2 = compute_tour(input, paths2);
    auto tour3 = compute_tour(input, paths3);
    auto tour4 = compute_tour(input, paths4);
    dump(tour.size(), tour2.size(), tour3.size(), tour4.size());

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
    {
        auto [score, A, ans] = solve(input, tour4, initial_A4);
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