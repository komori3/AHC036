#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#include <optional>
//#include <format>
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
} g_timer;
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

struct Perf {
    Timer t;
    const char* const func;
    Perf(const char* func_) : func(func_) {}
    ~Perf() {
        //std::format_to(std::ostream_iterator<char>(std::cerr), "[{:>50}] {:7.2f} / {:7.2f} [ms]\n", func, t.elapsed_ms(), g_timer.elapsed_ms());
        fprintf(stderr, "%s: %4.2f / %4.2f [ms]\n", func, t.elapsed_ms(), g_timer.elapsed_ms());
    }
};

template <typename T = int>
T ipow(T x, T n) {
    T ret = 1;
    for (T i = 0; i < n; i++) ret *= x;
    return ret;
}



constexpr int N = 600;

constexpr int M_MIN = N - 1;
constexpr int M_MAX = 3 * N - 6;
constexpr int LA_MIN = N;
constexpr int LA_MAX = 2 * N;
constexpr int LB_MIN = 4;
constexpr int LB_MAX = 24;

constexpr int inf = 1000000000;


struct Option {
    int seed;

    std::string stringify() const {
        return format("seed=%d", seed);
    }
};

namespace NInput {

    int M;
    int LA;
    int LB;
    std::vector<std::pair<int, int>> uvs;
    std::vector<int> ts;
    std::vector<std::pair<int, int>> xys;

    std::vector<int> tour_nodes;
    std::bitset<N> is_tour_node;

    void load(std::istream& in) {
        in >> M >> M >> LA >> LA >> LB; // ignore N, T
        uvs.resize(M);
        ts.resize(N);
        xys.resize(N);
        in >> uvs >> ts >> xys;
    }

    void initialize(int argc, char** argv, const Option& opt) {
        Perf perf(__FUNCTION__);
        if (argc >= 2 && std::string(argv[1]) == "local") {
            assert(0 <= opt.seed && opt.seed < 100);
            const std::string file(format("../../tools/in/%04d.txt", opt.seed));
            assert(std::filesystem::is_regular_file(file));
            std::ifstream ifs(file);
            load(ifs);
        }
        else {
            load(std::cin);
        }
        std::set<int> tset({ 0 });
        for (int t : ts) {
            tset.insert(t);
        }
        tour_nodes = std::vector<int>(tset.begin(), tset.end());
        is_tour_node.reset();
        for (int t : tour_nodes) is_tour_node[t] = true;
    }

}

namespace NOutput {

    void output(std::ostream& out, const std::vector<int>& A, const std::vector<std::string>& lines) {
        out << A[0];
        for (int i = 1; i < (int)A.size(); i++) out << ' ' << A[i];
        out << '\n';
        for (const auto& s : lines) out << s << '\n';
    }

    void output(int argc, char** argv, const Option& opt, const std::vector<int>& A, const std::vector<std::string>& lines) {
        if (argc >= 2 && std::string(argv[1]) == "local") {
            assert(0 <= opt.seed && opt.seed < 100);
            const std::string file(format("../../tools/out/%04d.txt", opt.seed));
            std::ofstream ofs(file);
            output(ofs, A, lines);
        }
        else {
            output(std::cout, A, lines);
        }
    }

}

namespace NGraph {

    std::array<std::array<int, N>, N> dss;   // distance matrix
    std::array<std::array<int, N>, N> pvss;  // parent
    std::array<std::array<int, N>, N> pess;  // edge from

    std::array<std::vector<std::pair<int, int>>, N> adjs;

    int shortest_tour_length;

    void initialize(int argc, char** argv, const Option& opt) {
        Perf perf(__FUNCTION__);
        using E = std::pair<int, int>; // (vto, eto)
        using Q = std::queue<int>;
        for (int e = 0; e < NInput::M; e++) {
            const auto& [u, v] = NInput::uvs[e];
            adjs[u].emplace_back(v, e);
            adjs[v].emplace_back(u, e);
        }
        for (int s = 0; s < N; s++) {
            auto& ds = dss[s];
            auto& pvs = pvss[s];
            auto& pes = pess[s];
            std::fill(ds.begin(), ds.end(), inf);
            std::fill(pvs.begin(), pvs.end(), -1);
            std::fill(pes.begin(), pes.end(), -1);
            Q qu;
            qu.push(s);
            ds[s] = 0;
            while (!qu.empty()) {
                auto u = qu.front(); qu.pop();
                for (const auto& [v, e] : adjs[u]) {
                    if (ds[v] != inf) continue;
                    ds[v] = ds[u] + 1;
                    pvs[v] = u;
                    pes[v] = e;
                    qu.push(v);
                }
            }
        }
        {
            shortest_tour_length = 0;
            int t = 0;
            for (int nt : NInput::ts) {
                shortest_tour_length += dss[t][nt];
                t = nt;
            }
        }
    }

}

namespace NSteiner {

    int T;                   // num terminal
    std::array<int, N> term; // terminal id -> vertex id
    std::bitset<N> used;     // terminal t is used if used[t]=true

    std::array<std::pair<int, int>, N> min;

    std::bitset<M_MAX> B; // used edges

    int edge_cost;

    void set_terminals(const std::vector<int>& ts) {
        T = (int)ts.size();
        std::memcpy(term.data(), ts.data(), sizeof(int) * T);
    }

    void run(int initial_tid) {

        B.reset();
        edge_cost = 0;

        used[initial_tid] = true;
        for (int t = 0; t < T; t++) {
            min[t] /* T */ = { NGraph::dss[term[initial_tid]][term[t]], term[initial_tid] }; // gds: NxN
        }
        while (true) {
            int u /* N */ = -1, t /* T */ = -1, d = inf;
            for (int k /* T */ = 0; k < T; k++) {
                int i = term[k]; // N
                if (!used[k] && chmin(d, min[k].first)) {
                    u = i, t = k;
                }
            }
            if (u == -1) break;
            int s = min[t].second; /* N */
            while (s != u) {
                for (int k = 0; k < T; k++) {
                    int i = term[k];
                    if (!used[k]) {
                        chmin(min[k], { NGraph::dss[u][i], u });
                    }
                }
                int v = NGraph::pvss[s][u], e = NGraph::pess[s][u];
                if (!B[e]) {
                    B[e] = true;
                    edge_cost++;
                }
                u = v;
            }
            used[t] = true;
        }
    }

    std::vector<std::vector<std::pair<int, int>>> to_graph() {
        std::vector<std::vector<std::pair<int, int>>> g(N);
        for (int e = 0; e < NInput::M; e++) {
            if (B[e]) {
                const auto& [u, v] = NInput::uvs[e];
                g[u].emplace_back(v, e);
                g[v].emplace_back(u, e);
            }
        }
        return g;
    }

}

namespace NLCA {

    using namespace std;

    using Graph = std::vector<std::vector<std::pair<int, int>>>;

    struct LCA {
        using int_type = short;
        static constexpr int K = 10;
        static constexpr int V = N;
        std::array<std::array<int_type, N>, K> parent;
        std::array<int_type, N> depth;
        // 初期化
        void init(const Graph& G, int root = 0) {
            memset(parent.data(), -1, sizeof(int_type) * N * 10);
            memset(depth.data(), -1, sizeof(int_type) * N);
            dfs(G, root, -1, 0);
            for (int k = 0; k + 1 < K; k++) {
                for (int v = 0; v < V; v++) {
                    parent[k + 1][v] = (parent[k][v] < 0) ? -1 : parent[k][parent[k][v]];
                }
            }
        }
        void dfs(const Graph& G, int v, int p, int d) {
            parent[0][v] = p;
            depth[v] = d;
            for (const auto& [e, _] : G[v]) {
                if (e != p) dfs(G, e, v, d + 1);
            }
        }
        int query(int u, int v) {
            if (depth[u] < depth[v]) std::swap(u, v);
            int K = parent.size();
            for (int k = 0; k < K; k++) {
                if ((depth[u] - depth[v]) >> k & 1) {
                    u = parent[k][u];
                }
            }
            if (u == v) return u;
            for (int k = K - 1; k >= 0; k--) {
                if (parent[k][u] != parent[k][v]) {
                    u = parent[k][u];
                    v = parent[k][v];
                }
            }
            return parent[0][u];
        }
        int dist(int u, int v) { return depth[u] + depth[v] - 2 * depth[query(u, v)]; }
    };

    LCA lca;

}

// ナイーブにツアーの長さを計算
int compute_tour_length(const std::vector<std::vector<std::pair<int, int>>>& G) {
    std::array<int, N> ds;
    FastQueue qu;
    int s = 0;
    int len = 0;
    for (int t : NInput::ts) {
        qu.reset();
        ds.fill(inf);
        ds[s] = 0;
        qu.push(s);
        while (!qu.empty()) {
            auto u = qu.pop();
            if (u == t) {
                len += ds[t];
                s = t;
                break;
            }
            for (const auto& [v, e] : G[u]) {
                if (ds[v] != inf) continue;
                ds[v] = ds[u] + 1;
                qu.push(v);
            }
        }
    }
    return len;
}

// 木に対するツアーの長さを計算 LCAで高速化
int compute_tour_length_tree(const std::vector<std::vector<std::pair<int, int>>>& G) {
    NLCA::lca.init(G);
    int s = 0, len = 0;
    for (int t : NInput::ts) {
        len += NLCA::lca.dist(s, t);
        s = t;
    }
    return len;
}

std::vector<int> compute_tour(const std::vector<std::vector<std::pair<int, int>>>& G) {
    std::array<int, N> pv;
    FastQueue qu;
    int s = 0;
    std::vector<int> tour({ s });
    for (int t : NInput::ts) {
        qu.reset();
        pv.fill(inf);
        pv[s] = -1;
        qu.push(s);
        while (!qu.empty()) {
            auto u = qu.pop();
            if (u == t) {
                break;
            }
            for (const auto& [v, e] : G[u]) {
                if (pv[v] != inf) continue;
                pv[v] = u;
                qu.push(v);
            }
        }
        int k = t, n = (int)tour.size();
        while (k != s) {
            tour.push_back(k);
            k = pv[k];
        }
        std::reverse(tour.begin() + n, tour.end());
        s = t;
    }
    return tour;
}

void remove_edge(std::vector<std::vector<std::pair<int, int>>>& G, int e) {
    const auto& [u, v] = NInput::uvs[e];
    G[u].erase(
        std::remove_if(G[u].begin(), G[u].end(), [&v](const std::pair<int, int>& p) {
            return p.first == v;
            }),
        G[u].end()
    );
    G[v].erase(
        std::remove_if(G[v].begin(), G[v].end(), [&u](const std::pair<int, int>& p) {
            return p.first == u;
            }),
        G[v].end()
    );
}

struct ModifyTreeResult {
    bool succeed;
    std::vector<int> added_edges;
    std::vector<int> removed_edges;
};

ModifyTreeResult modify_tree(std::vector<std::vector<std::pair<int, int>>>& G, std::bitset<M_MAX>& used_edge, Xorshift& rnd) {
    // ある点 s を選択
    // s から別の頂点へ、使われていない辺を経由してパスを繋ぐ
    // 出来たサイクルの、繋いだパス以外の辺を 1 つ取り除く
    int s = -1;
    while (true) {
        s = rnd.next_u32(N);
        if (G[s].empty()) continue; // 孤立点
        if (G[s].size() == NGraph::adjs[s].size()) continue; // 伸ばせるパスがない
        break;
    }
    int t = -1;
    std::vector<int> pv(N, inf), pe(N, inf);
    std::queue<int> qu;
    qu.push(s);
    pv[s] = pe[s] = -1;
    while (!qu.empty()) {
        int u = qu.front(); qu.pop();
        if (u != s && !G[u].empty()) {
            t = u;
            break;
        }
        for (const auto& [v, e] : NGraph::adjs[u]) {
            if (pv[v] != inf || used_edge[e]) continue;
            pv[v] = u;
            pe[v] = e;
            qu.push(v);
        }
    }
    if (t == -1) {
        return { false, {}, {} };
    }
    std::vector<int> added_edges;
    {
        int k = t;
        while (k != s) {
            added_edges.push_back(pe[k]);
            k = pv[k];
        }
    }
    // s->t のパスを求める
    pv.assign(N, inf);
    pe.assign(N, inf);
    pv[s] = pe[s] = -1;
    while (!qu.empty()) qu.pop();
    qu.push(s);
    while (!qu.empty()) {
        int u = qu.front(); qu.pop();
        if (u == t) break;
        for (const auto& [v, e] : G[u]) {
            if (pv[v] != inf) continue;
            pv[v] = u;
            pe[v] = e;
            qu.push(v);
        }
    }
    for (int e : added_edges) { // add
        const auto& [u, v] = NInput::uvs[e];
        G[u].emplace_back(v, e);
        G[v].emplace_back(u, e);
        used_edge[e] = true;
    }
    std::vector<int> remove_candidate_edges;
    {
        int k = t;
        while (k != s) {
            remove_candidate_edges.push_back(pe[k]);
            k = pv[k];
        }
    }
    std::vector<int> removed_edges;
    int removed_edge = remove_candidate_edges[rnd.next_u32(remove_candidate_edges.size())];
    remove_edge(G, removed_edge);
    used_edge[removed_edge] = false;
    removed_edges.push_back(removed_edge);
    if (true) {
        auto [v0, v1] = NInput::uvs[removed_edge];
        for (int x : {v0, v1}) {
            // terminal でない葉に接続する辺を破壊する
            while (G[x].size() == 1 && !NInput::is_tour_node[x]) {
                auto [nx, e] = G[x].front();
                remove_edge(G, e);
                used_edge[e] = false;
                removed_edges.push_back(e);
                x = nx;
            }
        }
    }

    return { true, added_edges, removed_edges };
}

void undo_modify_tree(std::vector<std::vector<std::pair<int, int>>>& G, std::bitset<M_MAX>& used_edge, const ModifyTreeResult& mtr) {
    for (int e : mtr.added_edges) {
        remove_edge(G, e);
        used_edge[e] = false;
    }
    for (int e : mtr.removed_edges) {
        const auto& [u, v] = NInput::uvs[e];
        G[u].emplace_back(v, e);
        G[v].emplace_back(u, e);
        used_edge[e] = true;
    }
}

void show(const std::bitset<M_MAX>& E, int delay = 0) {
#ifdef HAVE_OPENCV_HIGHGUI
    std::vector<bool> to_visit(N);
    to_visit[0] = true;
    for (int t : NInput::ts) to_visit[t] = true;
    constexpr int margin = 50;
    constexpr int width = 1000;
    std::vector<cv::Point> points;
    for (const auto& [x, y] : NInput::xys) {
        points.emplace_back(x + margin, y + margin);
    }
    cv::Mat3b img(width + margin * 2, width + margin * 2, cv::Vec3b(255, 255, 255));
    for (int n = 0; n < N; n++) {
        const auto& p = points[n];
        if (to_visit[n]) {
            cv::circle(img, p, 3, cv::Scalar(0, 0, 255), cv::FILLED);
        }
        else {
            cv::circle(img, p, 2, cv::Scalar(0, 0, 255), cv::FILLED);
        }
    }
    for (const auto& p : points) {
        cv::circle(img, p, 2, cv::Scalar(0, 0, 255), cv::FILLED);
    }
    for (const auto& [u, v] : NInput::uvs) {
        cv::line(img, points[u], points[v], cv::Scalar(224, 224, 224));
    }
    for (int e = 0; e < NInput::M; e++) {
        if (!E[e]) continue;
        const auto& [u, v] = NInput::uvs[e];
        cv::line(img, points[u], points[v], cv::Scalar(0, 0, 255));
    }
    cv::imshow("img", img);
    cv::waitKey(delay);
#endif
}

// 帰りがけで配列 A を作成
void dfs(const std::vector<std::vector<std::pair<int, int>>>& G, std::vector<bool>& visited, std::vector<int>& path, int p, int u) {
    visited[u] = true;
    //path.push_back(u);
    for (const auto& [v, e] : G[u]) {
        if (v == p) continue;
        if (visited[v]) continue;
        dfs(G, visited, path, u, v);
    }
    path.push_back(u);
}

// 帰りがけで配列 A を作成
void dfs(const std::vector<std::vector<std::pair<int, int>>>& G, std::vector<bool>& visited, std::vector<int>& path, int p, int u, Xorshift& rnd) {
    visited[u] = true;
    //path.push_back(u);
    std::vector<int> r(G[u].size());
    std::iota(r.begin(), r.end(), 0);
    shuffle_vector(r, rnd);
    for (int i = 0; i < G[u].size(); i++) {
        //for (const auto& [v, e] : G[u]) {
        const auto& [v, e] = G[u][r[i]];
        if (v == p) continue;
        if (visited[v]) continue;
        dfs(G, visited, path, u, v);
    }
    path.push_back(u);
}


std::vector<int> compute_initial_A(const std::vector<std::vector<std::pair<int, int>>>& G) {
    std::vector<bool> visited(N);
    std::vector<int> A;
    dfs(G, visited, A, -1, 0);
    assert(A.size() <= NInput::LA);
    return A;
}

std::vector<int> compute_initial_A(const std::vector<std::vector<std::pair<int, int>>>& G, Xorshift& rnd) {
    std::vector<bool> visited(N);
    std::vector<int> A;
    int s = -1;
    do {
        s = rnd.next_u32(N);
    } while (G[s].empty());
    dfs(G, visited, A, -1, s);
    assert(A.size() <= NInput::LA);
    return A;
}

std::pair<int, std::vector<int>> compute_modified_A(
    const std::vector<int>& tour,
    std::vector<int> A,
    const int thresh
) {
    static std::array<int, N> signal{};
    int head = 0, score = 0;
    while (head + 1 < (int)tour.size()) {
        int max_a = -1;
        int max_head = head;
        for (int i = 0; i < NInput::LB; i++) signal[A[i]]++;
        for (int a = 0; a + NInput::LB <= (int)A.size(); a++) {
            int nhead = head;
            while (nhead + 1 < (int)tour.size() && signal[tour[nhead + 1]]) nhead++;
            if (chmax(max_head, nhead)) {
                max_a = a;
            }
            if (a + NInput::LB == (int)A.size()) break;
            signal[A[a]]--;
            signal[A[a + NInput::LB]]++;
        }
        assert(head != max_head);
        for (int i = (int)A.size() - NInput::LB; i < (int)A.size(); i++) signal[A[i]]--;
        if (max_head - head <= thresh && A.size() < NInput::LA) {
            //A.push_back(tour[head + 1]); // これは弱い
            // これが強いので、tour の一部を追加する処理が有力？
            for (int k = head + 1; k < std::min((int)tour.size(), head + 1 + NInput::LB); k++) {
                A.push_back(tour[k]);
                if (A.size() == NInput::LA) break;
            }
            continue;
        }
        score++;
        head = max_head;
    }
    //dump(score);
    //while (A.size() < NInput::LA) A.push_back(0);
    return { score, A };
}

std::pair<int, std::vector<int>> compute_modified_A(
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
        std::tie(score, A) = compute_modified_A(tour, initial_A, thresh);
        if (chmin(best_score, score)) {
            best_A = A;
            best_ans = ans;
            dump(thresh, best_score);
        }
        else {
            break;
        }
    }

    return { best_score, best_A };
}

std::pair<int, std::vector<int>> compute_A(
    const std::vector<int>& tour,
    const std::vector<std::vector<std::pair<int, int>>>& g
) {
    auto initial_A = compute_initial_A(g);
    return compute_modified_A(tour, initial_A);
}

std::pair<int, std::vector<int>> compute_A(
    const std::vector<int>& tour,
    const std::vector<std::vector<std::pair<int, int>>>& g,
    Xorshift& rnd
) {
    auto initial_A = compute_initial_A(g, rnd);
    return compute_modified_A(tour, initial_A);
}

// ケツから貪欲はできなさそう？
std::tuple<int, std::vector<int>, std::vector<std::string>> solve_segment(
    const std::vector<int>& tour,
    const std::vector<int>& A
) {
    Perf perf(__FUNCTION__);

    const int NT = (int)tour.size();
    const int NA = (int)A.size();
    int head = 0;

    std::array<int, N> signal{};
    std::vector<int> B(A.begin(), A.begin() + NInput::LB);
    for (int i = 0; i < NInput::LB; i++) signal[A[i]]++;

    std::vector<std::tuple<int, int, int, int>> path;
    while (head + 1 < NT) {
        int max_nhead = head, max_len = -1, max_a = -1, max_b = -1;
        for (int len = (head == 0) ? NInput::LB : 1; len <= NInput::LB; len++) {

            for (int i = 0; i < len; i++) signal[B[i]]--;

            for (int b = 0; b + len <= NInput::LB; b++) {

                for (int i = 0; i < len; i++) signal[A[i]]++;

                for (int a = 0; a + len <= NA; a++) {

                    if (max_nhead + 1 < NT && signal[tour[max_nhead + 1]]) { // pruning
                        int nhead = head;
                        while (nhead + 1 < NT && signal[tour[nhead + 1]]) nhead++;
                        if (chmax(max_nhead, nhead)) {
                            max_len = len;
                            max_a = a;
                            max_b = b;
                        }
                    }

                    if (a + len < NA) {
                        signal[A[a]]--;
                        signal[A[a + len]]++;
                    }
                }

                for (int i = NA - len; i < NA; i++) signal[A[i]]--;

                if (b + len < NInput::LB) {
                    signal[B[b]]++;
                    signal[B[b + len]]--;
                }
            }

            for (int i = NInput::LB - len; i < NInput::LB; i++) signal[B[i]]++;
        }
        assert(max_nhead != head);
        path.emplace_back(max_nhead, max_len, max_a, max_b);
        for (int i = 0; i < max_len; i++) {
            signal[B[max_b + i]]--;
            signal[A[max_a + i]]++;
            B[max_b + i] = A[max_a + i];
        }
        head = max_nhead;
    }

    std::vector<std::string> ans;
    int score = 0;
    head = 0;
    for (int i = 0; i < (int)path.size(); i++) {
        const auto& [nhead, len, a, b] = path[i];
        ans.push_back(format("s %d %d %d", len, a, b));
        score++;
        for (int t = head + 1; t <= nhead; t++) {
            ans.push_back(format("m %d", tour[t]));
        }
        head = nhead;
    }
    //dump(score);
    return { score, A, ans };
}

int compute_signal_cost(
    const std::vector<int>& tour,
    const std::vector<int>& A
) {
    static std::array<int, N> signal{};
    //Perf perf(__FUNCTION__);
    int tail = (int)tour.size() - 1;
    int cost = 0;
    while (tail > 0) {
        int min_ntail = INT_MAX, min_idx = -1;
        for (int i = 0; i < NInput::LB; i++) signal[A[i]]++;
        for (int a_idx = 0; a_idx + NInput::LB <= (int)A.size(); a_idx++) {
            int ntail = tail;
            while (ntail > 0 && signal[tour[ntail]]) ntail--;
            if (chmin(min_ntail, ntail)) {
                min_idx = a_idx;
            }
            if (a_idx + NInput::LB == (int)A.size()) break;
            signal[A[a_idx]]--;
            signal[A[a_idx + NInput::LB]]++;
        }
        for (int i = (int)A.size() - NInput::LB; i < (int)A.size(); i++) signal[A[i]]--;
        if (min_ntail == tail) return inf;
        tail = min_ntail;
        cost++;
    }
    return cost;
}

int count_used_vertex(const std::vector<std::vector<std::pair<int, int>>>& g) {
    std::bitset<N> used;
    for (int u = 0; u < N; u++) {
        for (const auto& [v, e] : g[u]) {
            used[u] = used[v] = true;
        }
    }
    return used.count();
}

int compute_signal_cost(
    const std::vector<int>& tour,
    const std::vector<int>& A,
    const int pruning_thresh
) {
    static std::array<int, N> signal{};
    //Perf perf(__FUNCTION__);
    int tail = (int)tour.size() - 1;
    int cost = 0;
    while (tail > 0) {
        int min_ntail = INT_MAX, min_idx = -1;
        for (int i = 0; i < NInput::LB; i++) signal[A[i]]++;
        for (int a_idx = 0; a_idx + NInput::LB <= (int)A.size(); a_idx++) {
            int ntail = tail;
            while (ntail > 0 && signal[tour[ntail]]) ntail--;
            if (chmin(min_ntail, ntail)) {
                min_idx = a_idx;
            }
            if (a_idx + NInput::LB == (int)A.size()) break;
            signal[A[a_idx]]--;
            signal[A[a_idx + NInput::LB]]++;
        }
        for (int i = (int)A.size() - NInput::LB; i < (int)A.size(); i++) signal[A[i]]--;
        if (min_ntail == tail) return inf;
        tail = min_ntail;
        cost++;
        if (pruning_thresh < cost) return inf;
    }
    return cost;
}

struct SegmentModifier {

    Timer& timer;
    Xorshift& rnd;
    std::vector<int> tour;
    std::vector<int> A;

    int margin;
    int nsegment;
    std::vector<std::pair<int, int>> segments;
    std::vector<int> tour_lefts;

    int cost;

    void assign_segment(int segment_id, int tour_left) {
        const auto& [a_left, length] = segments[segment_id];
        for (int i = 0; i < length; i++) {
            A[a_left + i] = tour[tour_left + i];
        }
        tour_lefts[segment_id] = tour_left;
    }

    SegmentModifier(
        Timer& timer_,
        Xorshift& rnd_,
        const std::vector<int> tour_,
        const std::vector<int>& A_
    ) : timer(timer_), rnd(rnd_), tour(tour_), A(A_) {
        margin = NInput::LA - (int)A.size();
        nsegment = (margin + NInput::LB - 1) / NInput::LB;
        segments.resize(nsegment);
        for (int i = 0; i < nsegment; i++) {
            int a_left = (int)A.size() + i * NInput::LB;
            int end_pos = std::min(NInput::LA, a_left + NInput::LB);
            int length = end_pos - a_left;
            segments[i] = std::make_pair(a_left, length);
        }
        while (A.size() < NInput::LA) A.push_back(0);
        tour_lefts.resize(nsegment);
        for (int i = 0; i < nsegment; i++) {
            const auto& [a_left, length] = segments[i];
            int tour_left = rnd.next_u32((int)tour.size() - length);
            assign_segment(i, tour_left);
        }
        cost = compute_signal_cost(tour, A);
    }

    void run(double duration) {
        const double end_time = timer.elapsed_ms() + duration;
        int loop = 0;
        while (timer.elapsed_ms() < end_time) {
            int segment_id = rnd.next_u32(nsegment);
            const auto& [a_left, length] = segments[segment_id];
            int ptour_left = tour_lefts[segment_id];
            int tour_left = rnd.next_u32((int)tour.size() - length);
            assign_segment(segment_id, tour_left);
            int ncost = compute_signal_cost(tour, A, cost);
            if (cost < ncost) {
                assign_segment(segment_id, ptour_left);
            }
            else {
                cost = ncost;
                dump(cost);
            }
            loop++;
        }
        dump(loop, cost);
    }

};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

    //test_compute_signal_cost();
    //exit(1);

    // 使用頂点数が少ないほど配列 A の余剰スペースが増える (-> 一回の信号変化で進める距離が増える)
    // ツアーが短いほど移動回数が減る（-> 信号変化の回数も減る）
    // 両者はトレードオフの関係にあるので、いい塩梅を探す必要がある

    // 使用頂点を少なくしつつ、往復時に 2*LB 回程度の移動を稼げるので、木構造に近いグラフが良い気がしている

    // シュタイナー木を一本生成する
    // 辺のカット -> 連結 (or 連結 -> カット) 遷移を行う
    // ツアーの長さと使用頂点数を評価する

    // 木にエッジを幾つか追加することでコストが 10% くらい減りそう
    // 闇雲に全ての辺を接続する（誘導部分グラフにする）とコストは高くなる

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    Option opt;
    opt.seed = 27;

    std::cerr << opt << '\n';

    NInput::initialize(argc, argv, opt);
    NGraph::initialize(argc, argv, opt);

    dump(NInput::LA, NInput::LB);
    const int prod = NInput::LA * NInput::LB;
    const int estimate_time = 400.0 * prod / 28800;
    const double timelimit = 2500 + 400 - estimate_time;
    dump(prod, estimate_time, timelimit);

    NSteiner::set_terminals(NInput::tour_nodes);
    NSteiner::run(0);

    //NSteiner::show();

    auto g = NSteiner::to_graph();

    Xorshift rnd;
    auto used_edge = NSteiner::B;

    int tour_length = compute_tour_length(g);
    int nvertices = count_used_vertex(g);
    dump(tour_length, nvertices, NGraph::shortest_tour_length);

    int loop = 0;
    double start_time = timer.elapsed_ms(), now_time, end_time = 1250;
    double start_temp = 20.0, end_temp = 0.0;
    int vertice_coeff = 10;
    while ((now_time = timer.elapsed_ms()) < end_time) {
        loop++;
        auto mtr = modify_tree(g, used_edge, rnd);
        if (!mtr.succeed) continue;
        int ntour_length = compute_tour_length_tree(g);
        int nnvertices = count_used_vertex(g);
        int diff = (nnvertices - nvertices) * vertice_coeff + (ntour_length - tour_length);
        double temp = get_temp(start_temp, end_temp, now_time - start_time, end_time - start_time);
        double prob = exp(-diff / temp);
        if (rnd.next_double() < prob) {
            tour_length = ntour_length;
            nvertices = nnvertices;
        }
        else {
            undo_modify_tree(g, used_edge, mtr);
        }
        if (!(loop & 0xFFF)) {
            //if (timer.elapsed_ms() > 2000) {
            //    auto tour = compute_tour(g);
            //    auto [score, A] = compute_A(tour, g);
            //    if (chmin(best_score, score)) {
            //        best_g = g;
            //        dump(loop, timer.elapsed_ms(), tour_length, nvertices, score);
            //    }
            //}
            dump(loop, timer.elapsed_ms(), tour_length, nvertices);
        }
    }
    dump(loop, tour_length);

    //g = best_g;
    auto tour = compute_tour(g);
    dump(tour.size());

    auto A = compute_initial_A(g);

    SegmentModifier smod(timer, rnd, tour, A);
    smod.run(timelimit - timer.elapsed_ms());

    A = smod.A;

    std::vector<std::string> ans;
    int score;

    //for (int t = 0; t < 10; t++) {
    std::tie(score, A, ans) = solve_segment(tour, A);
    dump(score, NInput::LA, A.size());
    while (A.size() < NInput::LA) A.push_back(0);
    //}

    dump(timer.elapsed_ms());
    //assert(timer.elapsed_ms() < 2980);

    NOutput::output(argc, argv, opt, A, ans);

    dump(timer.elapsed_ms(), score);

    return 0;
}