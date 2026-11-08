// d1d.hpp -- header-only C++17 inference for the D1D / D1D-Turbo drum one-shot models.
//
// One header, standard library only (no BLAS, no ONNX, no DLLs, no other files).
// Generates kick / snare / hi-hat one-shots: 16384 samples, mono, 32 kHz (0.512 s).
//
//   #include "d1d.hpp"
//
//   d1d::Model model = d1d::Model::load("model.bin");        // or load(bytes, size)
//   d1d::GenParams p;
//   p.seed = 42;                                              // same seed -> same sound
//   std::vector<float> kick = model.generate(d1d::Drum::Kick, p);
//   d1d::save_wav("kick.wav", kick);                          // or wav_bytes(kick) in memory
//
// The same code runs both models. model.bin carries its sampler defaults
// (GenParams.steps / .cfg = 0 means "use them"):
//   D1D        50 Euler steps, CFG 2.0 = 100 net passes per sample (~12 s on 6 cores)
//   D1D-Turbo   3 Euler steps, CFG 1.0 =   3 net passes per sample (~0.35 s)
//
// Build: any C++17 compiler; link threads on POSIX (-pthread).
//   AVX2+FMA fast path: /arch:AVX2 (MSVC) or -march=native / -mavx2 -mfma (GCC/Clang).
//   Without them, or with -DD1D_NO_SIMD, a portable path (same results, slower).
//
// Threads: every Model owns its worker threads (GenParams-independent, set at load).
// Calls on ONE Model are serialised by an internal mutex; for parallel generation
// use one Model per thread. Errors throw std::runtime_error.
//
// PORTING NOTES
//  * Compute lives in conv1d() (most of the time), group_norm() and attention().
//  * Platform-specific code: only the `#if D1D_AVX2` block in conv_run().
//  * Threads: detail::ThreadPool / parallel_for() is the only use of <thread>.
//    With threads = 1 everything runs inline on the calling thread.
//  * model.bin is read byte by byte as little endian -> works on big-endian hosts.
//  * Noise: own RNG (SplitMix64 + Box-Muller): the same seed gives the same sound
//    on every compiler/platform. It is NOT the same noise as the Python generator.
//  * Architecture and inference spec: the model card (huggingface.co/sixf0ur/D1D-Turbo).
//
// model.bin format, all little endian:
//   char[4] "DDW1", u32 count, count x { u32 name_len, name, u32 ndim, u32 dims[ndim],
//   f32 data[prod(dims)] }; names = PyTorch state_dict keys + meta.steps, meta.cfg_w.
#ifndef D1D_HPP
#define D1D_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _MSC_VER
#define D1D_RESTRICT __restrict
#else
#define D1D_RESTRICT __restrict__
#endif

#if defined(__AVX2__) && defined(__FMA__) && !defined(D1D_NO_SIMD)
#define D1D_AVX2 1
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(__AVX2__) && !defined(D1D_NO_SIMD)
#define D1D_AVX2 1   // MSVC /arch:AVX2 implies FMA but does not define __FMA__
#include <immintrin.h>
#else
#define D1D_AVX2 0
#endif

namespace d1d {

// ================================================================== public constants
constexpr int kSampleRate = 32000;
constexpr int kLength = 16384;      // samples per one-shot (0.512 s)
constexpr int kOnset = 256;         // the hit starts here (8 ms pre-roll)

enum class Drum : int { Kick = 0, Snare = 1, Hat = 2 };

inline const char* drum_name(Drum d) {
    static const char* names[3] = {"kick", "snare", "hat"};
    const int i = (int)d;
    return i >= 0 && i < 3 ? names[i] : "?";
}
// "kick" / "snare" / "hat" -> Drum. Returns false for anything else.
inline bool parse_drum(const std::string& s, Drum& out) {
    for (int i = 0; i < 3; ++i)
        if (s == drum_name((Drum)i)) { out = (Drum)i; return true; }
    return false;
}

struct GenParams {
    uint64_t seed = 0;      // noise seed: same seed + same model -> same sound
    int steps = 0;          // Euler steps; 0 = model default (D1D 50, D1D-Turbo 3)
    float cfg = 0.0f;       // CFG weight; 0 = model default; 1 = one pass per step
    bool normalize = true;  // peak-normalise the output (the model's level is arbitrary)
    float peak = 0.95f;     // ... to this peak
    // Called after every Euler step with (done, total); return false to cancel
    // (generate() then returns an empty vector / false).
    std::function<bool(int, int)> progress;
};

// ================================================================== internals
namespace detail {

constexpr int L = kLength;
constexpr int COND = 128;         // conditioning vector size
constexpr int HEADS = 4;
constexpr int NS = 5;             // U-Net stages, channels 32 64 96 128 192
constexpr int BLOCKS = 2;         // res blocks per down stage (+1 per up stage)
constexpr int NULL_CLASS = 3;     // classes 0 kick, 1 snare, 2 hat; 3 = null (CFG)
constexpr int GN_GROUPS = 8;
constexpr float GN_EPS = 1e-5f;

// ------------------------------------------------------------------ threads
// Persistent workers: a net pass makes a few hundred parallel calls, so spawning
// threads per call would cost more than some of the ops themselves.
class ThreadPool {
public:
    explicit ThreadPool(int n) {
        for (int i = 1; i < n; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    int size() const { return (int)workers_.size() + 1; }

    void run(int n, const std::function<void(int)>& f) {
        if (workers_.empty() || n <= 1) {
            for (int i = 0; i < n; ++i) f(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &f;
            njob_ = n;
            next_.store(0);
            pending_ = workers_.size();
            ++gen_;
        }
        cv_.notify_all();
        work();                                   // the caller works too
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [&] { return pending_ == 0; });
        job_ = nullptr;
    }

private:
    void work() {
        for (int i; (i = next_.fetch_add(1)) < njob_;) (*job_)(i);
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            work();
            std::lock_guard<std::mutex> lk(m_);
            if (--pending_ == 0) done_.notify_one();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int)>* job_ = nullptr;
    std::atomic<int> next_{0};
    int njob_ = 0;
    size_t pending_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

// Recycles float buffers by exact size. Fresh large allocations are expensive
// (zeroed pages, page faults); a pass asks for the same sizes every time.
// Not thread-safe: tensors are created only by the calling thread, never inside
// parallel_for bodies.
class BufPool {
public:
    BufPool() = default;
    BufPool(const BufPool&) = delete;
    BufPool& operator=(const BufPool&) = delete;
    ~BufPool() {
        for (auto& kv : free_)
            for (float* p : kv.second) delete[] p;
    }
    float* get(size_t n) {
        auto it = free_.find(n);
        if (it != free_.end() && !it->second.empty()) {
            float* p = it->second.back();
            it->second.pop_back();
            return p;
        }
        return new float[n];
    }
    void put(size_t n, float* p) { free_[n].push_back(p); }

private:
    std::unordered_map<size_t, std::vector<float*>> free_;
};

// Execution context of the current call: set by Model for the duration of a
// public call (thread-local, so several Models may run on several threads).
enum { P_PAD, P_CONV, P_GN, P_ATTN, P_N };
struct Ctx {
    ThreadPool* pool = nullptr;
    BufPool* buf = nullptr;
    double* prof = nullptr;    // P_N seconds, or nullptr = not profiling
};
inline Ctx& tls() {
    thread_local Ctx c;
    return c;
}
inline BufPool& buf() {
    thread_local BufPool fallback;
    return tls().buf ? *tls().buf : fallback;
}
struct CtxScope {
    Ctx saved;
    explicit CtxScope(const Ctx& c) : saved(tls()) { tls() = c; }
    ~CtxScope() { tls() = saved; }
};
inline int nthreads() { return tls().pool ? tls().pool->size() : 1; }

// Runs f(0..n-1) on all threads of the current Model.
template <class F>
inline void parallel_for(int n, F&& f) {
    ThreadPool* pool = tls().pool;
    if (!pool || n <= 1) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    const std::function<void(int)> fn = std::ref(f);
    pool->run(n, fn);
}

struct ProfScope {
    int k;
    double* acc;
    std::chrono::steady_clock::time_point t0;
    explicit ProfScope(int kind) : k(kind), acc(tls().prof) {
        if (acc) t0 = std::chrono::steady_clock::now();
    }
    ~ProfScope() {
        if (acc) acc[k] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

// ------------------------------------------------------------------ storage
struct Arr {
    std::vector<uint32_t> shape;
    std::vector<float> d;
};
using Store = std::map<std::string, Arr>;

// Parses the DDW1 format from memory, bounds-checked, endian-independent.
inline Store parse_bin(const uint8_t* p, size_t size, const std::string& what) {
    size_t pos = 0;
    auto need = [&](size_t n) {
        if (n > size - pos) throw std::runtime_error("d1d: truncated " + what);
    };
    auto u32 = [&]() {
        need(4);
        const uint32_t v = (uint32_t)p[pos] | (uint32_t)p[pos + 1] << 8 |
                           (uint32_t)p[pos + 2] << 16 | (uint32_t)p[pos + 3] << 24;
        pos += 4;
        return v;
    };
    need(4);
    if (std::memcmp(p, "DDW1", 4) != 0) throw std::runtime_error("d1d: not a model file (magic) " + what);
    pos = 4;
    const uint16_t one = 1;
    const bool little = *(const uint8_t*)&one == 1;
    const uint32_t count = u32();
    Store s;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t nl = u32();
        need(nl);
        std::string name((const char*)p + pos, nl);
        pos += nl;
        const uint32_t nd = u32();
        if (nd > 8) throw std::runtime_error("d1d: bad tensor rank in " + what);
        Arr a;
        a.shape.resize(nd);
        uint64_t n = 1;
        for (uint32_t k = 0; k < nd; ++k) {
            a.shape[k] = u32();
            n *= a.shape[k];
        }
        if (n > (size - pos) / 4) throw std::runtime_error("d1d: truncated " + what);
        a.d.resize((size_t)n);
        if (little) {
            std::memcpy(a.d.data(), p + pos, (size_t)n * 4);
            pos += (size_t)n * 4;
        } else {
            for (size_t k = 0; k < (size_t)n; ++k) {
                const uint32_t v = u32();
                std::memcpy(&a.d[k], &v, 4);
            }
        }
        s[name] = std::move(a);
    }
    return s;
}

inline std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("d1d: cannot open " + path);
    const std::streamoff n = f.tellg();
    std::vector<uint8_t> b((size_t)n);
    f.seekg(0);
    if (n > 0 && !f.read((char*)b.data(), n)) throw std::runtime_error("d1d: cannot read " + path);
    return b;
}

// (C, L) activation, row-major: channel c is data()[c*L .. c*L+L).
// Memory is NOT initialised; every producer writes all of it.
struct Tensor {
    int C = 0, L = 0;
    float* p = nullptr;
    BufPool* pool = nullptr;
    Tensor() = default;
    Tensor(int c, int l) : C(c), L(l), pool(&buf()) { p = pool->get((size_t)c * l); }
    Tensor(const Tensor& o) : Tensor(o.C, o.L) { std::memcpy(p, o.p, o.size() * sizeof(float)); }
    Tensor(Tensor&& o) noexcept : C(o.C), L(o.L), p(o.p), pool(o.pool) { o.p = nullptr; }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            release();
            C = o.C; L = o.L; p = o.p; pool = o.pool;
            o.p = nullptr;
        }
        return *this;
    }
    Tensor& operator=(const Tensor& o) {
        if (this != &o) *this = Tensor(o);
        return *this;
    }
    ~Tensor() { release(); }
    size_t size() const { return (size_t)C * L; }
    float* data() { return p; }
    const float* data() const { return p; }
    float* row(int c) { return p + (size_t)c * L; }
    const float* row(int c) const { return p + (size_t)c * L; }

private:
    void release() {
        if (p) pool->put(size(), p);
        p = nullptr;
    }
};

// Channel-concatenation of one or two tensors, without copying: [a | b].
// Used for the U-Net skip connections ("concat([h, skip])").
struct View {
    const Tensor* a;
    const Tensor* b;
    View(const Tensor& x) : a(&x), b(nullptr) {}                // NOLINT: implicit on purpose
    View(const Tensor& x, const Tensor& y) : a(&x), b(&y) {
        if (x.L != y.L) throw std::runtime_error("d1d: concat length mismatch");
    }
    int C() const { return a->C + (b ? b->C : 0); }
    int L() const { return a->L; }
    const float* row(int c) const { return c < a->C ? a->row(c) : b->row(c - a->C); }
};

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Sum of a row with 8 independent float lanes (vectorisable without fast-math),
// flushed into double every 512 values so long rows keep full precision.
inline double row_sum(const float* D1D_RESTRICT x, int n, float sub = 0.0f, bool square = false) {
    double total = 0.0;
    for (int i0 = 0; i0 < n; i0 += 512) {
        const int i1 = std::min(n, i0 + 512);
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int i = i0;
        if (square)
            for (; i + 8 <= i1; i += 8)
                for (int u = 0; u < 8; ++u) {
                    const float d = x[i + u] - sub;
                    acc[u] += d * d;
                }
        else
            for (; i + 8 <= i1; i += 8)
                for (int u = 0; u < 8; ++u) acc[u] += x[i + u];
        for (; i < i1; ++i) acc[0] += square ? (x[i] - sub) * (x[i] - sub) : x[i];
        for (int u = 0; u < 8; ++u) total += acc[u];
    }
    return total;
}

// ------------------------------------------------------------------ conv1d
// PyTorch Conv1d semantics: cross-correlation, zero padding, weight (Cout, Cin, K).
//
// The output is computed in register blocks of CB channels x LV samples: each
// input value loaded feeds CB multiply-adds. Weights are re-packed once at load
// time to [Cout/CB][Cin][K][CB] so the CB weights used together are adjacent.
constexpr int CB = 4;    // output channels per block
constexpr int LV = 16;   // output samples per block (2 AVX2 registers)
constexpr int TL = 256;  // output samples per thread task (input tile stays in L2)

struct Conv {
    int cin = 0, cout = 0, k = 0, stride = 1, pad = 0;
    std::vector<float> wp;   // packed weights
    std::vector<float> b;    // bias (cout)
};

inline Conv make_conv(const Store& S, const std::string& name, int stride, int pad) {
    const Arr& w = S.at(name + ".weight");
    if (w.shape.size() != 3) throw std::runtime_error("d1d: " + name + " is not a conv weight");
    Conv cv;
    cv.cout = (int)w.shape[0];
    cv.cin = (int)w.shape[1];
    cv.k = (int)w.shape[2];
    cv.stride = stride;
    cv.pad = pad;
    const int nb = (cv.cout + CB - 1) / CB;
    cv.wp.assign((size_t)nb * cv.cin * cv.k * CB, 0.0f);   // zero rows pad Cout to CB
    for (int co = 0; co < cv.cout; ++co)
        for (int ci = 0; ci < cv.cin; ++ci)
            for (int kk = 0; kk < cv.k; ++kk)
                cv.wp[(((size_t)(co / CB) * cv.cin + ci) * cv.k + kk) * CB + co % CB] =
                    w.d[((size_t)co * cv.cin + ci) * cv.k + kk];
    auto bi = S.find(name + ".bias");
    if (bi == S.end() || (int)bi->second.d.size() != cv.cout)
        throw std::runtime_error("d1d: missing or bad " + name + ".bias");
    cv.b = bi->second.d;
    return cv;
}

// y = conv(xp) [+ add]. xp: padded input, Lpp floats per channel.
template <int STRIDE>
inline void conv_run(const Conv& cv, const float* D1D_RESTRICT xp, int Lpp, Tensor& y,
                     const Tensor* add) {
    const int Lout = y.L, cin = cv.cin, K = cv.k, cout = cv.cout;
    const int nLt = (Lout + TL - 1) / TL;
    const int nCb = (cout + CB - 1) / CB;
    // enough tasks to keep every thread busy even at short lengths
    int nCg = std::min(nCb, std::max(1, (4 * nthreads() + nLt - 1) / nLt));
    const int cbPer = (nCb + nCg - 1) / nCg;
    nCg = (nCb + cbPer - 1) / cbPer;

    parallel_for(nLt * nCg, [&](int task) {
        const int lt = task % nLt, cg = task / nLt;
        const int l0 = lt * TL, l1 = std::min(Lout, l0 + TL);
        const int cb1 = std::min(nCb, (cg + 1) * cbPer);
        for (int cb = cg * cbPer; cb < cb1; ++cb) {
            const float* D1D_RESTRICT wb = cv.wp.data() + (size_t)cb * cin * K * CB;
            float bias[CB];
            for (int c = 0; c < CB; ++c) {
                const int co = cb * CB + c;
                bias[c] = co < cout ? cv.b[co] : 0.0f;
            }
            for (int l = l0; l < l1; l += LV) {
                alignas(32) float acc[CB][LV];
#if D1D_AVX2
                if constexpr (STRIDE == 1) {   // explicit SIMD: 4 ch x 16 samples in 8 registers
                    __m256 a0[CB], a1[CB];
                    for (int c = 0; c < CB; ++c) a0[c] = a1[c] = _mm256_set1_ps(bias[c]);
                    for (int ci = 0; ci < cin; ++ci) {
                        const float* xr = xp + (size_t)ci * Lpp + l;
                        const float* wc = wb + (size_t)ci * K * CB;
                        for (int kk = 0; kk < K; ++kk) {
                            const __m256 x0 = _mm256_loadu_ps(xr + kk);
                            const __m256 x1 = _mm256_loadu_ps(xr + kk + 8);
                            const float* w4 = wc + kk * CB;
                            for (int c = 0; c < CB; ++c) {
                                const __m256 wv = _mm256_broadcast_ss(w4 + c);
                                a0[c] = _mm256_fmadd_ps(wv, x0, a0[c]);
                                a1[c] = _mm256_fmadd_ps(wv, x1, a1[c]);
                            }
                        }
                    }
                    for (int c = 0; c < CB; ++c) {
                        _mm256_store_ps(acc[c], a0[c]);
                        _mm256_store_ps(acc[c] + 8, a1[c]);
                    }
                } else
#endif
                {
                    for (int c = 0; c < CB; ++c)
                        for (int v = 0; v < LV; ++v) acc[c][v] = bias[c];
                    for (int ci = 0; ci < cin; ++ci) {
                        const float* D1D_RESTRICT xr = xp + (size_t)ci * Lpp + (size_t)l * STRIDE;
                        const float* D1D_RESTRICT wc = wb + (size_t)ci * K * CB;
                        for (int kk = 0; kk < K; ++kk) {
                            const float* D1D_RESTRICT xk = xr + kk;
                            const float* D1D_RESTRICT w4 = wc + kk * CB;
                            for (int c = 0; c < CB; ++c) {
                                const float wv = w4[c];
                                for (int v = 0; v < LV; ++v) acc[c][v] += wv * xk[v * STRIDE];
                            }
                        }
                    }
                }
                const int nv = std::min(LV, l1 - l);
                for (int c = 0; c < CB; ++c) {
                    const int co = cb * CB + c;
                    if (co >= cout) continue;
                    float* dst = y.row(co) + l;
                    if (add) {
                        const float* ar = add->row(co) + l;
                        for (int v = 0; v < nv; ++v) dst[v] = acc[c][v] + ar[v];
                    } else {
                        std::memcpy(dst, acc[c], nv * sizeof(float));
                    }
                }
            }
        }
    });
}

// Conv1d on a (possibly concatenated) input, optionally nearest-x2-upsampling
// the input first (the U-Net's upsample) and adding a residual `add`
// (Cout, Lout) to the output -- all three fused into the one padded copy.
inline Tensor conv1d(const Conv& cv, const View& x, bool up2 = false, const Tensor* add = nullptr) {
    if (x.C() != cv.cin) throw std::runtime_error("d1d: conv1d channel mismatch");
    const int Lin = x.L() * (up2 ? 2 : 1);
    const int Lp = Lin + 2 * cv.pad;
    const int Lout = (Lp - cv.k) / cv.stride + 1;
    // zero-padded input + slack at the end: a register block may read up to
    // LV*stride + K samples past the last valid output (those results are dropped).
    const int Lpp = Lp + LV * cv.stride + cv.k;
    Tensor xp(cv.cin, Lpp);
    {
        ProfScope ps(P_PAD);
        parallel_for(cv.cin, [&](int c) {
            float* d = xp.row(c);
            const float* s = x.row(c);
            std::fill(d, d + cv.pad, 0.0f);
            if (up2)
                for (int i = 0; i < x.L(); ++i) d[cv.pad + 2 * i] = d[cv.pad + 2 * i + 1] = s[i];
            else
                std::memcpy(d + cv.pad, s, Lin * sizeof(float));
            std::fill(d + cv.pad + Lin, d + Lpp, 0.0f);
        });
    }
    Tensor y(cv.cout, Lout);
    if (add && (add->C != cv.cout || add->L != Lout)) throw std::runtime_error("d1d: conv1d bad residual");
    ProfScope ps(P_CONV);
    if (cv.stride == 1) conv_run<1>(cv, xp.data(), Lpp, y, add);
    else if (cv.stride == 2) conv_run<2>(cv, xp.data(), Lpp, y, add);
    else throw std::runtime_error("d1d: conv1d unsupported stride");
    return y;
}

// ------------------------------------------------------------------ group norm
// GroupNorm(min(8,C), C), biased variance, eps 1e-5, per-channel affine.
// Optional FiLM afterwards: out = GN(x) * (1 + film_g) + film_b. Optional SiLU.
// x may be a concatenation; returns a new tensor.
inline Tensor group_norm(const View& x, const float* gamma, const float* beta,
                         const float* film_g, const float* film_b, bool act) {
    ProfScope ps(P_GN);
    const int C = x.C(), Ln = x.L(), G = std::min(GN_GROUPS, C), cg = C / G;
    std::vector<double> s(C), mean(G), inv(G);
    parallel_for(C, [&](int c) { s[c] = row_sum(x.row(c), Ln); });
    for (int g = 0; g < G; ++g) {
        double t = 0;
        for (int c = g * cg; c < (g + 1) * cg; ++c) t += s[c];
        mean[g] = t / ((double)cg * Ln);
    }
    // two-pass variance (like numpy's var): no cancellation for large means
    parallel_for(C, [&](int c) { s[c] = row_sum(x.row(c), Ln, (float)mean[c / cg], true); });
    for (int g = 0; g < G; ++g) {
        double t = 0;
        for (int c = g * cg; c < (g + 1) * cg; ++c) t += s[c];
        inv[g] = 1.0 / std::sqrt(t / ((double)cg * Ln) + (double)GN_EPS);
    }
    Tensor y(C, Ln);
    parallel_for(C, [&](int c) {
        const int g = c / cg;
        float a = (float)(gamma[c] * inv[g]);
        float b = (float)(beta[c] - mean[g] * gamma[c] * inv[g]);
        if (film_g) {
            const float s1 = 1.0f + film_g[c];
            a *= s1;
            b = b * s1 + film_b[c];
        }
        const float* D1D_RESTRICT xr = x.row(c);
        float* D1D_RESTRICT yr = y.row(c);
        if (act)
            for (int i = 0; i < Ln; ++i) yr[i] = silu(xr[i] * a + b);
        else
            for (int i = 0; i < Ln; ++i) yr[i] = xr[i] * a + b;
    });
    return y;
}

// y = W x + b, W (out, in) as nn.Linear
inline std::vector<float> linear(const Arr& W, const Arr& b, const std::vector<float>& x) {
    const int out = (int)W.shape[0], in = (int)W.shape[1];
    std::vector<float> y(out);
    for (int o = 0; o < out; ++o) {
        float s = b.d[o];
        for (int i = 0; i < in; ++i) s += W.d[(size_t)o * in + i] * x[i];
        y[o] = s;
    }
    return y;
}

// Portable N(0,1) noise: SplitMix64 + Box-Muller. Same seed -> same noise everywhere.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }   // (0,1)
    void normal(float* out, size_t n) {
        for (size_t i = 0; i < n; i += 2) {
            const double r = std::sqrt(-2.0 * std::log(uniform())), a = 6.283185307179586 * uniform();
            out[i] = (float)(r * std::cos(a));
            if (i + 1 < n) out[i + 1] = (float)(r * std::sin(a));
        }
    }
};

// ------------------------------------------------------------------ network
class Net {
public:
    explicit Net(Store s) : S_(std::move(s)) {
        check_architecture();
        for (auto& kv : S_) {
            const std::string& n = kv.first;
            const size_t dot = n.rfind(".weight");
            if (dot == std::string::npos || dot + 7 != n.size() || kv.second.shape.size() != 3)
                continue;
            const std::string base = n.substr(0, dot);
            const int k = (int)kv.second.shape[2];
            const bool down = base.rfind("downsample.", 0) == 0;
            // downsample: k4 s2 p1; everything else is "same" padding (k=1,3,7)
            convs_[base] = make_conv(S_, base, down ? 2 : 1, down ? 1 : k / 2);
        }
        // fixed sin-cos position table for the attention at L=1024, C=192:
        // P[c][pos] = c < 96 ? sin(pos*w_c) : cos(pos*w_{c-96}), w_k = 1/10000^(k/96),
        // computed in double like numpy_ref.sincos_1d, used as (C, L).
        const int C = 192, Lq = 1024, half = C / 2;
        pos_.assign((size_t)C * Lq, 0.0f);
        for (int k = 0; k < half; ++k) {
            const double w = 1.0 / std::pow(10000.0, (double)k / half);
            for (int p = 0; p < Lq; ++p) {
                pos_[(size_t)k * Lq + p] = (float)std::sin(p * w);
                pos_[(size_t)(k + half) * Lq + p] = (float)std::cos(p * w);
            }
        }
    }

    // One net pass: x (L) noisy waveform, t in [0,1), y class 0..3 -> velocity v (L).
    void forward(const float* x_in, float t, int y, float* v_out) const {
        const std::vector<float> c = conditioning(t, y);
        Tensor x(1, L);
        std::memcpy(x.data(), x_in, L * sizeof(float));

        // Skip stack (LIFO). Reserved up front, so references to back() stay valid
        // while the next tensor is computed and pushed.
        std::vector<Tensor> skips;
        skips.reserve(16);
        skips.push_back(conv1d(cv("stem"), x));
        for (int i = 0; i < NS; ++i) {                     // down: 16384 -> 1024
            for (int j = 0; j < BLOCKS; ++j)
                skips.push_back(res_block(pfx("down", i, j), skips.back(), c));
            if (i < NS - 1)
                skips.push_back(conv1d(cv("downsample." + std::to_string(i)), skips.back()));
        }
        Tensor h = res_block("mid1", skips.back(), c);     // bottleneck @ 1024
        h = attention("mid_attn", h);
        h = res_block("mid2", h, c);
        for (int u = 0; u < NS; ++u) {                     // up: u = 0 is the deepest
            for (int j = 0; j < BLOCKS + 1; ++j) {
                h = res_block(pfx("up", u, j), View(h, skips.back()), c);   // [h, skip]
                skips.pop_back();
            }
            const std::string a = "up_attn." + std::to_string(u);
            if (has(a + ".qkv.weight")) h = attention(a, h);
            const std::string us = "upsample." + std::to_string(u);
            if (has(us + ".weight")) h = conv1d(cv(us), h, /*up2=*/true);
        }
        if (!skips.empty()) throw std::runtime_error("d1d: skip stack not drained");
        h = group_norm(h, P("out_norm.weight"), P("out_norm.bias"), nullptr, nullptr, true);
        const Tensor v = conv1d(cv("out"), h);
        if (!gated()) {
            std::memcpy(v_out, v.data(), L * sizeof(float));
            return;
        }
        // Learned output gate (D1D-Turbo): the predicted sound x1 = x + (1-t) v is
        // scaled by g = clamp(1 + gate(h), 0, 1), so g == 0 is exact digital silence.
        // Written as a velocity, same op order as PyTorch:
        // v_eff = g v + (g - 1) x / max(1 - t, 1e-3).
        const Tensor gt = conv1d(cv("gate"), h);
        const float om = std::max(1.0f - t, 1e-3f);
        const float* vp = v.data();
        const float* gp = gt.data();
        for (int j = 0; j < L; ++j) {
            const float g = std::min(1.0f, std::max(0.0f, 1.0f + gp[j]));
            v_out[j] = g * vp[j] + (g - 1.0f) * x_in[j] / om;
        }
    }

    bool gated() const { return has("gate.weight"); }
    float meta(const std::string& n, float fallback) const {
        auto it = S_.find(n);
        return it != S_.end() && !it->second.d.empty() ? it->second.d[0] : fallback;
    }

private:
    Store S_;
    std::map<std::string, Conv> convs_;
    std::vector<float> pos_;

    bool has(const std::string& n) const { return S_.count(n) != 0; }
    const Arr& A(const std::string& n) const {
        auto it = S_.find(n);
        if (it == S_.end()) throw std::runtime_error("d1d: missing weight " + n);
        return it->second;
    }
    const float* P(const std::string& n) const { return A(n).d.data(); }
    const Conv& cv(const std::string& n) const {
        auto it = convs_.find(n);
        if (it == convs_.end()) throw std::runtime_error("d1d: missing conv " + n);
        return it->second;
    }
    static std::string pfx(const char* s, int i, int j) {
        return std::string(s) + "." + std::to_string(i) + "." + std::to_string(j);
    }
    // a few shape checks, so a wrong / foreign file fails with a clear message
    void check_architecture() const {
        auto expect = [&](const char* n, std::vector<uint32_t> shape) {
            auto it = S_.find(n);
            if (it == S_.end() || it->second.shape != shape)
                throw std::runtime_error(std::string("d1d: unexpected model architecture (") + n + ")");
        };
        expect("stem.weight", {32, 1, 7});
        expect("cls_emb.weight", {4, (uint32_t)COND});
        expect("mid_attn.qkv.weight", {576, 192, 1});
        expect("mid_attn.pos_gate", {1, 192, 1});
        expect("out.weight", {1, 32, 7});
    }

    // c = t_mlp(timestep_embedding(t)) + cls_emb[y]
    std::vector<float> conditioning(float t, int y) const {
        // timestep embedding, float32 like PyTorch: freqs = exp(-ln(1e4) * k / 64),
        // args = t * freqs * 1000 (NOTE the x1000), emb = [cos | sin] (cos first)
        const int half = COND / 2;
        std::vector<float> e(COND);
        const float nl = -(float)std::log(10000.0);
        for (int k = 0; k < half; ++k) {
            const float fr = std::exp(nl * (float)k / (float)half);
            const float a = t * fr * 1000.0f;
            e[k] = std::cos(a);
            e[k + half] = std::sin(a);
        }
        std::vector<float> h = linear(A("t_mlp.0.weight"), A("t_mlp.0.bias"), e);
        for (float& v : h) v = silu(v);
        std::vector<float> c = linear(A("t_mlp.2.weight"), A("t_mlp.2.bias"), h);
        const float* emb = P("cls_emb.weight") + (size_t)y * COND;
        for (int i = 0; i < COND; ++i) c[i] += emb[i];
        return c;
    }

    // FiLM residual block. x may be a concat [h, skip].
    Tensor res_block(const std::string& p, const View& x, const std::vector<float>& c) const {
        Tensor h = conv1d(cv(p + ".c1"),
                          group_norm(x, P(p + ".n1.weight"), P(p + ".n1.bias"), nullptr, nullptr, true));
        const std::vector<float> gb = linear(A(p + ".emb.weight"), A(p + ".emb.bias"), c);
        // first half = scale g, second half = shift b
        h = group_norm(h, P(p + ".n2.weight"), P(p + ".n2.bias"), gb.data(), gb.data() + h.C, true);
        if (has(p + ".skip.weight")) {
            const Tensor s = conv1d(cv(p + ".skip"), x);
            return conv1d(cv(p + ".c2"), h, false, &s);    // c2(h) + skip(x)
        }
        if (x.b) throw std::runtime_error("d1d: " + p + ": concat input without skip conv");
        return conv1d(cv(p + ".c2"), h, false, x.a);       // c2(h) + x
    }

    // Self-attention over time, only at L=1024, C=192.
    Tensor attention(const std::string& p, const Tensor& x) const {
        const int C = x.C, Ln = x.L, D = C / HEADS;
        Tensor h = group_norm(x, P(p + ".n.weight"), P(p + ".n.bias"), nullptr, nullptr, false);
        if (has(p + ".pos_gate")) {
            if ((size_t)C * Ln != pos_.size()) throw std::runtime_error("d1d: pos table shape");
            const float* gate = P(p + ".pos_gate");
            for (int ch = 0; ch < C; ++ch) {
                float* hr = h.row(ch);
                const float* pr = pos_.data() + (size_t)ch * Ln;
                for (int i = 0; i < Ln; ++i) hr[i] += gate[ch] * pr[i];
            }
        }
        const Tensor qkv = conv1d(cv(p + ".qkv"), h);   // (3C, L): [q heads | k heads | v heads]
        Tensor o(C, Ln);
        {
            ProfScope ps(P_ATTN);
            const float scale = 1.0f / std::sqrt((float)D);
            const int RB = 16;                             // query rows per task
            const int nrb = (Ln + RB - 1) / RB;
            // Layout stays (channel, time): the scores of one query are a sum over d
            // of q[d] * K[d][0..L), the output a dot product with V[d][0..L) -- both
            // contiguous, so no transposes. Reductions use 8 float lanes.
            parallel_for(HEADS * nrb, [&](int task) {
                const int hd = task / nrb, r0 = (task % nrb) * RB, r1 = std::min(Ln, r0 + RB);
                std::vector<float> s(Ln);
                for (int i = r0; i < r1; ++i) {
                    std::fill(s.begin(), s.end(), 0.0f);
                    for (int d = 0; d < D; ++d) {
                        const float q = qkv.row(hd * D + d)[i] * scale;
                        const float* D1D_RESTRICT k = qkv.row(C + hd * D + d);
                        float* D1D_RESTRICT sp = s.data();
                        for (int j = 0; j < Ln; ++j) sp[j] += q * k[j];
                    }
                    float mx = s[0];
                    for (int j = 1; j < Ln; ++j) mx = std::max(mx, s[j]);
                    for (int j = 0; j < Ln; ++j) s[j] = std::exp(s[j] - mx);
                    const float inv = (float)(1.0 / row_sum(s.data(), Ln));
                    for (int d = 0; d < D; ++d) {
                        const float* D1D_RESTRICT v = qkv.row(2 * C + hd * D + d);
                        const float* D1D_RESTRICT sp = s.data();
                        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        for (int j = 0; j + 8 <= Ln; j += 8)
                            for (int u = 0; u < 8; ++u) acc[u] += sp[j + u] * v[j + u];
                        float t = 0.0f;
                        for (int u = 0; u < 8; ++u) t += acc[u];
                        o.row(hd * D + d)[i] = t * inv;
                    }
                }
            });
        }
        return conv1d(cv(p + ".proj"), o, false, &x);    // proj(o) + x
    }
};

struct Impl {
    Net net;
    std::unique_ptr<ThreadPool> pool;
    BufPool buf;
    std::mutex mu;
    bool profiling = false;
    double prof[P_N] = {};
    Impl(Store s, int threads)
        : net(std::move(s)),
          pool(new ThreadPool(threads > 0 ? threads
                                          : (int)std::max(1u, std::thread::hardware_concurrency()))) {}
    Ctx ctx() { return Ctx{pool.get(), &buf, profiling ? prof : nullptr}; }
};

}  // namespace detail

// ================================================================== public API
class Model {
public:
    // threads: 0 = all logical cores, 1 = run on the calling thread only.
    static Model load(const std::string& path, int threads = 0) {
        const std::vector<uint8_t> b = detail::read_file(path);
        return Model(detail::parse_bin(b.data(), b.size(), path), threads);
    }
    // From memory, e.g. an embedded array or a downloaded buffer. The bytes are
    // copied; the buffer may be freed afterwards.
    static Model load(const void* data, size_t size, int threads = 0) {
        if (!data) throw std::runtime_error("d1d: null model buffer");
        return Model(detail::parse_bin((const uint8_t*)data, size, "model buffer"), threads);
    }

    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

    int default_steps() const { return std::max(1, (int)p_->net.meta("meta.steps", 3.0f)); }
    float default_cfg() const { return p_->net.meta("meta.cfg_w", 1.0f); }
    bool gated() const { return p_->net.gated(); }       // true for D1D-Turbo
    int threads() const { return p_->pool->size(); }

    // One one-shot: kLength samples at kSampleRate. Empty if cancelled by progress.
    std::vector<float> generate(Drum drum, const GenParams& p = GenParams()) const {
        std::vector<float> out(kLength);
        if (!generate(drum, p, out.data())) out.clear();
        return out;
    }
    // Same, into a caller-provided buffer of kLength floats. False if cancelled.
    bool generate(Drum drum, const GenParams& p, float* out) const {
        check_class((int)drum);
        make_noise(p.seed, out);
        if (!sample((int)drum, out, p.steps > 0 ? p.steps : default_steps(),
                    p.cfg > 0.0f ? p.cfg : default_cfg(), p.progress))
            return false;
        if (p.normalize) {
            float peak = 0.0f;
            for (int i = 0; i < kLength; ++i) peak = std::max(peak, std::fabs(out[i]));
            if (peak > 1e-6f)
                for (int i = 0; i < kLength; ++i) out[i] *= p.peak / peak;
        }
        return true;
    }

    // ---- low level (what generate() is made of; used by drumgen --test)
    // The start noise generate() uses for a seed: kLength floats ~ N(0, 1).
    static void make_noise(uint64_t seed, float* out) { detail::Rng(seed).normal(out, kLength); }

    // Rectified-flow Euler sampler, in place: x = start noise in, raw sound out
    // (~unit RMS, not normalised). cls 0..2; cfg 1 = no guidance. False if cancelled.
    bool sample(int cls, float* x, int steps, float cfg,
                const std::function<bool(int, int)>& progress = nullptr) const {
        check_class(cls);
        if (steps < 1) throw std::runtime_error("d1d: steps must be >= 1");
        std::lock_guard<std::mutex> lk(p_->mu);
        detail::CtxScope scope(p_->ctx());
        std::vector<float> v(kLength), vu(kLength);
        const double dt = 1.0 / steps;
        const float dtf = (float)dt;
        for (int i = 0; i < steps; ++i) {
            const float t = (float)(i * dt);              // 0, 1/steps, ... never 1
            p_->net.forward(x, t, cls, v.data());
            if (cfg != 1.0f) {
                p_->net.forward(x, t, detail::NULL_CLASS, vu.data());
                for (int j = 0; j < kLength; ++j) v[j] = vu[j] + cfg * (v[j] - vu[j]);
            }
            for (int j = 0; j < kLength; ++j) x[j] += v[j] * dtf;
            if (progress && !progress(i + 1, steps)) return false;
        }
        return true;
    }

    // One network pass: x (kLength) at time t in [0,1), class 0..3 (3 = null) -> v.
    void forward(const float* x, float t, int cls, float* v) const {
        if (cls < 0 || cls > detail::NULL_CLASS) throw std::runtime_error("d1d: class out of range");
        std::lock_guard<std::mutex> lk(p_->mu);
        detail::CtxScope scope(p_->ctx());
        p_->net.forward(x, t, cls, v);
    }

    // Wall time per op kind, accumulated while profiling is on:
    // [conv padding, conv compute, group norm, attention] in seconds.
    void set_profiling(bool on) {
        p_->profiling = on;
        std::fill(p_->prof, p_->prof + detail::P_N, 0.0);
    }
    std::array<double, 4> profile_seconds() const {
        return {p_->prof[0], p_->prof[1], p_->prof[2], p_->prof[3]};
    }

private:
    explicit Model(detail::Store s, int threads) : p_(new detail::Impl(std::move(s), threads)) {}
    static void check_class(int c) {
        if (c < 0 || c > 2) throw std::runtime_error("d1d: drum class must be 0 (kick), 1 (snare) or 2 (hat)");
    }
    std::unique_ptr<detail::Impl> p_;
};

// ================================================================== WAV output
// Mono WAV file image in memory. 16-bit PCM (default) or 32-bit float.
// Samples are clamped to [-1, 1] for PCM; generate() already peak-normalises.
inline std::vector<uint8_t> wav_bytes(const float* x, size_t n, int sample_rate = kSampleRate,
                                      bool float32 = false) {
    const uint32_t bps = float32 ? 4 : 2, data = (uint32_t)(n * bps);
    std::vector<uint8_t> w;
    w.reserve(44 + data);
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) w.push_back((uint8_t)(v >> (8 * i))); };
    auto u16 = [&](uint16_t v) { w.push_back((uint8_t)v); w.push_back((uint8_t)(v >> 8)); };
    auto tag = [&](const char* s) { w.insert(w.end(), s, s + 4); };
    tag("RIFF"); u32(36 + data); tag("WAVE");
    tag("fmt "); u32(16); u16(float32 ? 3 : 1); u16(1); u32((uint32_t)sample_rate);
    u32((uint32_t)sample_rate * bps); u16((uint16_t)bps); u16((uint16_t)(8 * bps));
    tag("data"); u32(data);
    for (size_t i = 0; i < n; ++i) {
        if (float32) {
            uint32_t v;
            std::memcpy(&v, &x[i], 4);
            u32(v);
        } else {
            const float c = std::max(-1.0f, std::min(1.0f, x[i]));
            u16((uint16_t)(int16_t)std::lrint(c * 32767.0f));
        }
    }
    return w;
}
inline std::vector<uint8_t> wav_bytes(const std::vector<float>& x, int sample_rate = kSampleRate,
                                      bool float32 = false) {
    return wav_bytes(x.data(), x.size(), sample_rate, float32);
}

inline void save_wav(const std::string& path, const float* x, size_t n, int sample_rate = kSampleRate,
                     bool float32 = false) {
    const std::vector<uint8_t> w = wav_bytes(x, n, sample_rate, float32);
    std::ofstream f(path, std::ios::binary);
    if (!f || !f.write((const char*)w.data(), (std::streamsize)w.size()))
        throw std::runtime_error("d1d: cannot write " + path);
}
inline void save_wav(const std::string& path, const std::vector<float>& x, int sample_rate = kSampleRate,
                     bool float32 = false) {
    save_wav(path, x.data(), x.size(), sample_rate, float32);
}

}  // namespace d1d

#endif  // D1D_HPP
