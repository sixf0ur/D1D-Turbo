// drumgen.cpp -- command-line generator for D1D / D1D-Turbo, built on d1d.hpp.
//
//   drumgen --model model.bin --cls all --n 4             generate WAVs
//   drumgen --model model.bin --test test_io.bin          verify against PyTorch
//   drumgen --model model.bin --bench                     time one net pass
//   drumgen --model model.bin --profile                   ... with time per op kind
//
// All inference code is in d1d.hpp (header-only library); this file is only the
// CLI. model.bin carries its sampler defaults (D1D: 50 steps, cfg 2.0; D1D-Turbo:
// 3 steps, cfg 1.0), overridable with --steps / --cfg.
//
// Build: build.bat (MSVC) or build.sh (GCC/Clang), or simply
//   cl /O2 /arch:AVX2 /std:c++17 /EHsc /MT drumgen.cpp
//   g++ -O3 -march=native -std=c++17 -pthread drumgen.cpp -o drumgen
#include "d1d.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a) {
    return std::chrono::duration<double>(Clock::now() - a).count();
}

// Compares this build against PyTorch reference outputs (test_io.bin).
static int run_test(const d1d::Model& m, const std::string& path) {
    const std::vector<uint8_t> bytes = d1d::detail::read_file(path);
    const d1d::detail::Store T = d1d::detail::parse_bin(bytes.data(), bytes.size(), path);
    const int L = d1d::kLength;
    const auto &px = T.at("pass_x"), &pt = T.at("pass_t"), &py = T.at("pass_y"), &pv = T.at("pass_v");
    double worst_pass = 0.0;
    std::vector<float> v(L);
    for (int r = 0; r < (int)pt.d.size(); ++r) {
        m.forward(px.d.data() + (size_t)r * L, pt.d[r], (int)py.d[r], v.data());
        double e = 0.0, ref = 0.0;
        for (int j = 0; j < L; ++j) {
            e = std::max(e, (double)std::fabs(v[j] - pv.d[(size_t)r * L + j]));
            ref = std::max(ref, (double)std::fabs(pv.d[(size_t)r * L + j]));
        }
        std::printf("pass %d  t=%.3f y=%d  max|err| %.2e  (max|v| %.2f)\n", r, pt.d[r],
                    (int)py.d[r], e, ref);
        worst_pass = std::max(worst_pass, e);
    }
    const auto &sn = T.at("samp_noise"), &so = T.at("samp_out");
    const int steps = (int)T.at("samp_steps").d[0];
    const float cfg = T.count("samp_cfg") ? T.at("samp_cfg").d[0] : 1.0f;
    double worst_samp = 0.0;
    for (int r = 0; r < 3; ++r) {
        std::vector<float> x(sn.d.begin() + (size_t)r * L, sn.d.begin() + (size_t)(r + 1) * L);
        m.sample(r, x.data(), steps, cfg);
        double e = 0.0;
        for (int j = 0; j < L; ++j) e = std::max(e, (double)std::fabs(x[j] - so.d[(size_t)r * L + j]));
        std::printf("sampler %-5s %d steps cfg %.1f  max|err| %.2e\n", d1d::drum_name((d1d::Drum)r),
                    steps, cfg, e);
        worst_samp = std::max(worst_samp, e);
    }
    // fp32 port tolerance: below ~1e-4 per pass
    const bool ok = worst_pass < 1e-4 && worst_samp < 1e-3;
    std::printf("%s  (worst pass %.2e, worst sampler %.2e)\n", ok ? "PASS" : "FAIL", worst_pass,
                worst_samp);
    return ok ? 0 : 1;
}

static void usage() {
    std::printf(
        "drumgen [--model model.bin] [options]\n"
        "  --cls kick|snare|hat|all   what to generate (default all)\n"
        "  --n N                      samples per class (default 4)\n"
        "  --steps S                  Euler steps (default: stored in model.bin)\n"
        "  --cfg W                    CFG weight, 1.0 = one pass per step (default: model.bin)\n"
        "  --seed S                   base seed; sample k uses S+k (default: time based)\n"
        "  --threads T                threads (default: all logical cores)\n"
        "  --out DIR                  output folder (default out_cpp)\n"
        "  --float                    write 32-bit float WAVs instead of 16-bit PCM\n"
        "  --test test_io.bin         verify against PyTorch reference, then exit\n"
        "  --bench                    time net passes, then exit\n"
        "  --profile                  like --bench, with time per op kind\n");
}

int main(int argc, char** argv) {
    std::string model_path = "model.bin", test_path, cls = "all", out = "out_cpp";
    int n = 4, steps = 0, threads = 0;
    float cfg = 0.0f;
    bool bench = false, prof = false, seeded = false, f32 = false;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") model_path = val();
        else if (a == "--test") test_path = val();
        else if (a == "--cls") cls = val();
        else if (a == "--n") n = std::atoi(val().c_str());
        else if (a == "--steps") steps = std::max(1, std::atoi(val().c_str()));
        else if (a == "--cfg") cfg = std::max(1e-3f, (float)std::atof(val().c_str()));
        else if (a == "--seed") { seed = std::strtoull(val().c_str(), nullptr, 10); seeded = true; }
        else if (a == "--threads") threads = std::max(1, std::atoi(val().c_str()));
        else if (a == "--out") out = val();
        else if (a == "--float") f32 = true;
        else if (a == "--bench") bench = true;
        else if (a == "--profile") bench = prof = true;
        else { usage(); return 2; }
    }
    try {
        auto t0 = Clock::now();
        d1d::Model m = d1d::Model::load(model_path, threads);
        std::printf("model %s loaded in %.2f s, %d threads, %s\n", model_path.c_str(), secs(t0),
                    m.threads(), D1D_AVX2 ? "AVX2" : "portable");
        if (steps == 0) steps = m.default_steps();
        if (cfg <= 0.0f) cfg = m.default_cfg();
        if (!test_path.empty()) return run_test(m, test_path);

        if (bench) {
            std::vector<float> x(d1d::kLength), v(d1d::kLength);
            d1d::Model::make_noise(1, x.data());
            m.forward(x.data(), 0.5f, 0, v.data());       // warm-up (fills the buffer pool)
            m.set_profiling(prof);
            const int reps = 10;
            t0 = Clock::now();
            for (int r = 0; r < reps; ++r) m.forward(x.data(), 0.5f, 0, v.data());
            const double ms = secs(t0) / reps * 1000.0;
            if (prof) {
                const char* names[4] = {"conv pad", "conv compute", "groupnorm", "attention"};
                const auto t = m.profile_seconds();
                double sum = 0;
                for (int k = 0; k < 4; ++k) {
                    std::printf("  %-13s %7.1f ms\n", names[k], t[k] / reps * 1000.0);
                    sum += t[k];
                }
                std::printf("  %-13s %7.1f ms\n", "rest", ms - sum / reps * 1000.0);
            }
            const int passes = steps * (cfg != 1.0f ? 2 : 1);
            std::printf("%.1f ms per net pass -> %.0f ms per sample (%d steps, cfg %.1f = %d passes)\n",
                        ms, ms * passes, steps, cfg, passes);
            return 0;
        }

        std::vector<d1d::Drum> classes;
        d1d::Drum d;
        if (cls == "all") classes = {d1d::Drum::Kick, d1d::Drum::Snare, d1d::Drum::Hat};
        else if (d1d::parse_drum(cls, d)) classes = {d};
        else { usage(); return 2; }
        if (!seeded) seed = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
        std::filesystem::create_directories(out);
        std::printf("seed %llu, %d steps, cfg %.1f, output gate %s\n", (unsigned long long)seed,
                    steps, cfg, m.gated() ? "on" : "off (model has none)");
        d1d::GenParams p;
        p.steps = steps;
        p.cfg = cfg;
        uint64_t k = 0;
        for (d1d::Drum c : classes) {
            for (int i = 0; i < n; ++i, ++k) {
                p.seed = seed + k;
                t0 = Clock::now();
                const std::vector<float> x = m.generate(c, p);
                char name[64];
                std::snprintf(name, sizeof name, "%s_%02d.wav", d1d::drum_name(c), i);
                d1d::save_wav((std::filesystem::path(out) / name).string(), x, d1d::kSampleRate, f32);
                std::printf("%s  seed %llu  %.0f ms\n", name, (unsigned long long)p.seed,
                            secs(t0) * 1000.0);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
