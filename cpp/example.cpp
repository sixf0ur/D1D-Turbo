// example.cpp -- minimal use of the header-only library d1d.hpp.
//
//   cl /O2 /arch:AVX2 /std:c++17 /EHsc example.cpp          (MSVC)
//   g++ -O3 -march=native -std=c++17 -pthread example.cpp     (GCC / Clang)
//   example model.bin
#include "d1d.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "model.bin";
    try {
        // 1) load from a file ...
        d1d::Model model = d1d::Model::load(path);

        // ... or from bytes in memory (an embedded array, a download, an asset pack).
        std::ifstream f(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        d1d::Model from_memory = d1d::Model::load(bytes.data(), bytes.size(), /*threads=*/0);

        std::printf("model: %d steps, cfg %.1f by default, %s, %d threads\n", model.default_steps(),
                    model.default_cfg(), model.gated() ? "D1D-Turbo (gated)" : "D1D", model.threads());

        // 2) generation parameters: the seed picks the sound, the rest has good defaults
        d1d::GenParams p;
        p.seed = 42;
        p.progress = [](int done, int total) {        // optional; return false to cancel
            std::printf("\r  step %d/%d", done, total);
            std::fflush(stdout);
            return true;
        };

        // 3) generate: 16384 floats at 32 kHz, peak-normalised to 0.95
        std::vector<float> kick = model.generate(d1d::Drum::Kick, p);
        std::printf("\n");
        d1d::save_wav("kick_42.wav", kick);

        // 4) or keep the WAV in memory (e.g. hand it to an audio engine / send it)
        p.progress = nullptr;
        p.seed = 7;
        std::vector<float> snare = from_memory.generate(d1d::Drum::Snare, p);
        std::vector<uint8_t> wav = d1d::wav_bytes(snare);        // complete .wav file image
        std::printf("snare: %zu samples -> %zu WAV bytes\n", snare.size(), wav.size());

        // 5) the same seed gives the same sound again
        p.seed = 42;
        std::vector<float> again = model.generate(d1d::Drum::Kick, p);
        std::printf("deterministic: %s\n", again == kick ? "yes" : "no");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
