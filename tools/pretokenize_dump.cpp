// pretokenize_dump: a thin pipe around toby::tokenize::pretokenize, so a script
// can drive the real implementation without linking against C++.
//
// Reads one hex-encoded input per line on stdin. Writes one line per input: the
// resulting chunks, hex-encoded, comma-separated. An input that throws produces
// the single token THREW.
//
// Hex on both sides deliberately: the interesting inputs are full of tabs,
// newlines and quotes, and any text encoding of those needs escaping rules that
// both ends have to agree on. Hex has none.
//
//   $ printf 'youre' | xxd -p | ./pretokenize_dump
//
// Used by tools/pretokenizer_diff.py. Also handy on its own for "what does it do
// with this exact string?".

#include "toby/tokenize/pretokenizer.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <format>
#include <iostream>
#include <print>
#include <string>
#include <string_view>

namespace {

// Hand-rolled rather than std::from_chars/strtol: both want a pointer pair, and
// indexing the view keeps this free of pointer arithmetic.
constexpr int hex_digit(const char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

// Returns false on malformed input rather than throwing: a decode failure is a
// bug in the caller, and we want to say so plainly instead of reporting THREW
// and looking like a pretokenizer failure.
bool unhex(std::string_view hex, std::string& out) {
    if (hex.size() % 2 != 0) {
        return false;
    }

    out.clear();
    out.reserve(hex.size() / 2);

    for (std::size_t i = 0; i < hex.size(); i += 2) {
        const int high = hex_digit(hex[i]);
        const int low = hex_digit(hex[i + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        out.push_back(static_cast<char>((high * 16) + low));
    }

    return true;
}

std::string tohex(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const char byte : bytes) {
        // Via unsigned char, not char: char is signed here, so 0xC3 would format
        // as ffffffc3 and the hex would no longer round-trip.
        out += std::format("{:02x}", static_cast<unsigned char>(byte));
    }
    return out;
}

// Split out from main so main itself has a single try/catch and cannot let an
// exception escape -- getline, string growth and println can all throw, and an
// exception leaving main is UB-adjacent (std::terminate, no unwinding).
int run() {
    std::string line;
    std::string input;

    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!unhex(line, input)) {
            std::println(std::cerr, "pretokenize_dump: not a hex string: {}", line);
            return 2;
        }

        try {
            const auto chunks = toby::tokenize::pretokenize(input);
            std::string out;
            for (const auto& chunk : chunks) {
                if (!out.empty()) {
                    out += ',';
                }
                out += tohex(chunk);
            }
            std::println("{}", out);
        } catch (const std::exception&) {
            std::println("THREW");
        }

        // The driver writes every input up front and then reads; without this it
        // blocks on our buffered output while we block on its closed-but-unread
        // pipe. Cheap at this scale.
        std::cout.flush();
    }

    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        // fputs, not println: the handler itself must not throw.
        std::fputs("pretokenize_dump: ", stderr);
        std::fputs(e.what(), stderr);
        std::fputs("\n", stderr);
        return 1;
    } catch (...) {
        std::fputs("pretokenize_dump: unknown exception\n", stderr);
        return 1;
    }
}
