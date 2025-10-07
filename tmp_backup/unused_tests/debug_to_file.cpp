#include <iostream>
#include <fstream>
#include <string>

// Debug output that writes to file instead of stdout
class DebugLogger {
private:
    std::ofstream log_file;

public:
    DebugLogger() {
        log_file.open("C:\\PacMapDotnet\\src\\PacMapDemo\\bin\\Debug\\net8.0-windows\\pacmap_debug.log", std::ios::app);
        if (log_file.is_open()) {
            log_file << "\n=== PACMAP DEBUG SESSION STARTED ===\n";
            log_file.flush();
        }
    }

    void log(const std::string& message) {
        if (log_file.is_open()) {
            log_file << message << "\n";
            log_file.flush();
        }
        // Also try console output
        std::cout << message << std::endl;
    }

    ~DebugLogger() {
        if (log_file.is_open()) {
            log_file << "=== DEBUG SESSION ENDED ===\n\n";
            log_file.close();
        }
    }
};

// Global debug logger
static DebugLogger debug_logger;

// Export debug function for use throughout PACMAP code
extern "C" void pacmap_debug_log(const char* message) {
    debug_logger.log(std::string(message));
}

extern "C" void pacmap_debug_log_int(const char* prefix, int value) {
    debug_logger.log(std::string(prefix) + std::to_string(value));
}

extern "C" void pacmap_debug_log_float(const char* prefix, float value) {
    debug_logger.log(std::string(prefix) + std::to_string(value));
}