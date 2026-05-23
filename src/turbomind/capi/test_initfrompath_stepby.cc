//! Step-by-step InitFromPath test with detailed logging
//! This binary tests each step of InitFromPath separately to isolate the hang

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>

#include "include/turbomind_c.h"

// Log to file and stderr
static FILE* log_file = nullptr;

#define LOG(fmt, ...) \
    do { \
        auto now = std::chrono::system_clock::now(); \
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( \
            now.time_since_epoch()) % 1000; \
        time_t s = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count(); \
        struct tm* t = localtime(&s); \
        char timebuf[64]; \
        strftime(timebuf, sizeof(timebuf), "%H:%M:%S", t); \
        fprintf(stderr, "[%s.%03ld] " fmt "\n", timebuf, ms.count()); \
        fflush(stderr); \
        if (log_file) { \
            fprintf(log_file, "[%s.%03ld] " fmt "\n", timebuf, ms.count()); \
            fflush(log_file); \
        } \
    } while(0)

#define LOG_SECTION(name) \
    do { \
        LOG("=========================================="); \
        LOG("%s", name); \
        LOG("=========================================="); \
    } while(0)

int main(int argc, char* argv[]) {
    const char* model_dir = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ";
    if (argc > 1) {
        model_dir = argv[1];
    }

    // Open log file
    log_file = fopen("/tmp/initfrompath_stepby.log", "w");
    if (!log_file) {
        std::cerr << "Failed to open log file" << std::endl;
        return 1;
    }

    LOG_SECTION("InitFromPath Step-by-Step Test");

    LOG("Model directory: %s", model_dir);
    LOG("Process ID: %d", getpid());
    LOG("Thread ID: %lu", (unsigned long)std::this_thread::get_id());

    // Check if model directory exists
    {
        std::ifstream check(model_dir);
        if (!check.good()) {
            LOG("ERROR: Model directory does not exist: %s", model_dir);
            fclose(log_file);
            return 1;
        }
        LOG("Model directory exists: OK");
    }

    LOG_SECTION("Step 1: TM_EngineConfig_Create");

    TM_EngineConfig* config = TM_EngineConfig_Create();
    if (!config) {
        LOG("ERROR: TM_EngineConfig_Create returned NULL");
        fclose(log_file);
        return 1;
    }
    LOG("EngineConfig created: OK (ptr=%p)", (void*)config);

    LOG("Setting config parameters...");
    TM_EngineConfig_SetDataType(config, TM_DATATYPE_FP16);
    LOG("  data_type = FP16");
    TM_EngineConfig_SetSessionLen(config, 4096);
    LOG("  session_len = 4096");
    TM_EngineConfig_SetCacheBlockSeqLen(config, 64);
    TM_EngineConfig_SetCacheMaxBlockCount(config, 0.8f);
    TM_EngineConfig_SetMaxBatchSize(config, 32);
    TM_EngineConfig_SetMaxPrefillIters(config, 1);
    TM_EngineConfig_SetEnableMetrics(config, true);
    TM_EngineConfig_SetAttnTpSize(config, 1);
    TM_EngineConfig_SetAttnDpSize(config, 1);
    TM_EngineConfig_SetAttnCpSize(config, 1);
    TM_EngineConfig_SetMlpTpSize(config, 1);
    TM_EngineConfig_SetNNodes(config, 1);
    TM_EngineConfig_SetNodeRank(config, 0);
    TM_EngineConfig_AddDevice(config, 0);
    LOG("  All config parameters set");

    LOG_SECTION("Step 2: TM_TurboMind_Create");

    TM_TurboMind* tm = TM_TurboMind_Create(model_dir, config);
    if (!tm) {
        LOG("ERROR: TM_TurboMind_Create returned NULL");
        TM_Error* err = TM_GetLastError();
        if (err) {
            LOG("  Error code: %d", err->code);
            LOG("  Error message: %s", err->message);
        }
        TM_EngineConfig_Destroy(config);
        fclose(log_file);
        return 1;
    }
    LOG("TurboMind created: OK (ptr=%p)", (void*)tm);

    // Clean up config (TurboMind owns it internally)
    TM_EngineConfig_Destroy(config);
    LOG("Config destroyed (TurboMind owns it)");

    LOG_SECTION("Step 3: Individual InitFromPath Steps");

    // Check TM_TurboMind_IsDummyNode
    bool is_dummy = TM_TurboMind_IsDummyNode(tm);
    LOG("TM_TurboMind_IsDummyNode: %s", is_dummy ? "true" : "false");

    LOG("");
    LOG("==========================================");
    LOG("About to call TM_TurboMind_InitFromPath");
    LOG("==========================================");
    LOG("");

    // Test each step individually
    const int index = 0;
    const int device_id = 0;
    int trust_remote_code = 1;

    LOG("Calling TM_TurboMind_CreateContext...");
    TM_TurboMind_CreateContext(tm, index);
    LOG("  TM_TurboMind_CreateContext completed");

    LOG("Calling TM_TurboMind_CreateRoot...");
    TM_TurboMind_CreateRoot(tm, index);
    LOG("  TM_TurboMind_CreateRoot completed");

    LOG("");
    LOG("==========================================");
    LOG("Now calling TM_TurboMind_ProcessWeights...");
    LOG("==========================================");

    auto before_process = std::chrono::steady_clock::now();
    LOG("START: ProcessWeights at %ld", (long)std::time(nullptr));

    TM_TurboMind_ProcessWeights(tm, index);

    auto after_process = std::chrono::steady_clock::now();
    auto process_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        after_process - before_process).count();
    LOG("END: ProcessWeights completed in %ld ms", process_duration);

    LOG("");
    LOG("==========================================");
    LOG("Now calling TM_TurboMind_CreateEngine...");
    LOG("==========================================");

    auto before_engine = std::chrono::steady_clock::now();
    LOG("START: CreateEngine at %ld", (long)std::time(nullptr));

    TM_TurboMind_CreateEngine(tm, index);

    auto after_engine = std::chrono::steady_clock::now();
    auto engine_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        after_engine - before_engine).count();
    LOG("END: CreateEngine completed in %ld ms", engine_duration);

    LOG("");
    LOG_SECTION("All Steps Completed Successfully");
    LOG("");

    // Clean up
    TM_TurboMind_Destroy(tm);
    LOG("TurboMind destroyed");
    LOG("Test completed successfully");

    fclose(log_file);
    return 0;
}