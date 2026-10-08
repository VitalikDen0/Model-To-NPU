#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "BuildId.hpp"
#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnSampleApp.hpp"
#include "QnnSampleAppUtils.hpp"

static void* sg_backendHandle{nullptr};
static void* sg_modelHandle{nullptr};

namespace qnn {
namespace tools {
namespace sample_app {

struct ServerOptions {
  std::string backEndPath;
  std::string cachedBinaryPath;
  std::string configFilePath;
  std::string defaultOutputPath;
  std::string opPackagePaths;
  std::string requestFifoPath;
  std::string responseFifoPath;
  std::string systemLibraryPath;
  bool debug{false};
  bool nativeOutput{false};
  bool dumpOutputs{true};
  QnnLog_Level_t logLevel{QNN_LOG_LEVEL_ERROR};
  ProfilingLevel profilingLevel{ProfilingLevel::OFF};
};

struct GraphRuntimeConfig {
  std::vector<std::string> graphNames;
  bool hasVtcmSizeMb{false};
  uint32_t vtcmSizeMb{0};
  bool hasNumHvxThreads{false};
  uint64_t numHvxThreads{0};
};

static std::string trimLine(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

static bool ensureFifo(const std::string& path) {
  struct stat st {};
  if (0 == stat(path.c_str(), &st)) {
    if (S_ISFIFO(st.st_mode)) {
      return true;
    }
    std::cerr << "Path exists but is not a fifo: " << path << "\n";
    return false;
  }
  if (mkfifo(path.c_str(), 0666) == 0 || errno == EEXIST) {
    return true;
  }
  std::cerr << "mkfifo failed for " << path << ": " << strerror(errno) << "\n";
  return false;
}

static bool readTextFile(const std::string& path, std::string& content) {
  std::ifstream input(path.c_str(), std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    return false;
  }
  content.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return true;
}

static std::string dirnameOf(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0) {
    return path.substr(0, 1);
  }
  return path.substr(0, pos);
}

static bool isAbsolutePath(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  if (path[0] == '/' || path[0] == '\\') {
    return true;
  }
  return path.size() > 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':';
}

static std::string joinPath(const std::string& baseDir, const std::string& relativePath) {
  if (relativePath.empty() || isAbsolutePath(relativePath)) {
    return relativePath;
  }
  if (baseDir.empty() || baseDir == ".") {
    return relativePath;
  }
  const char sep = (baseDir.find('\\') != std::string::npos) ? '\\' : '/';
  if (baseDir.back() == '/' || baseDir.back() == '\\') {
    return baseDir + relativePath;
  }
  return baseDir + sep + relativePath;
}

static bool extractJsonString(const std::string& text,
                              const std::string& key,
                              std::string& value) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern) || match.size() < 2) {
    return false;
  }
  value = match[1].str();
  return true;
}

static bool extractJsonUint(const std::string& text,
                            const std::string& key,
                            uint64_t& value) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern) || match.size() < 2) {
    return false;
  }
  value = std::stoull(match[1].str());
  return true;
}

static bool extractGraphNames(const std::string& block, std::vector<std::string>& graphNames) {
  std::smatch listMatch;
  const std::regex listPattern("\\\"graph_names\\\"\\s*:\\s*\\[([^\\]]*)\\]");
  if (!std::regex_search(block, listMatch, listPattern) || listMatch.size() < 2) {
    return false;
  }
  const std::string listBody = listMatch[1].str();
  const std::regex namePattern("\\\"([^\\\"]+)\\\"");
  for (std::sregex_iterator it(listBody.begin(), listBody.end(), namePattern), end; it != end; ++it) {
    graphNames.push_back((*it)[1].str());
  }
  return !graphNames.empty();
}

static std::vector<std::string> extractObjectBlocks(const std::string& text, const std::string& arrayKey) {
  std::vector<std::string> blocks;
  const auto keyPos = text.find("\"" + arrayKey + "\"");
  if (keyPos == std::string::npos) {
    return blocks;
  }
  const auto arrayStart = text.find('[', keyPos);
  if (arrayStart == std::string::npos) {
    return blocks;
  }

  int arrayDepth = 0;
  bool inString = false;
  bool escape = false;
  size_t objectStart = std::string::npos;
  int objectDepth = 0;

  for (size_t i = arrayStart; i < text.size(); ++i) {
    const char ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch == '\\') {
      escape = true;
      continue;
    }
    if (ch == '"') {
      inString = !inString;
      continue;
    }
    if (inString) {
      continue;
    }
    if (ch == '[') {
      ++arrayDepth;
      continue;
    }
    if (ch == ']') {
      --arrayDepth;
      if (arrayDepth == 0) {
        break;
      }
      continue;
    }
    if (arrayDepth <= 0) {
      continue;
    }
    if (ch == '{') {
      if (objectDepth == 0) {
        objectStart = i;
      }
      ++objectDepth;
      continue;
    }
    if (ch == '}') {
      --objectDepth;
      if (objectDepth == 0 && objectStart != std::string::npos) {
        blocks.push_back(text.substr(objectStart, i - objectStart + 1));
        objectStart = std::string::npos;
      }
    }
  }
  return blocks;
}

static bool loadGraphRuntimeConfigs(const std::string& configFilePath,
                                    std::vector<GraphRuntimeConfig>& configs) {
  if (configFilePath.empty()) {
    return true;
  }

  std::string configText;
  if (!readTextFile(configFilePath, configText)) {
    std::cerr << "Failed to read config file: " << configFilePath << "\n";
    return false;
  }

  std::string graphConfigPath = configFilePath;
  std::string nestedPath;
  if (extractJsonString(configText, "config_file_path", nestedPath)) {
    graphConfigPath = joinPath(dirnameOf(configFilePath), nestedPath);
    if (!readTextFile(graphConfigPath, configText)) {
      std::cerr << "Failed to read backend extension graph config: " << graphConfigPath << "\n";
      return false;
    }
  }

  for (const auto& block : extractObjectBlocks(configText, "graphs")) {
    GraphRuntimeConfig config;
    extractGraphNames(block, config.graphNames);

    uint64_t value = 0;
    if (extractJsonUint(block, "vtcm_mb", value)) {
      config.hasVtcmSizeMb = true;
      config.vtcmSizeMb = static_cast<uint32_t>(value);
    }
    if (extractJsonUint(block, "hvx_threads", value)) {
      config.hasNumHvxThreads = true;
      config.numHvxThreads = value;
    }

    if (config.hasVtcmSizeMb || config.hasNumHvxThreads) {
      configs.push_back(config);
    }
  }

  if (configs.empty()) {
    std::cerr << "No graph runtime configs found in " << graphConfigPath << "\n";
  }
  return true;
}

static void showHelp() {
  std::cout
      << "\nPersistent QNN context runner\n"
      << "\nRequired arguments:\n"
      << "  --retrieve_context <FILE>  Path to serialized context binary\n"
      << "  --backend <FILE>           Path to backend library (libQnnHtp.so)\n"
      << "  --system_library <FILE>    Path to libQnnSystem.so\n"
      << "  --request_fifo <FILE>      FIFO that receives input_list + output_dir\n"
      << "  --response_fifo <FILE>     FIFO that receives OK / ERR responses\n"
      << "\nOptional arguments:\n"
      << "  --config_file <FILE>       Optional qnn-net-run style HTP graph config JSON\n"
      << "  --output_dir <DIR>         Default output directory\n"
      << "  --op_packages <VAL>        Optional op packages list\n"
      << "  --debug                    Dump all layer outputs\n"
      << "  --use_native_output_files  Save native output files\n"
      << "  --profiling_level <VAL>    basic | detailed\n"
      << "  --log_level <VAL>          error | warn | info | verbose"
#ifdef QNN_ENABLE_DEBUG
      << " | debug"
#endif
      << "\n"
      << "  --help                     Show this help\n"
      << std::endl;
}

static bool parseArgs(int argc, char** argv, ServerOptions& opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto requireValue = [&](const char* name, std::string& target) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return false;
      }
      target = argv[++i];
      return true;
    };

    if (arg == "--help") {
      showHelp();
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--retrieve_context") {
      if (!requireValue("--retrieve_context", opts.cachedBinaryPath)) return false;
    } else if (arg == "--backend") {
      if (!requireValue("--backend", opts.backEndPath)) return false;
    } else if (arg == "--system_library") {
      if (!requireValue("--system_library", opts.systemLibraryPath)) return false;
    } else if (arg == "--config_file") {
      if (!requireValue("--config_file", opts.configFilePath)) return false;
    } else if (arg == "--request_fifo") {
      if (!requireValue("--request_fifo", opts.requestFifoPath)) return false;
    } else if (arg == "--response_fifo") {
      if (!requireValue("--response_fifo", opts.responseFifoPath)) return false;
    } else if (arg == "--output_dir") {
      if (!requireValue("--output_dir", opts.defaultOutputPath)) return false;
    } else if (arg == "--op_packages") {
      if (!requireValue("--op_packages", opts.opPackagePaths)) return false;
    } else if (arg == "--debug") {
      opts.debug = true;
    } else if (arg == "--use_native_output_files") {
      opts.nativeOutput = true;
    } else if (arg == "--profiling_level") {
      std::string level;
      if (!requireValue("--profiling_level", level)) return false;
      opts.profilingLevel = parseProfilingLevel(level);
      if (opts.profilingLevel == ProfilingLevel::INVALID) {
        std::cerr << "Invalid profiling level: " << level << "\n";
        return false;
      }
    } else if (arg == "--log_level") {
      std::string level;
      if (!requireValue("--log_level", level)) return false;
      opts.logLevel = parseLogLevel(level);
      if (opts.logLevel == QNN_LOG_LEVEL_MAX) {
        std::cerr << "Invalid log level: " << level << "\n";
        return false;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (opts.cachedBinaryPath.empty() || opts.backEndPath.empty() || opts.systemLibraryPath.empty() ||
      opts.requestFifoPath.empty() || opts.responseFifoPath.empty()) {
    std::cerr << "Missing required arguments.\n";
    return false;
  }
  return true;
}

static bool readRequest(const std::string& requestFifo,
                        std::string& inputListPath,
                        std::string& outputDir) {
  std::ifstream req(requestFifo.c_str());
  if (!req.is_open()) {
    return false;
  }
  if (!std::getline(req, inputListPath)) {
    return false;
  }
  if (!std::getline(req, outputDir)) {
    return false;
  }
  inputListPath = trimLine(inputListPath);
  outputDir     = trimLine(outputDir);
  return true;
}

static void writeResponse(const std::string& responseFifo, const std::string& message) {
  std::ofstream rsp(responseFifo.c_str());
  if (!rsp.is_open()) {
    std::cerr << "Failed to open response fifo: " << responseFifo << "\n";
    return;
  }
  rsp << message << "\n";
  rsp.flush();
}

}  // namespace sample_app
}  // namespace tools
}  // namespace qnn

int main(int argc, char** argv) {
  using namespace qnn::tools;
  using namespace qnn::tools::sample_app;

  if (!qnn::log::initializeLogging()) {
    std::cerr << "ERROR: Unable to initialize logging\n";
    return EXIT_FAILURE;
  }

  ServerOptions opts;
  if (!parseArgs(argc, argv, opts)) {
    showHelp();
    return EXIT_FAILURE;
  }
  if (!qnn::log::setLogLevel(opts.logLevel)) {
    std::cerr << "ERROR: Unable to set log level\n";
    return EXIT_FAILURE;
  }

  if (!ensureFifo(opts.requestFifoPath) || !ensureFifo(opts.responseFifoPath)) {
    return EXIT_FAILURE;
  }

  QnnFunctionPointers qnnFunctionPointers;
  auto statusCode = dynamicloadutil::getQnnFunctionPointers(opts.backEndPath,
                                                            "",
                                                            &qnnFunctionPointers,
                                                            &sg_backendHandle,
                                                            false,
                                                            &sg_modelHandle);
  if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
    std::cerr << "ERROR: Unable to initialize QNN backend function pointers\n";
    return EXIT_FAILURE;
  }
  statusCode = dynamicloadutil::getQnnSystemFunctionPointers(opts.systemLibraryPath,
                                                             &qnnFunctionPointers);
  if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
    std::cerr << "ERROR: Unable to initialize QNN system function pointers\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<QnnSampleApp> app(new QnnSampleApp(qnnFunctionPointers,
                                                     "",
                                                     opts.opPackagePaths,
                                                     sg_backendHandle,
                                                     opts.defaultOutputPath,
                                                     opts.debug,
                                                     opts.nativeOutput ? iotensor::OutputDataType::NATIVE_ONLY
                                                                       : iotensor::OutputDataType::FLOAT_ONLY,
                                                     iotensor::InputDataType::FLOAT,
                                                     opts.profilingLevel,
                                                     opts.dumpOutputs,
                                                     opts.cachedBinaryPath,
                                                     "",
                                                     1));

  QNN_INFO("qnn-context-runner build version: %s", qnn::tools::getBuildId().c_str());

  std::vector<GraphRuntimeConfig> graphRuntimeConfigs;
  if (!loadGraphRuntimeConfigs(opts.configFilePath, graphRuntimeConfigs)) {
    return EXIT_FAILURE;
  }

  if (StatusCode::SUCCESS != app->initialize()) {
    return app->reportError("Initialization failure");
  }
  if (StatusCode::SUCCESS != app->initializeBackend()) {
    return app->reportError("Backend initialization failure");
  }

  auto devicePropertySupportStatus = app->isDevicePropertySupported();
  if (StatusCode::FAILURE != devicePropertySupportStatus) {
    auto createDeviceStatus = app->createDevice();
    if (StatusCode::SUCCESS != createDeviceStatus) {
      return app->reportError("Device creation failure");
    }
    auto perfStatus = app->enableHighPerformanceMode();
    if (StatusCode::SUCCESS == perfStatus) {
      QNN_INFO("HTP performance mode enabled.");
    } else {
      QNN_WARN("HTP performance mode configuration unavailable; continuing with backend defaults.");
    }
  }

  if (StatusCode::SUCCESS != app->initializeProfiling()) {
    return app->reportError("Profiling initialization failure");
  }
  if (StatusCode::SUCCESS != app->registerOpPackages()) {
    return app->reportError("Register Op Packages failure");
  }
  if (StatusCode::SUCCESS != app->createFromBinary()) {
    return app->reportError("Create From Binary failure");
  }
  for (const auto& graphConfig : graphRuntimeConfigs) {
    auto graphConfigStatus = app->setGraphHtpConfig(graphConfig.graphNames,
                                                    graphConfig.hasVtcmSizeMb,
                                                    graphConfig.vtcmSizeMb,
                                                    graphConfig.hasNumHvxThreads,
                                                    graphConfig.numHvxThreads);
    if (StatusCode::SUCCESS != graphConfigStatus &&
        StatusCode::QNN_FEATURE_UNSUPPORTED != graphConfigStatus) {
      return app->reportError("Apply HTP graph config failure");
    }
    if (StatusCode::QNN_FEATURE_UNSUPPORTED == graphConfigStatus) {
      QNN_WARN("Graph runtime config requested but graphSetConfig is unsupported.");
    }
  }

  QNN_INFO("Persistent context loaded: %s", opts.cachedBinaryPath.c_str());
  QNN_INFO("Listening on request fifo: %s", opts.requestFifoPath.c_str());

  while (true) {
    std::string inputListPath;
    std::string outputDir;
    if (!readRequest(opts.requestFifoPath, inputListPath, outputDir)) {
      continue;
    }
    if (inputListPath == "__quit__") {
      writeResponse(opts.responseFifoPath, "OK");
      break;
    }

    const std::string effectiveOutputDir = outputDir.empty() ? opts.defaultOutputPath : outputDir;
    QNN_INFO("Request: %s -> %s", inputListPath.c_str(), effectiveOutputDir.c_str());

    if (StatusCode::SUCCESS != app->setInputOutputPaths(inputListPath, effectiveOutputDir)) {
      writeResponse(opts.responseFifoPath, "ERR:failed to update input/output paths");
      continue;
    }
    if (StatusCode::SUCCESS != app->executeGraphs()) {
      writeResponse(opts.responseFifoPath, "ERR:graph execution failed");
      continue;
    }
    writeResponse(opts.responseFifoPath, "OK");
  }

  if (StatusCode::SUCCESS != app->freeContext()) {
    app->reportError("Context free failure");
  }
  if (StatusCode::FAILURE != devicePropertySupportStatus) {
    auto freeDeviceStatus = app->freeDevice();
    if (StatusCode::SUCCESS != freeDeviceStatus) {
      app->reportError("Device free failure");
    }
  }

  if (sg_backendHandle) {
    pal::dynamicloading::dlClose(sg_backendHandle);
  }
  if (sg_modelHandle) {
    pal::dynamicloading::dlClose(sg_modelHandle);
  }
  return EXIT_SUCCESS;
}
