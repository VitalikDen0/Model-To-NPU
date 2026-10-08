#!/usr/bin/env python3
"""Build Android persistent QNN context runner from the locally installed QAIRT SampleApp.

This script does NOT commit or copy Qualcomm source into the repository permanently.
It stages a temporary copy of QAIRT's SampleApp, patches the copy to keep graphs alive
between requests, swaps in our custom persistent main(), and builds an Android executable
via ndk-build.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NDK_ROOT = Path(r"C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358")
SAMPLE_APP_ROOT_REL = Path("examples") / "QNN" / "SampleApp" / "SampleApp"
RUNNER_MAIN = Path(__file__).with_name("qnn_context_runner_main.cpp")
OUT_DIR = Path(__file__).with_name("out") / "arm64-v8a"


def detect_default_qairt_root() -> Path:
    candidates = [
        Path(r"D:\platform-tools\sdxl_npu\qairt_2.44\qairt\2.44.0.260225"),
        Path(r"C:\Qualcomm\AIStack\QAIRT\2.44.0.260225"),
        Path(r"C:\Qualcomm\AIStack\QAIRT\2.31.0.250130"),
    ]
    for candidate in candidates:
        if (candidate / SAMPLE_APP_ROOT_REL).exists():
            return candidate
    return candidates[-1]


DEFAULT_QAIRT_ROOT = detect_default_qairt_root()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Patch anchor not found for {label}")
    return text.replace(old, new, 1)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def patch_sample_app(staged_root: Path, qairt_root: Path) -> None:
    android_mk = staged_root / "make" / "Android.mk"
    qnn_hpp = staged_root / "src" / "QnnSampleApp.hpp"
    qnn_cpp = staged_root / "src" / "QnnSampleApp.cpp"
    main_cpp = staged_root / "src" / "main.cpp"

    android_mk_text = _read_text(android_mk)
    android_mk_text = _replace_once(
        android_mk_text,
        "PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../../../../../include/QNN",
        f"PACKAGE_C_INCLUDES += -I {qairt_root / 'include' / 'QNN'}",
        "Android.mk QNN include path",
    )
    android_mk_text = _replace_once(
        android_mk_text,
        "PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../include/flatbuffers",
        f"PACKAGE_C_INCLUDES += -I {qairt_root / 'examples' / 'QNN' / 'SampleApp' / 'SampleApp' / 'include' / 'flatbuffers'}",
        "Android.mk flatbuffers include path",
    )
    android_mk_text = _replace_once(
        android_mk_text,
        "LOCAL_MODULE                   := qnn-sample-app",
        "LOCAL_MODULE                   := qnn-context-runner",
        "Android module rename",
    )
    _write_text(android_mk, android_mk_text)

    qnn_hpp_text = _read_text(qnn_hpp)
    qnn_hpp_text = _replace_once(
        qnn_hpp_text,
        "  StatusCode freeDevice();\n\n  StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);",
        "  StatusCode freeDevice();\n\n  StatusCode setInputOutputPaths(std::string inputListPaths, std::string outputPath);\n\n  StatusCode enableHighPerformanceMode();\n\n  StatusCode setGraphHtpConfig(const std::vector<std::string>& graphNames,\n                               bool hasVtcmSizeMb,\n                               uint32_t vtcmSizeMb,\n                               bool hasNumHvxThreads,\n                               uint64_t numHvxThreads);\n\n  StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);",
        "QnnSampleApp public API extension",
    )
    _write_text(qnn_hpp, qnn_hpp_text)

    qnn_cpp_text = _read_text(qnn_cpp)
    qnn_cpp_text = _replace_once(
        qnn_cpp_text,
        '#include "Logger.hpp"\n',
        '#include "Logger.hpp"\n#include "QnnGraph.h"\n#include "QnnProperty.h"\n#include "HTP/QnnHtpDevice.h"\n#include "HTP/QnnHtpGraph.h"\n',
        "QnnSampleApp include extension",
    )
    qnn_cpp_text = _replace_once(
        qnn_cpp_text,
        "  // Read Input File List\n  bool readSuccess;\n  std::tie(m_inputFileLists, m_inputNameToIndex, readSuccess) = readInputLists(m_inputListPaths);\n  if (!readSuccess) {\n    exitWithMessage(\"Could not read input lists\", EXIT_FAILURE);\n  }\n",
        "  // Read Input File List\n  if (!m_inputListPaths.empty()) {\n    bool readSuccess;\n    std::tie(m_inputFileLists, m_inputNameToIndex, readSuccess) = readInputLists(m_inputListPaths);\n    if (!readSuccess) {\n      exitWithMessage(\"Could not read input lists\", EXIT_FAILURE);\n    }\n  }\n",
        "initialize() optional input list",
    )
    qnn_cpp_text = _replace_once(
        qnn_cpp_text,
        "sample_app::StatusCode sample_app::QnnSampleApp::initializeProfiling() {",
        "sample_app::StatusCode sample_app::QnnSampleApp::setInputOutputPaths(std::string inputListPaths,\n                                                                  std::string outputPath) {\n  std::vector<std::string> parsedInputListPaths;\n  if (!inputListPaths.empty()) {\n    split(parsedInputListPaths, inputListPaths, ',');\n  }\n\n  std::vector<std::vector<std::vector<std::string>>> parsedInputFileLists;\n  std::vector<std::unordered_map<std::string, uint32_t>> parsedInputNameToIndex;\n  bool readSuccess = true;\n  if (!parsedInputListPaths.empty()) {\n    std::tie(parsedInputFileLists, parsedInputNameToIndex, readSuccess) = readInputLists(parsedInputListPaths);\n    if (!readSuccess) {\n      QNN_ERROR(\"Could not read input lists\");\n      return StatusCode::FAILURE;\n    }\n  }\n\n  m_inputListPaths = parsedInputListPaths;\n  m_inputFileLists = parsedInputFileLists;\n  m_inputNameToIndex = parsedInputNameToIndex;\n  if (!outputPath.empty()) {\n    m_outputPath = outputPath;\n  } else if (m_outputPath.empty()) {\n    m_outputPath = s_defaultOutputPath;\n  }\n#ifndef __hexagon__\n  if (m_dumpOutputs && !::pal::FileOp::checkFileExists(m_outputPath) &&\n      !pal::Directory::makePath(m_outputPath)) {\n    QNN_ERROR(\"Could not create output directory: %s\", m_outputPath.c_str());\n    return StatusCode::FAILURE;\n  }\n#endif\n  return StatusCode::SUCCESS;\n}\n\nsample_app::StatusCode sample_app::QnnSampleApp::enableHighPerformanceMode() {\n  if (nullptr == m_qnnFunctionPointers.qnnInterface.propertyHasCapability ||\n      nullptr == m_qnnFunctionPointers.qnnInterface.deviceGetInfrastructure) {\n    return StatusCode::QNN_FEATURE_UNSUPPORTED;\n  }\n  auto propertyStatus =\n      m_qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_DEVICE_SUPPORT_INFRASTRUCTURE);\n  if (QNN_PROPERTY_SUPPORTED != propertyStatus && QNN_SUCCESS != propertyStatus) {\n    return StatusCode::QNN_FEATURE_UNSUPPORTED;\n  }\n\n  QnnDevice_Infrastructure_t deviceInfraOpaque = nullptr;\n  auto infraStatus = m_qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfraOpaque);\n  if (QNN_SUCCESS != infraStatus || nullptr == deviceInfraOpaque) {\n    return StatusCode::FAILURE;\n  }\n\n  const auto* htpInfra = reinterpret_cast<const QnnHtpDevice_Infrastructure_t*>(deviceInfraOpaque);\n  if (htpInfra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF ||\n      nullptr == htpInfra->perfInfra.setPowerConfig) {\n    return StatusCode::QNN_FEATURE_UNSUPPORTED;\n  }\n\n  QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};\n  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;\n  dcvs.dcvsV3Config.contextId = 0;\n  dcvs.dcvsV3Config.setDcvsEnable = 1;\n  dcvs.dcvsV3Config.dcvsEnable = 1;\n  dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;\n  dcvs.dcvsV3Config.setSleepDisable = 1;\n  dcvs.dcvsV3Config.sleepDisable = 1;\n  dcvs.dcvsV3Config.setBusParams = 1;\n  dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;\n  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;\n  dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;\n  dcvs.dcvsV3Config.setCoreParams = 1;\n  dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;\n  dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;\n  dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;\n\n  const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, nullptr};\n  auto setStatus = htpInfra->perfInfra.setPowerConfig(0, configs);\n  if (QNN_SUCCESS != setStatus) {\n    QNN_WARN(\"Failed to enable HTP performance mode: %d\", setStatus);\n    return StatusCode::FAILURE;\n  }\n  return StatusCode::SUCCESS;\n}\n\nsample_app::StatusCode sample_app::QnnSampleApp::setGraphHtpConfig(\n    const std::vector<std::string>& graphNames,\n    bool hasVtcmSizeMb,\n    uint32_t vtcmSizeMb,\n    bool hasNumHvxThreads,\n    uint64_t numHvxThreads) {\n  if ((!hasVtcmSizeMb && !hasNumHvxThreads) || nullptr == m_graphsInfo || 0 == m_graphsCount) {\n    return StatusCode::SUCCESS;\n  }\n  if (nullptr == m_qnnFunctionPointers.qnnInterface.graphSetConfig) {\n    return StatusCode::QNN_FEATURE_UNSUPPORTED;\n  }\n\n  bool applied = false;\n  for (size_t graphIdx = 0; graphIdx < m_graphsCount; ++graphIdx) {\n    auto& graphInfo = (*m_graphsInfo)[graphIdx];\n    const std::string graphName = graphInfo.graphName ? graphInfo.graphName : \"\";\n\n    bool matches = graphNames.empty();\n    if (!matches) {\n      for (const auto& candidate : graphNames) {\n        if (candidate == graphName) {\n          matches = true;\n          break;\n        }\n      }\n    }\n    if (!matches) {\n      continue;\n    }\n\n    QnnHtpGraph_CustomConfig_t vtcmConfig = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;\n    QnnHtpGraph_CustomConfig_t hvxConfig = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;\n    QnnGraph_Config_t graphConfigs[3]{};\n    const QnnGraph_Config_t* graphConfigPtrs[3] = {nullptr, nullptr, nullptr};\n    size_t configCount = 0;\n\n    if (hasVtcmSizeMb) {\n      vtcmConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB;\n      vtcmConfig.vtcmSizeInMB = vtcmSizeMb;\n      graphConfigs[configCount].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;\n      graphConfigs[configCount].customConfig = reinterpret_cast<QnnGraph_CustomConfig_t>(&vtcmConfig);\n      graphConfigPtrs[configCount] = &graphConfigs[configCount];\n      ++configCount;\n    }\n    if (hasNumHvxThreads) {\n      hvxConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;\n      hvxConfig.numHvxThreads = numHvxThreads;\n      graphConfigs[configCount].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;\n      graphConfigs[configCount].customConfig = reinterpret_cast<QnnGraph_CustomConfig_t>(&hvxConfig);\n      graphConfigPtrs[configCount] = &graphConfigs[configCount];\n      ++configCount;\n    }\n\n    auto err = m_qnnFunctionPointers.qnnInterface.graphSetConfig(graphInfo.graph, graphConfigPtrs);\n    if (QNN_SUCCESS != err) {\n      QNN_ERROR(\"Failed to set HTP graph config for %s: %d\", graphName.c_str(), err);\n      return StatusCode::FAILURE;\n    }\n    applied = true;\n    QNN_INFO(\"Applied HTP graph config to %s (vtcm_mb=%u, hvx_threads=%llu)\",\n             graphName.c_str(),\n             hasVtcmSizeMb ? vtcmSizeMb : 0u,\n             static_cast<unsigned long long>(hasNumHvxThreads ? numHvxThreads : 0ull));\n  }\n\n  if (!applied) {\n    QNN_WARN(\"No graphs matched requested HTP graph config\");\n  }\n  return StatusCode::SUCCESS;\n}\n\nsample_app::StatusCode sample_app::QnnSampleApp::initializeProfiling() {",
        "insert runtime setters and perf helper",
    )
    qnn_cpp_text = _replace_once(
        qnn_cpp_text,
        "\n  qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);\n  m_graphsInfo = nullptr;\n  return returnStatus;\n}",
        "\n  return returnStatus;\n}",
        "preserve graphs for persistent execution",
    )
    _write_text(qnn_cpp, qnn_cpp_text)

    shutil.copy2(RUNNER_MAIN, main_cpp)


def find_ndk_build(ndk_root: Path) -> Path:
    candidates = [
        ndk_root / "ndk-build.cmd",
        ndk_root / "ndk-build",
        ndk_root / "build" / "ndk-build.cmd",
        ndk_root / "build" / "ndk-build",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to find ndk-build under {ndk_root}")


def run_build(staged_root: Path, ndk_build: Path) -> Path:
    cmd = [
        str(ndk_build),
        "NDK_PROJECT_PATH=.",
        "APP_BUILD_SCRIPT=make/Android.mk",
        "NDK_APPLICATION_MK=make/Application.mk",
        "-j8",
    ]
    print("[build]", " ".join(cmd))
    subprocess.run(cmd, cwd=staged_root, check=True)
    output = staged_root / "libs" / "arm64-v8a" / "qnn-context-runner"
    if not output.exists():
        raise FileNotFoundError(f"Build finished but binary not found: {output}")
    return output


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build persistent QNN context runner for Android")
    ap.add_argument("--qairt-root", type=Path, default=DEFAULT_QAIRT_ROOT)
    ap.add_argument("--ndk-root", type=Path, default=DEFAULT_NDK_ROOT)
    ap.add_argument("--keep-temp", action="store_true", help="Keep patched temp SampleApp tree")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    sample_root = args.qairt_root / SAMPLE_APP_ROOT_REL
    if not sample_root.exists():
        print(f"ERROR: QAIRT SampleApp not found: {sample_root}", file=sys.stderr)
        return 1
    if not RUNNER_MAIN.exists():
        print(f"ERROR: runner main file not found: {RUNNER_MAIN}", file=sys.stderr)
        return 1

    ndk_build = find_ndk_build(args.ndk_root)
    print(f"[info] QAIRT root: {args.qairt_root}")
    print(f"[info] SampleApp : {sample_root}")
    print(f"[info] NDK build : {ndk_build}")

    temp_root_obj = None
    if args.keep_temp:
        staged_root = REPO_ROOT / "NPU" / "build" / "context_runner_sampleapp"
        if staged_root.exists():
            shutil.rmtree(staged_root)
        staged_root.parent.mkdir(parents=True, exist_ok=True)
    else:
        temp_root_obj = tempfile.TemporaryDirectory(prefix="qnn_context_runner_")
        staged_root = Path(temp_root_obj.name) / "SampleApp"

    print(f"[info] Staging to: {staged_root}")
    shutil.copytree(sample_root, staged_root)
    patch_sample_app(staged_root, args.qairt_root)

    built_binary = run_build(staged_root, ndk_build)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_binary = args.out_dir / built_binary.name
    shutil.copy2(built_binary, final_binary)
    print(f"[ok] Built binary: {final_binary}")

    if temp_root_obj is not None:
        temp_root_obj.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
