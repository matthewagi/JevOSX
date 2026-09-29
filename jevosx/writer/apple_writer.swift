// JevOSX on-device writer: a small bridge to Apple's Foundation Models framework (macOS 26+, Apple Intelligence).
//
// JevOSX compiles this file once with the Command Line Tools:
//     xcrun --sdk macosx swiftc -O -parse-as-library -o ~/.jevosx/bin/jevosx-writer-<hash> apple_writer.swift
//
// Protocol (one request per process, JSON on stdin and stdout):
//     jevosx-writer --check   → {"available": true|false, "reason": "...", "framework": true|false, "os": "26.3"}
//     jevosx-writer < request → {"ok": true, "text": "..."} | {"ok": false, "error": "<code>", "message": "..."}
//     request = {"instructions": "...", "prompt": "...", "temperature": 0.7, "max_tokens": 800}
//
// It also compiles against SDKs without FoundationModels (older Command Line Tools); --check then reports
// "sdkMissing" instead of the install failing.

import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

let protocolVersion = 1

struct WriterRequest: Decodable {
    let instructions: String?
    let prompt: String
    let temperature: Double?
    let max_tokens: Int?
}

func emit(_ object: [String: Any]) {
    var payload = object
    payload["protocol"] = protocolVersion
    guard let data = try? JSONSerialization.data(withJSONObject: payload, options: []) else {
        return
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

func osVersion() -> String {
    let version = ProcessInfo.processInfo.operatingSystemVersion
    return "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
}

#if canImport(FoundationModels)
@available(macOS 26.0, *)
func unavailableReason() -> String? {
    switch SystemLanguageModel.default.availability {
    case .available:
        return nil
    case .unavailable(.appleIntelligenceNotEnabled):
        return "appleIntelligenceNotEnabled"
    case .unavailable(.deviceNotEligible):
        return "deviceNotEligible"
    case .unavailable(.modelNotReady):
        return "modelNotReady"
    case .unavailable(_):
        return "unknown"
    }
}

@available(macOS 26.0, *)
func errorCode(_ error: LanguageModelSession.GenerationError) -> String {
    switch error {
    case .guardrailViolation:
        return "guardrailViolation"
    case .exceededContextWindowSize:
        return "exceededContextWindowSize"
    case .assetsUnavailable:
        return "assetsUnavailable"
    default:
        return "generationError"
    }
}

@available(macOS 26.0, *)
func generate(_ request: WriterRequest) async -> [String: Any] {
    if let reason = unavailableReason() {
        return ["ok": false, "error": reason, "message": "Apple's on-device model is unavailable (\(reason))"]
    }
    let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [], instructions: request.instructions)
    var options = GenerationOptions()
    if let temperature = request.temperature {
        options.temperature = temperature
    }
    if let maxTokens = request.max_tokens {
        options.maximumResponseTokens = maxTokens
    }
    do {
        let response = try await session.respond(to: request.prompt, options: options)
        return ["ok": true, "text": response.content]
    } catch let error as LanguageModelSession.GenerationError {
        return ["ok": false, "error": errorCode(error), "message": error.localizedDescription]
    } catch {
        return ["ok": false, "error": "failed", "message": String(describing: error)]
    }
}
#endif

func check() -> [String: Any] {
    var result: [String: Any] = ["os": osVersion()]
    #if canImport(FoundationModels)
    result["framework"] = true
    if #available(macOS 26.0, *) {
        let reason = unavailableReason()
        result["available"] = reason == nil
        result["reason"] = reason ?? "available"
    } else {
        result["available"] = false
        result["reason"] = "osTooOld"
    }
    #else
    result["framework"] = false
    result["available"] = false
    result["reason"] = "sdkMissing"
    #endif
    return result
}

@main
struct JevOSXWriter {
    static func main() async {
        let arguments = CommandLine.arguments.dropFirst()
        if arguments.contains("--check") {
            emit(check())
            return
        }
        let input = FileHandle.standardInput.readDataToEndOfFile()
        guard let request = try? JSONDecoder().decode(WriterRequest.self, from: input) else {
            emit(["ok": false, "error": "badRequest", "message": "expected one JSON request on stdin"])
            exit(2)
        }
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) {
            emit(await generate(request))
        } else {
            emit(["ok": false, "error": "osTooOld", "message": "Apple's on-device model needs macOS 26 or newer"])
        }
        #else
        emit(["ok": false, "error": "sdkMissing", "message": "built with an SDK that has no FoundationModels framework"])
        #endif
    }
}
