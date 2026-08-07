import AVFoundation
import CoreML
import Darwin
import Dispatch
import FluidAudio
import Foundation

private let modelID = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
private let modelFolder = "parakeet-tdt-0.6b-v3"

private struct TranscriptionOutput: Encodable {
    let transcript: String
    let load_s: Double
    let infer_s: Double
    let audio_s: Double
    let rtf: Double
    let model_id: String
    let compute_units: String
}

private struct SelfcheckOutput: Encodable {
    let load_s: Double
    let model_id: String
    let compute_units: String
}

private struct ErrorOutput: Encodable {
    let error: String
}

private enum Command {
    case transcribe(audio: URL, modelDirectory: URL?)
    case selfcheck(modelDirectory: URL?)
}

private enum CLIError: LocalizedError {
    case usage(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message):
            return message
        }
    }
}

@main
private struct ParakeetAneCLI {
    static func main() async {
        do {
            let command = try parseCommand(Array(CommandLine.arguments.dropFirst()))
            switch command {
            case .transcribe(let audioURL, let modelDirectory):
                try await transcribe(audioURL: audioURL, modelDirectory: modelDirectory)
            case .selfcheck(let modelDirectory):
                try await selfcheck(modelDirectory: modelDirectory)
            }
        } catch {
            writeDiagnostic(error.localizedDescription)
            writeJSON(ErrorOutput(error: error.localizedDescription))
            exit(EXIT_FAILURE)
        }
    }
}

private func parseCommand(_ arguments: [String]) throws -> Command {
    guard let subcommand = arguments.first else {
        throw CLIError.usage("expected subcommand: transcribe or selfcheck")
    }

    var audioPath: String?
    var modelPath: String?
    var index = 1
    while index < arguments.count {
        switch arguments[index] {
        case "--audio":
            index += 1
            guard index < arguments.count else {
                throw CLIError.usage("--audio requires a path")
            }
            audioPath = arguments[index]
        case "--model-dir":
            index += 1
            guard index < arguments.count else {
                throw CLIError.usage("--model-dir requires a path")
            }
            modelPath = arguments[index]
        case "--json":
            break
        default:
            throw CLIError.usage("unknown argument: \(arguments[index])")
        }
        index += 1
    }

    let modelDirectory = modelPath.map {
        URL(fileURLWithPath: NSString(string: $0).expandingTildeInPath, isDirectory: true)
            .appendingPathComponent(modelFolder, isDirectory: true)
    }

    switch subcommand {
    case "transcribe":
        guard let audioPath else {
            throw CLIError.usage("transcribe requires --audio <wav-path>")
        }
        return .transcribe(
            audio: URL(fileURLWithPath: audioPath),
            modelDirectory: modelDirectory
        )
    case "selfcheck":
        guard audioPath == nil else {
            throw CLIError.usage("selfcheck does not accept --audio")
        }
        return .selfcheck(modelDirectory: modelDirectory)
    default:
        throw CLIError.usage("unknown subcommand: \(subcommand)")
    }
}

private func loadManager(modelDirectory: URL?) async throws -> (AsrModels, AsrManager, Double) {
    let configuration = AsrModels.defaultConfiguration()
    let started = DispatchTime.now().uptimeNanoseconds
    let models = try await AsrModels.downloadAndLoad(
        to: modelDirectory,
        configuration: configuration,
        version: .v3
    )
    let manager = AsrManager()
    try await manager.loadModels(models)
    return (models, manager, elapsedSeconds(since: started))
}

private func transcribe(audioURL: URL, modelDirectory: URL?) async throws {
    let audioFile = try AVAudioFile(forReading: audioURL)
    guard audioFile.processingFormat.channelCount == 1 else {
        throw CLIError.usage("input WAV must be mono")
    }
    let audioSeconds = Double(audioFile.length) / audioFile.processingFormat.sampleRate
    let samples = try AudioConverter().resampleAudioFile(audioURL)

    let (models, manager, loadSeconds) = try await loadManager(modelDirectory: modelDirectory)
    var decoderState = try TdtDecoderState()

    let started = DispatchTime.now().uptimeNanoseconds
    let result = try await manager.transcribe(samples, decoderState: &decoderState)
    let inferSeconds = elapsedSeconds(since: started)

    writeJSON(
        TranscriptionOutput(
            transcript: result.text,
            load_s: loadSeconds,
            infer_s: inferSeconds,
            audio_s: audioSeconds,
            rtf: audioSeconds > 0 ? inferSeconds / audioSeconds : 0,
            model_id: modelID,
            compute_units: describeComputeUnits(models.configuration.computeUnits)
        )
    )
}

private func selfcheck(modelDirectory: URL?) async throws {
    let (models, _, loadSeconds) = try await loadManager(modelDirectory: modelDirectory)
    writeJSON(
        SelfcheckOutput(
            load_s: loadSeconds,
            model_id: modelID,
            compute_units: describeComputeUnits(models.configuration.computeUnits)
        )
    )
}

private func elapsedSeconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
}

private func describeComputeUnits(_ units: MLComputeUnits) -> String {
    switch units {
    case .cpuOnly:
        return "cpuOnly"
    case .cpuAndGPU:
        return "cpuAndGPU"
    case .cpuAndNeuralEngine:
        return "cpuAndNeuralEngine"
    case .all:
        return "all"
    @unknown default:
        return "unknown(\(units.rawValue))"
    }
}

private func writeJSON<T: Encodable>(_ value: T) {
    do {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(value)
        data.append(0x0A)
        FileHandle.standardOutput.write(data)
    } catch {
        let fallback = Data("{\"error\":\"failed to encode JSON output\"}\n".utf8)
        FileHandle.standardOutput.write(fallback)
        writeDiagnostic("failed to encode JSON output: \(error.localizedDescription)")
        exit(EXIT_FAILURE)
    }
}

private func writeDiagnostic(_ message: String) {
    FileHandle.standardError.write(Data("parakeet-ane: \(message)\n".utf8))
}
