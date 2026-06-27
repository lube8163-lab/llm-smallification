import CoreML
import Darwin
import Foundation

enum ProbeComputeSelection: String, CaseIterable, Identifiable {
    case all
    case cpuAndGPU
    case cpuOnly
    case cpuAndNeuralEngine

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: "All"
        case .cpuAndGPU: "CPU+GPU"
        case .cpuOnly: "CPU"
        case .cpuAndNeuralEngine: "CPU+ANE"
        }
    }

    var units: MLComputeUnits {
        switch self {
        case .all: .all
        case .cpuAndGPU: .cpuAndGPU
        case .cpuOnly: .cpuOnly
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        }
    }
}

struct ProbeStep: Identifiable {
    let id = UUID()
    let name: String
    let seconds: Double?
    let memoryMB: Double
    let detail: String

    var durationText: String {
        guard let seconds else { return "-" }
        return String(format: "%.4fs", seconds)
    }

    var memoryText: String {
        String(format: "%.1f MB", memoryMB)
    }
}

struct ProbeReport {
    let steps: [ProbeStep]
    let summary: String
}

struct ProbeFailure: Error {
    let steps: [ProbeStep]
    let message: String
}

enum ProbeMemory {
    static func currentMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return -1 }
        return Double(info.phys_footprint) / 1_048_576.0
    }

    static func currentText() -> String {
        let value = currentMB()
        guard value >= 0 else { return "unknown" }
        return String(format: "%.1f MB", value)
    }
}

enum ProbeRunner {
    private static let embeddingName = "gemma4_12b_embedding_seq4_int4_block32"
    private static let decoderName = "gemma4_12b_layer0_decoder_seq4_mask_int4_block32"
    private static let lmHeadName = "gemma4_12b_lm_head_1tok_int4_block32"

    static func run(computeSelection: ProbeComputeSelection) -> Result<ProbeReport, ProbeFailure> {
        var steps: [ProbeStep] = []

        func record(_ name: String, seconds: Double? = nil, detail: String = "") {
            appendStep(ProbeStep(name: name, seconds: seconds, memoryMB: ProbeMemory.currentMB(), detail: detail), to: &steps)
        }

        do {
            print("[CoreMLProbe] run started compute=\(computeSelection.title)")
            record("Start", detail: computeSelection.title)

            let config = MLModelConfiguration()
            config.computeUnits = computeSelection.units

            let embedding = try loadModel(named: embeddingName, config: config, steps: &steps)
            let decoder = try loadModel(named: decoderName, config: config, steps: &steps)
            let lmHead = try loadModel(named: lmHeadName, config: config, steps: &steps)

            let inputIDs = try makeInputIDs()
            let positionIDs = try makePositionIDs()
            let mask = try makeCausalMask()

            let embeddingOutput = try timedPrediction(
                name: "Embedding",
                model: embedding,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "input_ids": MLFeatureValue(multiArray: inputIDs)
                ]),
                steps: &steps
            )
            let hidden = try requireArray(named: "hidden", output: embeddingOutput)

            let decoderOutput = try timedPrediction(
                name: "Decoder layer 0",
                model: decoder,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "x": MLFeatureValue(multiArray: hidden),
                    "position_ids": MLFeatureValue(multiArray: positionIDs),
                    "attention_mask": MLFeatureValue(multiArray: mask)
                ]),
                steps: &steps
            )
            let decoded = try requireArray(named: "y", output: decoderOutput)
            let lastHidden = try copyLastToken(from: decoded)

            let headOutput = try timedPrediction(
                name: "LM head",
                model: lmHead,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "hidden": MLFeatureValue(multiArray: lastHidden)
                ]),
                steps: &steps
            )
            let logits = try requireArray(named: "logits", output: headOutput)
            let top = topLogitSummary(logits)
            record("Top logits", detail: top)
            print("[CoreMLProbe] run finished OK: \(top)")

            return .success(ProbeReport(steps: steps, summary: "OK: \(top)"))
        } catch {
            record("Error", detail: String(describing: error))
            print("[CoreMLProbe] run failed: \(String(describing: error))")
            return .failure(ProbeFailure(steps: steps, message: String(describing: error)))
        }
    }

    private static func appendStep(_ step: ProbeStep, to steps: inout [ProbeStep]) {
        steps.append(step)
        print("[CoreMLProbe] \(step.name) duration=\(step.durationText) memory=\(step.memoryText) detail=\(step.detail)")
    }

    private static func loadModel(named name: String, config: MLModelConfiguration, steps: inout [ProbeStep]) throws -> MLModel {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc", subdirectory: "Models") else {
            throw ProbeError.missingModel(name)
        }

        let start = Date()
        let model = try MLModel(contentsOf: url, configuration: config)
        appendStep(ProbeStep(
            name: "Load \(name)",
            seconds: Date().timeIntervalSince(start),
            memoryMB: ProbeMemory.currentMB(),
            detail: url.lastPathComponent
        ), to: &steps)
        return model
    }

    private static func timedPrediction(
        name: String,
        model: MLModel,
        provider: MLFeatureProvider,
        steps: inout [ProbeStep]
    ) throws -> MLFeatureProvider {
        let start = Date()
        let output = try model.prediction(from: provider)
        appendStep(ProbeStep(
            name: name,
            seconds: Date().timeIntervalSince(start),
            memoryMB: ProbeMemory.currentMB(),
            detail: output.featureNames.sorted().joined(separator: ", ")
        ), to: &steps)
        return output
    }

    private static func requireArray(named name: String, output: MLFeatureProvider) throws -> MLMultiArray {
        guard let array = output.featureValue(for: name)?.multiArrayValue else {
            throw ProbeError.missingOutput(name)
        }
        return array
    }

    private static func makeInputIDs() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 4], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        pointer[0] = 2
        pointer[1] = 123
        pointer[2] = 4567
        pointer[3] = 106
        return array
    }

    private static func makePositionIDs() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 4], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for index in 0..<array.count {
            pointer[index] = Int32(index)
        }
        return array
    }

    private static func makeCausalMask() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 1, 4, 4], dataType: .float16)
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        for index in 0..<array.count {
            pointer[index] = 0
        }
        for row in 0..<4 {
            for column in 0..<4 where column > row {
                pointer[row * 4 + column] = Float16(-65504.0)
            }
        }
        return array
    }

    private static func copyLastToken(from decoded: MLMultiArray) throws -> MLMultiArray {
        guard decoded.dataType == .float16, decoded.count >= 4 * 3840 else {
            throw ProbeError.unexpectedShape("decoder output \(decoded.shape)")
        }

        let array = try MLMultiArray(shape: [1, 1, 3840], dataType: .float16)
        let source = decoded.dataPointer.bindMemory(to: Float16.self, capacity: decoded.count)
        let destination = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        let sourceOffset = 3 * 3840
        for index in 0..<3840 {
            destination[index] = source[sourceOffset + index]
        }
        return array
    }

    private static func topLogitSummary(_ logits: MLMultiArray) -> String {
        guard logits.dataType == .float16 else {
            return "dtype \(logits.dataType.rawValue)"
        }

        let pointer = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
        var bestIndex = 0
        var bestValue = Float(pointer[0])
        for index in 1..<logits.count {
            let value = Float(pointer[index])
            if value > bestValue {
                bestValue = value
                bestIndex = index
            }
        }
        return "#\(bestIndex) \(String(format: "%.3f", bestValue))"
    }
}

enum ProbeError: LocalizedError {
    case missingModel(String)
    case missingOutput(String)
    case unexpectedShape(String)

    var errorDescription: String? {
        switch self {
        case .missingModel(let name):
            "Missing model: Models/\(name).mlmodelc"
        case .missingOutput(let name):
            "Missing output: \(name)"
        case .unexpectedShape(let detail):
            "Unexpected shape: \(detail)"
        }
    }
}
