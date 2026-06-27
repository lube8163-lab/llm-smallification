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

    static func selectedFromProcess(default fallback: ProbeComputeSelection) -> ProbeComputeSelection {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["COREML_PROBE_COMPUTE"], let selection = ProbeComputeSelection(rawValue: value) {
            return selection
        }
        return fallback
    }
}

enum ProbeRunMode: String, CaseIterable, Identifiable {
    case loadEmbedding = "load-embedding"
    case loadDecoder = "load-decoder"
    case loadLMHead = "load-lm-head"
    case loadAllSequential = "load-all-sequential"
    case loadDecoderStack = "load-decoder-stack"
    case embeddingOnly = "embedding-only"
    case decoderOnly = "decoder-only"
    case decoderStack = "decoder-stack"
    case lmHeadOnly = "lm-head-only"
    case fullSequential = "full-sequential"
    case fullStackSequential = "full-stack-sequential"
    case generateOneToken = "generate-one-token"
    case generateTokenLoop = "generate-token-loop"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .loadEmbedding: "Load embedding"
        case .loadDecoder: "Load layer 0"
        case .loadLMHead: "Load LM head"
        case .loadAllSequential: "Load all sequential"
        case .loadDecoderStack: "Load decoder stack"
        case .embeddingOnly: "Embedding predict"
        case .decoderOnly: "Layer 0 synthetic"
        case .decoderStack: "Decoder stack"
        case .lmHeadOnly: "LM head synthetic"
        case .fullSequential: "Full sequential"
        case .fullStackSequential: "Full stack sequential"
        case .generateOneToken: "Generate one token"
        case .generateTokenLoop: "Generate token loop"
        }
    }

    var usesInputIDs: Bool {
        switch self {
        case .embeddingOnly, .fullSequential, .fullStackSequential, .generateOneToken, .generateTokenLoop:
            true
        case .loadEmbedding, .loadDecoder, .loadLMHead, .loadAllSequential, .loadDecoderStack, .decoderOnly, .decoderStack, .lmHeadOnly:
            false
        }
    }

    var usesGeneratedTokenCount: Bool {
        self == .generateTokenLoop
    }

    static func selectedFromProcess(default fallback: ProbeRunMode) -> ProbeRunMode {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["COREML_PROBE_MODE"], let mode = ProbeRunMode(rawValue: value) {
            return mode
        }

        let prefix = "--mode="
        if let argument = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix(prefix) }) {
            let value = String(argument.dropFirst(prefix.count))
            if let mode = ProbeRunMode(rawValue: value) {
                return mode
            }
        }
        return fallback
    }
}

enum ProbeLayerSelection: String, CaseIterable, Identifiable {
    case first1 = "first-1"
    case first2 = "first-2"
    case first4 = "first-4"
    case first8 = "first-8"
    case first16 = "first-16"
    case first24 = "first-24"
    case first32 = "first-32"
    case first40 = "first-40"
    case first44 = "first-44"
    case first48 = "first-48"
    case all

    var id: String { rawValue }

    var title: String {
        switch self {
        case .first1: "First 1"
        case .first2: "First 2"
        case .first4: "First 4"
        case .first8: "First 8"
        case .first16: "First 16"
        case .first24: "First 24"
        case .first32: "First 32"
        case .first40: "First 40"
        case .first44: "First 44"
        case .first48: "First 48"
        case .all: "All available"
        }
    }

    var requestedCount: Int? {
        switch self {
        case .first1: 1
        case .first2: 2
        case .first4: 4
        case .first8: 8
        case .first16: 16
        case .first24: 24
        case .first32: 32
        case .first40: 40
        case .first44: 44
        case .first48: 48
        case .all: nil
        }
    }

    static func selectedFromProcess(default fallback: ProbeLayerSelection) -> ProbeLayerSelection {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["COREML_PROBE_LAYERS"], let selection = selection(from: value) {
            return selection
        }

        let prefix = "--layers="
        if let argument = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix(prefix) }) {
            let value = String(argument.dropFirst(prefix.count))
            if let selection = selection(from: value) {
                return selection
            }
        }
        return fallback
    }

    private static func selection(from value: String) -> ProbeLayerSelection? {
        if let selection = ProbeLayerSelection(rawValue: value) {
            return selection
        }

        return switch value {
        case "1": .first1
        case "2": .first2
        case "4": .first4
        case "8": .first8
        case "16": .first16
        case "24": .first24
        case "32": .first32
        case "40": .first40
        case "44": .first44
        case "48": .first48
        default: nil
        }
    }
}

enum ProbeCacheClearPolicy: String, CaseIterable, Identifiable {
    case afterEveryModel = "every-model"
    case afterEvery4DecoderLayers = "every-4-layers"
    case afterEvery8DecoderLayers = "every-8-layers"
    case afterEachToken = "per-token"
    case runEndOnly = "run-end-only"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .afterEveryModel: "Every model"
        case .afterEvery4DecoderLayers: "Every 4 layers"
        case .afterEvery8DecoderLayers: "Every 8 layers"
        case .afterEachToken: "Per token"
        case .runEndOnly: "Run end"
        }
    }

    var clearsAfterNonDecoderRelease: Bool {
        self == .afterEveryModel
    }

    var clearsAtRunEnd: Bool {
        self != .afterEveryModel
    }

    func decoderReleaseReason(layerPosition: Int, totalLayers: Int, layerName: String) -> String? {
        let isLastLayer = layerPosition == totalLayers
        switch self {
        case .afterEveryModel:
            return "released \(layerName)"
        case .afterEvery4DecoderLayers where layerPosition.isMultiple(of: 4) || isLastLayer:
            return "released \(layerName) policy=\(rawValue) position=\(layerPosition)/\(totalLayers)"
        case .afterEvery8DecoderLayers where layerPosition.isMultiple(of: 8) || isLastLayer:
            return "released \(layerName) policy=\(rawValue) position=\(layerPosition)/\(totalLayers)"
        case .afterEvery4DecoderLayers, .afterEvery8DecoderLayers, .afterEachToken, .runEndOnly:
            return nil
        }
    }

    static func selectedFromProcess(default fallback: ProbeCacheClearPolicy) -> ProbeCacheClearPolicy {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["COREML_PROBE_CACHE_POLICY"], let policy = policy(from: value) {
            return policy
        }

        let prefix = "--cache-policy="
        if let argument = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix(prefix) }) {
            let value = String(argument.dropFirst(prefix.count))
            if let policy = policy(from: value) {
                return policy
            }
        }
        return fallback
    }

    private static func policy(from value: String) -> ProbeCacheClearPolicy? {
        if let policy = ProbeCacheClearPolicy(rawValue: value) {
            return policy
        }

        return switch value {
        case "every-layer", "every": .afterEveryModel
        case "4", "every-4", "after-4": .afterEvery4DecoderLayers
        case "8", "every-8", "after-8": .afterEvery8DecoderLayers
        case "token", "each-token": .afterEachToken
        case "end", "run-end": .runEndOnly
        default: nil
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

struct DecoderLayerModel {
    let index: Int
    let name: String
}

struct DecoderLayerPlan {
    let availableCount: Int
    let layers: [DecoderLayerModel]
    let selection: ProbeLayerSelection

    var detail: String {
        let requested = selection.requestedCount.map(String.init) ?? "all"
        return "selected=\(layers.count), available=\(availableCount), requested=\(requested)"
    }
}

struct TokenPrediction {
    let step: Int
    let index: Int
    let logit: Float
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
    private static let decoderPrefix = "gemma4_12b_layer"
    private static let decoderSuffix = "_decoder_seq4_mask_int4_block32"
    private static let lmHeadName = "gemma4_12b_lm_head_1tok_int4_block32"
    private static let defaultInputIDs: [Int32] = [2, 123, 4567, 106]
    static let defaultGeneratedTokenCount = 2
    static let maxGeneratedTokenCount = 4

    static var defaultInputIDsText: String {
        formatInputIDs(defaultInputIDs)
    }

    static func selectedInputIDsTextFromProcess() -> String {
        processInputIDsText() ?? defaultInputIDsText
    }

    static func selectedGeneratedTokenCountFromProcess() -> Int {
        guard let rawValue = processGeneratedTokenCountText(),
              let count = Int(rawValue),
              (1...maxGeneratedTokenCount).contains(count) else {
            return defaultGeneratedTokenCount
        }
        return count
    }

    static func run(
        computeSelection: ProbeComputeSelection,
        mode: ProbeRunMode,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy = .afterEveryModel,
        inputIDsText: String? = nil,
        generatedTokenCount: Int? = nil
    ) -> Result<ProbeReport, ProbeFailure> {
        var steps: [ProbeStep] = []

        do {
            print("[CoreMLProbe] run started compute=\(computeSelection.title) mode=\(mode.rawValue) layers=\(layerSelection.rawValue) cache=\(cacheClearPolicy.rawValue)")
            recordStep("Start", detail: "\(computeSelection.title), \(mode.title), \(layerSelection.title), cache=\(cacheClearPolicy.title)", steps: &steps)
            clearCoreMLRuntimeCache(reason: "run start", steps: &steps)

            let config = MLModelConfiguration()
            config.computeUnits = computeSelection.units

            let summary: String
            switch mode {
            case .loadEmbedding:
                try runLoadOnly(
                    named: embeddingName,
                    config: config,
                    cacheClearReason: cacheClearPolicy.clearsAfterNonDecoderRelease ? "released \(embeddingName)" : nil,
                    steps: &steps
                )
                summary = "OK: loaded embedding"
            case .loadDecoder:
                try runLoadOnly(
                    named: decoderName,
                    config: config,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: decoderName),
                    steps: &steps
                )
                summary = "OK: loaded layer 0"
            case .loadLMHead:
                try runLoadOnly(
                    named: lmHeadName,
                    config: config,
                    cacheClearReason: cacheClearPolicy.clearsAfterNonDecoderRelease ? "released \(lmHeadName)" : nil,
                    steps: &steps
                )
                summary = "OK: loaded LM head"
            case .loadAllSequential:
                try runLoadOnly(
                    named: embeddingName,
                    config: config,
                    cacheClearReason: cacheClearPolicy.clearsAfterNonDecoderRelease ? "released \(embeddingName)" : nil,
                    steps: &steps
                )
                try runLoadOnly(
                    named: decoderName,
                    config: config,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: decoderName),
                    steps: &steps
                )
                try runLoadOnly(
                    named: lmHeadName,
                    config: config,
                    cacheClearReason: cacheClearPolicy.clearsAfterNonDecoderRelease ? "released \(lmHeadName)" : nil,
                    steps: &steps
                )
                summary = "OK: loaded all sequentially"
            case .loadDecoderStack:
                let plan = try decoderLayerPlan(selection: layerSelection)
                recordStep("Decoder stack", detail: plan.detail, steps: &steps)
                for (offset, layer) in plan.layers.enumerated() {
                    let layerPosition = offset + 1
                    try runLoadOnly(
                        named: layer.name,
                        config: config,
                        cacheClearReason: cacheClearPolicy.decoderReleaseReason(
                            layerPosition: layerPosition,
                            totalLayers: plan.layers.count,
                            layerName: layer.name
                        ),
                        steps: &steps
                    )
                }
                summary = "OK: loaded \(plan.layers.count) decoder layers"
            case .embeddingOnly:
                let inputIDs = try selectedInputIDs(overrideText: inputIDsText)
                let hidden = try runEmbedding(config: config, inputIDs: inputIDs, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                summary = "OK: hidden \(hidden.shape)"
            case .decoderOnly:
                let hidden = try makeHidden(seqLength: 4)
                let decoded = try runDecoder(
                    layer: DecoderLayerModel(index: 0, name: decoderName),
                    hidden: hidden,
                    config: config,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: decoderName),
                    steps: &steps
                )
                summary = "OK: decoded \(decoded.shape)"
            case .decoderStack:
                let hidden = try makeHidden(seqLength: 4)
                let decoded = try runDecoderStack(
                    hidden: hidden,
                    config: config,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    steps: &steps
                )
                summary = "OK: decoded stack \(decoded.shape)"
            case .lmHeadOnly:
                let logits = try runLMHead(hidden: try makeLastHidden(), config: config, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let top = topLogitSummary(logits)
                recordStep("Top logits", detail: top, steps: &steps)
                summary = "OK: \(top)"
            case .fullSequential:
                let inputIDs = try selectedInputIDs(overrideText: inputIDsText)
                let hidden = try runEmbedding(config: config, inputIDs: inputIDs, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let decoded = try runDecoder(
                    layer: DecoderLayerModel(index: 0, name: decoderName),
                    hidden: hidden,
                    config: config,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: decoderName),
                    steps: &steps
                )
                let lastHidden = try copyLastToken(from: decoded)
                let logits = try runLMHead(hidden: lastHidden, config: config, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let top = topLogitSummary(logits)
                recordStep("Top logits", detail: top, steps: &steps)
                summary = "OK: \(top)"
            case .fullStackSequential:
                let inputIDs = try selectedInputIDs(overrideText: inputIDsText)
                let hidden = try runEmbedding(config: config, inputIDs: inputIDs, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let decoded = try runDecoderStack(
                    hidden: hidden,
                    config: config,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    steps: &steps
                )
                let lastHidden = try copyLastToken(from: decoded)
                let logits = try runLMHead(hidden: lastHidden, config: config, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let top = topLogitSummary(logits)
                recordStep("Top logits", detail: top, steps: &steps)
                summary = "OK: \(top)"
            case .generateOneToken:
                let token = try runGenerateOneToken(
                    config: config,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    inputIDsText: inputIDsText,
                    steps: &steps
                )
                summary = "OK: next token #\(token.index) \(String(format: "%.3f", token.logit))"
            case .generateTokenLoop:
                let predictions = try runGenerateTokenLoop(
                    config: config,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    inputIDsText: inputIDsText,
                    generatedTokenCount: generatedTokenCount,
                    steps: &steps
                )
                let tokens = predictions.map { "#\($0.index)" }.joined(separator: ",")
                summary = "OK: generated \(predictions.count) tokens \(tokens)"
            }

            if cacheClearPolicy.clearsAtRunEnd {
                clearCoreMLRuntimeCache(reason: "run end policy=\(cacheClearPolicy.rawValue)", steps: &steps)
            }

            print("[CoreMLProbe] run finished \(summary)")
            return .success(ProbeReport(steps: steps, summary: summary))
        } catch {
            clearCoreMLRuntimeCache(reason: "error cleanup", steps: &steps)
            recordStep("Error", detail: String(describing: error), steps: &steps)
            print("[CoreMLProbe] run failed: \(String(describing: error))")
            return .failure(ProbeFailure(steps: steps, message: String(describing: error)))
        }
    }

    private static func recordStep(_ name: String, seconds: Double? = nil, detail: String = "", steps: inout [ProbeStep]) {
        appendStep(ProbeStep(name: name, seconds: seconds, memoryMB: ProbeMemory.currentMB(), detail: detail), to: &steps)
    }

    private static func appendStep(_ step: ProbeStep, to steps: inout [ProbeStep]) {
        steps.append(step)
        print("[CoreMLProbe] \(step.name) duration=\(step.durationText) memory=\(step.memoryText) detail=\(step.detail)")
    }

    private static func runLoadOnly(
        named name: String,
        config: MLModelConfiguration,
        cacheClearReason: String?,
        steps: inout [ProbeStep]
    ) throws {
        try autoreleasepool {
            _ = try loadModel(named: name, config: config, steps: &steps)
            recordStep("Loaded \(name)", detail: "leaving autorelease scope", steps: &steps)
        }
        recordStep("Released \(name)", detail: "ARC scope exited", steps: &steps)
        if let cacheClearReason {
            clearCoreMLRuntimeCache(reason: cacheClearReason, steps: &steps)
        }
    }

    private static func runEmbedding(
        config: MLModelConfiguration,
        inputIDs: [Int32],
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        recordStep("Prompt IDs", detail: inputIDs.map(String.init).joined(separator: ","), steps: &steps)

        let hidden = try autoreleasepool {
            let embedding = try loadModel(named: embeddingName, config: config, steps: &steps)
            return try predictEmbedding(model: embedding, inputIDs: inputIDs, name: "Embedding", steps: &steps)
        }
        recordStep("Released \(embeddingName)", detail: "hidden retained", steps: &steps)
        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released \(embeddingName)", steps: &steps)
        }
        return hidden
    }

    private static func runGenerateOneToken(
        config: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        inputIDsText: String?,
        steps: inout [ProbeStep]
    ) throws -> (index: Int, logit: Float) {
        let inputIDs = try selectedInputIDs(overrideText: inputIDsText)
        let hidden = try runEmbedding(config: config, inputIDs: inputIDs, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let decoded = try runDecoderStack(
            hidden: hidden,
            config: config,
            layerSelection: layerSelection,
            cacheClearPolicy: cacheClearPolicy,
            steps: &steps
        )
        let lastHidden = try copyLastToken(from: decoded)
        let logits = try runLMHead(hidden: lastHidden, config: config, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let token = try topLogit(logits)
        recordStep(
            "Next token",
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))",
            steps: &steps
        )
        return token
    }

    private static func runGenerateTokenLoop(
        config: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        inputIDsText: String?,
        generatedTokenCount: Int?,
        steps: inout [ProbeStep]
    ) throws -> [TokenPrediction] {
        var inputWindow = try selectedInputIDs(overrideText: inputIDsText)
        let tokenCount = try selectedGeneratedTokenCount(override: generatedTokenCount)
        recordStep(
            "Generation loop",
            detail: "tokens=\(tokenCount) strategy=reuse embedding+lm-head reload decoder stack cache=\(cacheClearPolicy.rawValue)",
            steps: &steps
        )

        let predictions = try autoreleasepool { () throws -> [TokenPrediction] in
            let embedding = try loadModel(named: embeddingName, config: config, steps: &steps)
            let lmHead = try loadModel(named: lmHeadName, config: config, steps: &steps)
            var localPredictions: [TokenPrediction] = []

            for step in 1...tokenCount {
                let prediction = try autoreleasepool { () throws -> TokenPrediction in
                    recordStep("Prompt IDs \(step)", detail: formatInputIDs(inputWindow), steps: &steps)
                    let hidden = try predictEmbedding(
                        model: embedding,
                        inputIDs: inputWindow,
                        name: "Embedding token \(step)",
                        steps: &steps
                    )
                    let decoded = try runDecoderStack(
                        hidden: hidden,
                        config: config,
                        layerSelection: layerSelection,
                        cacheClearPolicy: cacheClearPolicy,
                        steps: &steps
                    )
                    let lastHidden = try copyLastToken(from: decoded)
                    let logits = try predictLMHead(
                        model: lmHead,
                        hidden: lastHidden,
                        name: "LM head token \(step)",
                        steps: &steps
                    )
                    let token = try topLogit(logits)
                    recordStep(
                        "Generated token \(step)",
                        detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))",
                        steps: &steps
                    )
                    return TokenPrediction(step: step, index: token.index, logit: token.logit)
                }

                if cacheClearPolicy == .afterEachToken {
                    clearCoreMLRuntimeCache(reason: "generated token \(step) policy=\(cacheClearPolicy.rawValue)", steps: &steps)
                }

                guard let nextToken = Int32(exactly: prediction.index) else {
                    throw ProbeError.invalidInputIDs("generated token does not fit Int32: \(prediction.index)")
                }
                inputWindow.removeFirst()
                inputWindow.append(nextToken)
                localPredictions.append(prediction)
            }

            return localPredictions
        }

        recordStep("Released generation endpoints", detail: "\(embeddingName), \(lmHeadName)", steps: &steps)
        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released generation endpoints", steps: &steps)
        }
        let detail = predictions
            .map { "\($0.step):#\($0.index)=\(String(format: "%.3f", $0.logit))" }
            .joined(separator: ", ")
        recordStep("Generated tokens", detail: detail, steps: &steps)
        return predictions
    }

    private static func runDecoderStack(
        hidden: MLMultiArray,
        config: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let plan = try decoderLayerPlan(selection: layerSelection)
        recordStep("Decoder stack", detail: plan.detail, steps: &steps)

        var current = hidden
        for (offset, layer) in plan.layers.enumerated() {
            let layerPosition = offset + 1
            current = try runDecoder(
                layer: layer,
                hidden: current,
                config: config,
                cacheClearReason: cacheClearPolicy.decoderReleaseReason(
                    layerPosition: layerPosition,
                    totalLayers: plan.layers.count,
                    layerName: layer.name
                ),
                steps: &steps
            )
        }
        return current
    }

    private static func runDecoder(
        layer: DecoderLayerModel,
        hidden: MLMultiArray,
        config: MLModelConfiguration,
        cacheClearReason: String?,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let decoded = try autoreleasepool {
            let decoder = try loadModel(named: layer.name, config: config, steps: &steps)
            let positionIDs = try makePositionIDs()
            let mask = try makeCausalMask()
            let output = try timedPrediction(
                name: "Decoder layer \(layer.index)",
                model: decoder,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "x": MLFeatureValue(multiArray: hidden),
                    "position_ids": MLFeatureValue(multiArray: positionIDs),
                    "attention_mask": MLFeatureValue(multiArray: mask)
                ]),
                steps: &steps
            )
            return try requireArray(named: "y", output: output)
        }
        recordStep("Released \(layer.name)", detail: "decoded retained", steps: &steps)
        if let cacheClearReason {
            clearCoreMLRuntimeCache(reason: cacheClearReason, steps: &steps)
        }
        return decoded
    }

    private static func decoderLayerPlan(selection: ProbeLayerSelection) throws -> DecoderLayerPlan {
        var byIndex: [Int: DecoderLayerModel] = [:]
        let urls = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: "Models") ?? []

        for url in urls {
            let name = url.deletingPathExtension().lastPathComponent
            guard name.hasPrefix(decoderPrefix), name.hasSuffix(decoderSuffix) else {
                continue
            }

            let start = name.index(name.startIndex, offsetBy: decoderPrefix.count)
            let end = name.index(name.endIndex, offsetBy: -decoderSuffix.count)
            let numberText = String(name[start..<end])
            guard let index = Int(numberText) else {
                continue
            }

            let candidate = DecoderLayerModel(index: index, name: name)
            if let existing = byIndex[index] {
                byIndex[index] = candidate.name.count > existing.name.count ? candidate : existing
            } else {
                byIndex[index] = candidate
            }
        }

        let layers = byIndex.values.sorted { $0.index < $1.index }
        guard !layers.isEmpty else {
            throw ProbeError.missingModel("\(decoderPrefix)*\(decoderSuffix).mlmodelc")
        }

        let selectedLayers: [DecoderLayerModel]
        if let count = selection.requestedCount {
            selectedLayers = Array(layers.prefix(count))
        } else {
            selectedLayers = layers
        }

        return DecoderLayerPlan(
            availableCount: layers.count,
            layers: selectedLayers,
            selection: selection
        )
    }

    private static func runLMHead(
        hidden: MLMultiArray,
        config: MLModelConfiguration,
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let logits = try autoreleasepool {
            let lmHead = try loadModel(named: lmHeadName, config: config, steps: &steps)
            return try predictLMHead(model: lmHead, hidden: hidden, name: "LM head", steps: &steps)
        }
        recordStep("Released \(lmHeadName)", detail: "logits retained", steps: &steps)
        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released \(lmHeadName)", steps: &steps)
        }
        return logits
    }

    private static func predictEmbedding(
        model: MLModel,
        inputIDs: [Int32],
        name: String,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let inputIDs = try makeInputIDs(values: inputIDs)
        let output = try timedPrediction(
            name: name,
            model: model,
            provider: MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIDs)
            ]),
            steps: &steps
        )
        return try requireArray(named: "hidden", output: output)
    }

    private static func predictLMHead(
        model: MLModel,
        hidden: MLMultiArray,
        name: String,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let output = try timedPrediction(
            name: name,
            model: model,
            provider: MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLFeatureValue(multiArray: hidden)
            ]),
            steps: &steps
        )
        return try requireArray(named: "logits", output: output)
    }

    private static func clearCoreMLRuntimeCache(reason: String, steps: inout [ProbeStep]) {
        let fileManager = FileManager.default
        let start = Date()
        let targets = coreMLRuntimeCacheTargets(fileManager: fileManager)
        var removed: [String] = []
        var failures: [String] = []

        for target in targets {
            guard fileManager.fileExists(atPath: target.path) else {
                continue
            }

            do {
                try fileManager.removeItem(at: target)
                removed.append(cacheTargetLabel(target))
            } catch {
                failures.append("\(cacheTargetLabel(target)): \(error.localizedDescription)")
            }
        }

        guard !removed.isEmpty || !failures.isEmpty else {
            return
        }

        let detailParts = [
            "reason=\(reason)",
            "removed=\(removed.isEmpty ? "none" : removed.joined(separator: ","))",
            failures.isEmpty ? nil : "failed=\(failures.joined(separator: ","))"
        ].compactMap { $0 }

        appendStep(ProbeStep(
            name: "Clear Core ML cache",
            seconds: Date().timeIntervalSince(start),
            memoryMB: ProbeMemory.currentMB(),
            detail: detailParts.joined(separator: " ")
        ), to: &steps)
    }

    private static func coreMLRuntimeCacheTargets(fileManager: FileManager) -> [URL] {
        guard let cachesDirectory = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            return []
        }

        let bundleID = Bundle.main.bundleIdentifier ?? "lab.lube8163.CoreMLProbe"
        var targets = [
            cachesDirectory
                .appendingPathComponent(bundleID, isDirectory: true)
                .appendingPathComponent("com.apple.e5rt.e5bundlecache", isDirectory: true),
            cachesDirectory
                .appendingPathComponent("com.apple.e5rt.e5bundlecache", isDirectory: true)
        ]

        if let children = try? fileManager.contentsOfDirectory(
            at: cachesDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for child in children where child.lastPathComponent == bundleID {
                targets.append(child.appendingPathComponent("com.apple.e5rt.e5bundlecache", isDirectory: true))
            }
        }

        var seen: Set<String> = []
        return targets.filter { seen.insert($0.path).inserted }
    }

    private static func cacheTargetLabel(_ url: URL) -> String {
        let parent = url.deletingLastPathComponent().lastPathComponent
        return "\(parent)/\(url.lastPathComponent)"
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

    private static func makeHidden(seqLength: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: seqLength), 3840], dataType: .float16)
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        for index in 0..<array.count {
            pointer[index] = 0
        }
        return array
    }

    private static func makeLastHidden() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 1, 3840], dataType: .float16)
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        for index in 0..<array.count {
            pointer[index] = 0
        }
        return array
    }

    private static func selectedInputIDs(overrideText: String?) throws -> [Int32] {
        guard let rawValue = cleanedOverride(overrideText) ?? processInputIDsText() else {
            return defaultInputIDs
        }

        let values = rawValue.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard values.count == 4 else {
            throw ProbeError.invalidInputIDs("expected exactly 4 comma-separated token IDs, got \(values.count)")
        }

        return try values.map { value in
            guard let parsed = Int32(value) else {
                throw ProbeError.invalidInputIDs("not an Int32 token ID: \(value)")
            }
            return parsed
        }
    }

    private static func selectedGeneratedTokenCount(override: Int?) throws -> Int {
        let count: Int
        if let override {
            count = override
        } else if let rawValue = processGeneratedTokenCountText(), let parsed = Int(rawValue) {
            count = parsed
        } else {
            count = defaultGeneratedTokenCount
        }

        guard (1...maxGeneratedTokenCount).contains(count) else {
            throw ProbeError.invalidTokenCount("expected 1...\(maxGeneratedTokenCount), got \(count)")
        }
        return count
    }

    private static func cleanedOverride(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func processInputIDsText() -> String? {
        let environment = ProcessInfo.processInfo.environment
        let prefix = "--input-ids="
        return environment["COREML_PROBE_INPUT_IDS"]
            ?? ProcessInfo.processInfo.arguments
                .first(where: { $0.hasPrefix(prefix) })
                .map { String($0.dropFirst(prefix.count)) }
    }

    private static func processGeneratedTokenCountText() -> String? {
        let environment = ProcessInfo.processInfo.environment
        let prefix = "--tokens="
        return environment["COREML_PROBE_GENERATE_TOKENS"]
            ?? ProcessInfo.processInfo.arguments
                .first(where: { $0.hasPrefix(prefix) })
                .map { String($0.dropFirst(prefix.count)) }
    }

    private static func formatInputIDs(_ inputIDs: [Int32]) -> String {
        inputIDs.map(String.init).joined(separator: ",")
    }

    private static func makeInputIDs(values: [Int32]) throws -> MLMultiArray {
        guard values.count == 4 else {
            throw ProbeError.invalidInputIDs("expected 4 token IDs, got \(values.count)")
        }

        let array = try MLMultiArray(shape: [1, 4], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for index in 0..<values.count {
            pointer[index] = values[index]
        }
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
        guard let token = try? topLogit(logits) else {
            return "dtype \(logits.dataType.rawValue) shape \(logits.shape)"
        }

        return "#\(token.index) \(String(format: "%.3f", token.logit))"
    }

    private static func topLogit(_ logits: MLMultiArray) throws -> (index: Int, logit: Float) {
        guard logits.dataType == .float16 else {
            throw ProbeError.unexpectedShape("logits dtype \(logits.dataType.rawValue), shape \(logits.shape)")
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
        return (bestIndex, bestValue)
    }
}

enum ProbeError: LocalizedError {
    case missingModel(String)
    case missingOutput(String)
    case invalidInputIDs(String)
    case invalidTokenCount(String)
    case unexpectedShape(String)

    var errorDescription: String? {
        switch self {
        case .missingModel(let name):
            "Missing model: Models/\(name).mlmodelc"
        case .missingOutput(let name):
            "Missing output: \(name)"
        case .invalidInputIDs(let detail):
            "Invalid input IDs: \(detail)"
        case .invalidTokenCount(let detail):
            "Invalid token count: \(detail)"
        case .unexpectedShape(let detail):
            "Unexpected shape: \(detail)"
        }
    }
}
