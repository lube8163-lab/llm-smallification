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

    static let endpointCases: [ProbeComputeSelection] = [
        .cpuOnly,
        .cpuAndGPU
    ]

    static func selectedFromProcess(default fallback: ProbeComputeSelection) -> ProbeComputeSelection {
        selectedFromProcess(
            environmentKey: "COREML_PROBE_COMPUTE",
            argumentPrefix: "--compute=",
            default: fallback
        )
    }

    static func selectedEndpointFromProcess(default fallback: ProbeComputeSelection) -> ProbeComputeSelection {
        selectedFromProcess(
            environmentKey: "COREML_PROBE_ENDPOINT_COMPUTE",
            argumentPrefix: "--endpoint-compute=",
            default: selectedFromProcess(default: fallback)
        )
    }

    static func selectedDecoderFromProcess(default fallback: ProbeComputeSelection) -> ProbeComputeSelection {
        selectedFromProcess(
            environmentKey: "COREML_PROBE_DECODER_COMPUTE",
            argumentPrefix: "--decoder-compute=",
            default: selectedFromProcess(default: fallback)
        )
    }

    private static func selectedFromProcess(
        environmentKey: String,
        argumentPrefix: String,
        default fallback: ProbeComputeSelection
    ) -> ProbeComputeSelection {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment[environmentKey], let selection = ProbeComputeSelection(rawValue: value) {
            return selection
        }

        if let argument = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix(argumentPrefix) }) {
            let value = String(argument.dropFirst(argumentPrefix.count))
            if let selection = ProbeComputeSelection(rawValue: value) {
                return selection
            }
        }
        return fallback
    }
}

struct ProbeComputePlan {
    let endpoint: ProbeComputeSelection
    let decoder: ProbeComputeSelection

    static func shared(_ selection: ProbeComputeSelection) -> ProbeComputePlan {
        ProbeComputePlan(endpoint: selection, decoder: selection)
    }

    var logDetail: String {
        if endpoint == decoder {
            return endpoint.title
        }
        return "endpoints=\(endpoint.title), decoder=\(decoder.title)"
    }
}

enum ProbeSequenceLength: Int, CaseIterable, Identifiable {
    case seq4 = 4
    case seq20 = 20

    var id: Int { rawValue }

    var title: String {
        "Seq \(rawValue)"
    }

    var embeddingName: String {
        "gemma4_12b_embedding_seq\(rawValue)_int4_block32"
    }

    var decoderName: String {
        "gemma4_12b_layer0_decoder_seq\(rawValue)_mask_int4_block32"
    }

    var decoderSuffix: String {
        "_decoder_seq\(rawValue)_mask_int4_block32"
    }

    var defaultInputIDs: [Int32] {
        switch self {
        case .seq4:
            [2, 123, 4567, 106]
        case .seq20:
            [
                2, 105, 2364, 107, 85141, 236924, 238906, 237234, 7604, 31600,
                3335, 106, 107, 105, 4368, 107, 100, 45518, 107, 101
            ]
        }
    }

    static func selectedFromProcess(default fallback: ProbeSequenceLength) -> ProbeSequenceLength {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["COREML_PROBE_SEQ_LEN"], let selection = selection(from: value) {
            return selection
        }

        let prefix = "--seq-len="
        if let argument = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix(prefix) }) {
            let value = String(argument.dropFirst(prefix.count))
            if let selection = selection(from: value) {
                return selection
            }
        }
        return fallback
    }

    static func bestAvailable() -> ProbeSequenceLength {
        if modelExists(named: ProbeSequenceLength.seq20.embeddingName) {
            return .seq20
        }
        return .seq4
    }

    private static func selection(from value: String) -> ProbeSequenceLength? {
        if let parsed = Int(value) {
            return ProbeSequenceLength(rawValue: parsed)
        }
        return switch value.lowercased() {
        case "seq4", "s4": .seq4
        case "seq20", "s20": .seq20
        default: nil
        }
    }

    private static func modelExists(named name: String) -> Bool {
        Bundle.main.url(forResource: name, withExtension: "mlmodelc", subdirectory: "Models") != nil
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

    var usesEndpointModels: Bool {
        switch self {
        case .loadEmbedding, .loadLMHead, .loadAllSequential, .embeddingOnly,
                .lmHeadOnly, .fullSequential, .fullStackSequential,
                .generateOneToken, .generateTokenLoop:
            true
        case .loadDecoder, .loadDecoderStack, .decoderOnly, .decoderStack:
            false
        }
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

struct TokenWindow {
    var values: [Int32]
    var positionStart: Int

    var seqLength: Int {
        values.count
    }
}

struct LoadedProbeModel {
    let model: MLModel
    let name: String
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
    private static let decoderPrefix = "gemma4_12b_layer"
    private static let lmHeadName = "gemma4_12b_norm_lm_head_1tok_int4_block32"
    private static let legacyLMHeadName = "gemma4_12b_lm_head_1tok_int4_block32"
    private static let lmHeadNames = [lmHeadName, legacyLMHeadName]
    static let defaultGeneratedTokenCount = 2
    static let maxGeneratedTokenCount = 32

    static var defaultInputIDsText: String {
        defaultInputIDsText(sequenceLength: .seq4)
    }

    static func defaultInputIDsText(sequenceLength: ProbeSequenceLength) -> String {
        formatInputIDs(sequenceLength.defaultInputIDs)
    }

    static func selectedSequenceLengthFromProcess() -> ProbeSequenceLength {
        ProbeSequenceLength.selectedFromProcess(default: ProbeSequenceLength.bestAvailable())
    }

    static func selectedInputIDsTextFromProcess(sequenceLength: ProbeSequenceLength) -> String {
        processInputIDsText() ?? defaultInputIDsText(sequenceLength: sequenceLength)
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
        sequenceLength: ProbeSequenceLength = selectedSequenceLengthFromProcess(),
        inputIDsText: String? = nil,
        generatedTokenCount: Int? = nil
    ) -> Result<ProbeReport, ProbeFailure> {
        run(
            computePlan: .shared(computeSelection),
            mode: mode,
            layerSelection: layerSelection,
            cacheClearPolicy: cacheClearPolicy,
            sequenceLength: sequenceLength,
            inputIDsText: inputIDsText,
            generatedTokenCount: generatedTokenCount
        )
    }

    static func run(
        computePlan: ProbeComputePlan,
        mode: ProbeRunMode,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy = .afterEveryModel,
        sequenceLength: ProbeSequenceLength = selectedSequenceLengthFromProcess(),
        inputIDsText: String? = nil,
        generatedTokenCount: Int? = nil
    ) -> Result<ProbeReport, ProbeFailure> {
        var steps: [ProbeStep] = []

        do {
            print("[CoreMLProbe] run started compute=\(computePlan.logDetail) mode=\(mode.rawValue) seq=\(sequenceLength.rawValue) layers=\(layerSelection.rawValue) cache=\(cacheClearPolicy.rawValue)")
            recordStep("Start", detail: "\(computePlan.logDetail), \(mode.title), \(sequenceLength.title), \(layerSelection.title), cache=\(cacheClearPolicy.title)", steps: &steps)
            clearCoreMLRuntimeCache(reason: "run start", steps: &steps)
            try validateComputePlan(computePlan, mode: mode)

            let endpointConfig = makeConfig(computePlan.endpoint)
            let decoderConfig = makeConfig(computePlan.decoder)

            let summary: String
            switch mode {
            case .loadEmbedding:
                try runLoadOnly(
                    named: sequenceLength.embeddingName,
                    config: endpointConfig,
                    cacheClearReason: cacheClearPolicy.clearsAfterNonDecoderRelease ? "released \(sequenceLength.embeddingName)" : nil,
                    steps: &steps
                )
                summary = "OK: loaded embedding"
            case .loadDecoder:
                try runLoadOnly(
                    named: sequenceLength.decoderName,
                    config: decoderConfig,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: sequenceLength.decoderName),
                    steps: &steps
                )
                summary = "OK: loaded layer 0"
            case .loadLMHead:
                try runLoadLMHeadOnly(
                    config: endpointConfig,
                    cacheClearPolicy: cacheClearPolicy,
                    steps: &steps
                )
                summary = "OK: loaded LM head"
            case .loadAllSequential:
                try runLoadOnly(
                    named: sequenceLength.embeddingName,
                    config: endpointConfig,
                    cacheClearReason: cacheClearPolicy.clearsAfterNonDecoderRelease ? "released \(sequenceLength.embeddingName)" : nil,
                    steps: &steps
                )
                try runLoadOnly(
                    named: sequenceLength.decoderName,
                    config: decoderConfig,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: sequenceLength.decoderName),
                    steps: &steps
                )
                try runLoadLMHeadOnly(
                    config: endpointConfig,
                    cacheClearPolicy: cacheClearPolicy,
                    steps: &steps
                )
                summary = "OK: loaded all sequentially"
            case .loadDecoderStack:
                let plan = try decoderLayerPlan(selection: layerSelection, sequenceLength: sequenceLength)
                recordStep("Decoder stack", detail: plan.detail, steps: &steps)
                for (offset, layer) in plan.layers.enumerated() {
                    let layerPosition = offset + 1
                    try runLoadOnly(
                        named: layer.name,
                        config: decoderConfig,
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
                let inputWindow = try selectedInputWindow(overrideText: inputIDsText, sequenceLength: sequenceLength)
                let hidden = try runEmbedding(config: endpointConfig, inputIDs: inputWindow.values, sequenceLength: sequenceLength, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                summary = "OK: hidden \(hidden.shape)"
            case .decoderOnly:
                let hidden = try makeHidden(seqLength: sequenceLength.rawValue)
                let decoded = try runDecoder(
                    layer: DecoderLayerModel(index: 0, name: sequenceLength.decoderName),
                    hidden: hidden,
                    config: decoderConfig,
                    sequenceLength: sequenceLength,
                    positionStart: 0,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: sequenceLength.decoderName),
                    steps: &steps
                )
                summary = "OK: decoded \(decoded.shape)"
            case .decoderStack:
                let hidden = try makeHidden(seqLength: sequenceLength.rawValue)
                let decoded = try runDecoderStack(
                    hidden: hidden,
                    config: decoderConfig,
                    layerSelection: layerSelection,
                    sequenceLength: sequenceLength,
                    positionStart: 0,
                    cacheClearPolicy: cacheClearPolicy,
                    steps: &steps
                )
                summary = "OK: decoded stack \(decoded.shape)"
            case .lmHeadOnly:
                let logits = try runLMHead(hidden: try makeLastHidden(), config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let top = topLogitSummary(logits)
                recordStep("Top logits", detail: top, steps: &steps)
                summary = "OK: \(top)"
            case .fullSequential:
                let inputWindow = try selectedInputWindow(overrideText: inputIDsText, sequenceLength: sequenceLength)
                let hidden = try runEmbedding(config: endpointConfig, inputIDs: inputWindow.values, sequenceLength: sequenceLength, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let decoded = try runDecoder(
                    layer: DecoderLayerModel(index: 0, name: sequenceLength.decoderName),
                    hidden: hidden,
                    config: decoderConfig,
                    sequenceLength: sequenceLength,
                    positionStart: inputWindow.positionStart,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: sequenceLength.decoderName),
                    steps: &steps
                )
                let lastHidden = try copyLastToken(from: decoded, sequenceLength: sequenceLength)
                let logits = try runLMHead(hidden: lastHidden, config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let top = topLogitSummary(logits)
                recordStep("Top logits", detail: top, steps: &steps)
                summary = "OK: \(top)"
            case .fullStackSequential:
                let inputWindow = try selectedInputWindow(overrideText: inputIDsText, sequenceLength: sequenceLength)
                let hidden = try runEmbedding(config: endpointConfig, inputIDs: inputWindow.values, sequenceLength: sequenceLength, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let decoded = try runDecoderStack(
                    hidden: hidden,
                    config: decoderConfig,
                    layerSelection: layerSelection,
                    sequenceLength: sequenceLength,
                    positionStart: inputWindow.positionStart,
                    cacheClearPolicy: cacheClearPolicy,
                    steps: &steps
                )
                let lastHidden = try copyLastToken(from: decoded, sequenceLength: sequenceLength)
                let logits = try runLMHead(hidden: lastHidden, config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let top = topLogitSummary(logits)
                recordStep("Top logits", detail: top, steps: &steps)
                summary = "OK: \(top)"
            case .generateOneToken:
                let token = try runGenerateOneToken(
                    endpointConfig: endpointConfig,
                    decoderConfig: decoderConfig,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
                    inputIDsText: inputIDsText,
                    steps: &steps
                )
                summary = "OK: next token #\(token.index) \(String(format: "%.3f", token.logit))"
            case .generateTokenLoop:
                let predictions = try runGenerateTokenLoop(
                    endpointConfig: endpointConfig,
                    decoderConfig: decoderConfig,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
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

            recordPeakMemory(steps: &steps)
            print("[CoreMLProbe] run finished \(summary)")
            return .success(ProbeReport(steps: steps, summary: summary))
        } catch {
            clearCoreMLRuntimeCache(reason: "error cleanup", steps: &steps)
            recordStep("Error", detail: String(describing: error), steps: &steps)
            recordPeakMemory(steps: &steps)
            print("[CoreMLProbe] run failed: \(String(describing: error))")
            return .failure(ProbeFailure(steps: steps, message: String(describing: error)))
        }
    }

    private static func makeConfig(_ selection: ProbeComputeSelection) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = selection.units
        return config
    }

    private static func validateComputePlan(_ computePlan: ProbeComputePlan, mode: ProbeRunMode) throws {
        guard mode.usesEndpointModels else { return }
        switch computePlan.endpoint {
        case .cpuOnly, .cpuAndGPU:
            return
        case .all, .cpuAndNeuralEngine:
            throw ProbeError.unsafeComputeConfiguration(
                "Endpoint \(computePlan.endpoint.title) is disabled for modes that load embedding or LM head because endpoint All exceeded the iPhone high-water memory limit during MLModel load. Use endpoint CPU with decoder CPU+GPU."
            )
        }
    }

    private static func recordStep(_ name: String, seconds: Double? = nil, detail: String = "", steps: inout [ProbeStep]) {
        appendStep(ProbeStep(name: name, seconds: seconds, memoryMB: ProbeMemory.currentMB(), detail: detail), to: &steps)
    }

    private static func recordPeakMemory(steps: inout [ProbeStep]) {
        let peak = steps.map(\.memoryMB).filter { $0 >= 0 }.max() ?? -1
        guard peak >= 0 else { return }
        appendStep(ProbeStep(name: "Peak memory", seconds: nil, memoryMB: peak, detail: "max observed during run"), to: &steps)
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

    private static func runLoadLMHeadOnly(
        config: MLModelConfiguration,
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws {
        let loadedName = try autoreleasepool {
            let loaded = try loadLMHead(config: config, steps: &steps)
            recordStep("Loaded \(loaded.name)", detail: "leaving autorelease scope", steps: &steps)
            return loaded.name
        }
        recordStep("Released \(loadedName)", detail: "ARC scope exited", steps: &steps)
        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released \(loadedName)", steps: &steps)
        }
    }

    private static func runEmbedding(
        config: MLModelConfiguration,
        inputIDs: [Int32],
        sequenceLength: ProbeSequenceLength,
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        recordStep("Prompt IDs", detail: inputIDs.map(String.init).joined(separator: ","), steps: &steps)

        let hidden = try autoreleasepool {
            let embedding = try loadModel(named: sequenceLength.embeddingName, config: config, steps: &steps)
            return try predictEmbedding(model: embedding, inputIDs: inputIDs, name: "Embedding", steps: &steps)
        }
        recordStep("Released \(sequenceLength.embeddingName)", detail: "hidden retained", steps: &steps)
        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released \(sequenceLength.embeddingName)", steps: &steps)
        }
        return hidden
    }

    private static func runGenerateOneToken(
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        sequenceLength: ProbeSequenceLength,
        inputIDsText: String?,
        steps: inout [ProbeStep]
    ) throws -> (index: Int, logit: Float) {
        let inputWindow = try selectedInputWindow(overrideText: inputIDsText, sequenceLength: sequenceLength)
        let hidden = try runEmbedding(config: endpointConfig, inputIDs: inputWindow.values, sequenceLength: sequenceLength, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let decoded = try runDecoderStack(
            hidden: hidden,
            config: decoderConfig,
            layerSelection: layerSelection,
            sequenceLength: sequenceLength,
            positionStart: inputWindow.positionStart,
            cacheClearPolicy: cacheClearPolicy,
            steps: &steps
        )
        let lastHidden = try copyLastToken(from: decoded, sequenceLength: sequenceLength)
        let logits = try runLMHead(hidden: lastHidden, config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let token = try topLogit(logits)
        recordStep(
            "Next token",
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))",
            steps: &steps
        )
        return token
    }

    private static func runGenerateTokenLoop(
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        sequenceLength: ProbeSequenceLength,
        inputIDsText: String?,
        generatedTokenCount: Int?,
        steps: inout [ProbeStep]
    ) throws -> [TokenPrediction] {
        var inputWindow = try selectedInputWindow(overrideText: inputIDsText, sequenceLength: sequenceLength)
        let tokenCount = try selectedGeneratedTokenCount(override: generatedTokenCount)
        recordStep(
            "Generation loop",
            detail: "tokens=\(tokenCount) seq=\(sequenceLength.rawValue) strategy=reuse embedding+lm-head reload decoder stack cache=\(cacheClearPolicy.rawValue)",
            steps: &steps
        )

        let generationResult = try autoreleasepool { () throws -> (predictions: [TokenPrediction], lmHeadName: String) in
            let embedding = try loadModel(named: sequenceLength.embeddingName, config: endpointConfig, steps: &steps)
            let lmHead = try loadLMHead(config: endpointConfig, steps: &steps)
            var localPredictions: [TokenPrediction] = []

            for step in 1...tokenCount {
                let tokenStart = Date()
                let prediction = try autoreleasepool { () throws -> TokenPrediction in
                    recordStep(
                        "Prompt IDs \(step)",
                        detail: "\(formatInputIDs(inputWindow.values)) positionStart=\(inputWindow.positionStart)",
                        steps: &steps
                    )
                    if let repeatedToken = repeatedToken(in: inputWindow.values) {
                        recordStep(
                            "Repeated input window \(step)",
                            detail: "#\(repeatedToken) repeated across all \(sequenceLength.rawValue) positions",
                            steps: &steps
                        )
                    }
                    let hidden = try predictEmbedding(
                        model: embedding,
                        inputIDs: inputWindow.values,
                        name: "Embedding token \(step)",
                        steps: &steps
                    )
                    let decoded = try runDecoderStack(
                        hidden: hidden,
                        config: decoderConfig,
                        layerSelection: layerSelection,
                        sequenceLength: sequenceLength,
                        positionStart: inputWindow.positionStart,
                        cacheClearPolicy: cacheClearPolicy,
                        steps: &steps
                    )
                    let lastHidden = try copyLastToken(from: decoded, sequenceLength: sequenceLength)
                    let logits = try predictLMHead(
                        model: lmHead.model,
                        hidden: lastHidden,
                        name: "LM head token \(step)",
                        steps: &steps
                    )
                    let token = try topLogit(logits)
                    recordStep(
                        "Top logits token \(step)",
                        detail: topLogitsSummary(logits, count: 5),
                        steps: &steps
                    )
                    recordStep(
                        "Generated token \(step)",
                        detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))",
                        steps: &steps
                    )
                    return TokenPrediction(step: step, index: token.index, logit: token.logit)
                }
                appendStep(ProbeStep(
                    name: "Token \(step) total",
                    seconds: Date().timeIntervalSince(tokenStart),
                    memoryMB: ProbeMemory.currentMB(),
                    detail: "#\(prediction.index) logit=\(String(format: "%.3f", prediction.logit))"
                ), to: &steps)

                if cacheClearPolicy == .afterEachToken {
                    clearCoreMLRuntimeCache(reason: "generated token \(step) policy=\(cacheClearPolicy.rawValue)", steps: &steps)
                }

                guard let nextToken = Int32(exactly: prediction.index) else {
                    throw ProbeError.invalidInputIDs("generated token does not fit Int32: \(prediction.index)")
                }
                inputWindow.values.removeFirst()
                inputWindow.values.append(nextToken)
                inputWindow.positionStart += 1
                localPredictions.append(prediction)
            }

            return (localPredictions, lmHead.name)
        }

        let predictions = generationResult.predictions
        recordStep("Released generation endpoints", detail: "\(sequenceLength.embeddingName), \(generationResult.lmHeadName)", steps: &steps)
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
        sequenceLength: ProbeSequenceLength,
        positionStart: Int,
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let plan = try decoderLayerPlan(selection: layerSelection, sequenceLength: sequenceLength)
        recordStep("Decoder stack", detail: plan.detail, steps: &steps)

        var current = hidden
        for (offset, layer) in plan.layers.enumerated() {
            let layerPosition = offset + 1
            current = try runDecoder(
                layer: layer,
                hidden: current,
                config: config,
                sequenceLength: sequenceLength,
                positionStart: positionStart,
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
        sequenceLength: ProbeSequenceLength,
        positionStart: Int,
        cacheClearReason: String?,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let decoded = try autoreleasepool {
            let decoder = try loadModel(named: layer.name, config: config, steps: &steps)
            let positionIDs = try makePositionIDs(seqLength: sequenceLength.rawValue, start: positionStart)
            let mask = try makeCausalMask(seqLength: sequenceLength.rawValue)
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

    private static func decoderLayerPlan(selection: ProbeLayerSelection, sequenceLength: ProbeSequenceLength) throws -> DecoderLayerPlan {
        var byIndex: [Int: DecoderLayerModel] = [:]
        let urls = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: "Models") ?? []

        for url in urls {
            let name = url.deletingPathExtension().lastPathComponent
            guard name.hasPrefix(decoderPrefix), name.hasSuffix(sequenceLength.decoderSuffix) else {
                continue
            }

            let start = name.index(name.startIndex, offsetBy: decoderPrefix.count)
            let end = name.index(name.endIndex, offsetBy: -sequenceLength.decoderSuffix.count)
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
            throw ProbeError.missingModel("\(decoderPrefix)*\(sequenceLength.decoderSuffix).mlmodelc")
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
        let result = try autoreleasepool { () throws -> (logits: MLMultiArray, name: String) in
            let lmHead = try loadLMHead(config: config, steps: &steps)
            let logits = try predictLMHead(model: lmHead.model, hidden: hidden, name: "LM head", steps: &steps)
            return (logits, lmHead.name)
        }
        recordStep("Released \(result.name)", detail: "logits retained", steps: &steps)
        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released \(result.name)", steps: &steps)
        }
        return result.logits
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

    private static func loadLMHead(config: MLModelConfiguration, steps: inout [ProbeStep]) throws -> LoadedProbeModel {
        for name in lmHeadNames {
            guard let url = modelURL(named: name) else {
                continue
            }

            if name == lmHeadName {
                recordStep(
                    "LM head target",
                    detail: "using preferred norm+lm_head endpoint with final norm/softcap",
                    steps: &steps
                )
            } else {
                recordStep(
                    "LM head fallback",
                    detail: "missing \(lmHeadName); using legacy \(name) without final norm/softcap",
                    steps: &steps
                )
            }
            return LoadedProbeModel(
                model: try loadModel(named: name, url: url, config: config, steps: &steps),
                name: name
            )
        }

        throw ProbeError.missingModel(lmHeadNames.joined(separator: " or "))
    }

    private static func loadModel(named name: String, config: MLModelConfiguration, steps: inout [ProbeStep]) throws -> MLModel {
        guard let url = modelURL(named: name) else {
            throw ProbeError.missingModel(name)
        }

        return try loadModel(named: name, url: url, config: config, steps: &steps)
    }

    private static func modelURL(named name: String) -> URL? {
        Bundle.main.url(forResource: name, withExtension: "mlmodelc", subdirectory: "Models")
    }

    private static func loadModel(
        named name: String,
        url: URL,
        config: MLModelConfiguration,
        steps: inout [ProbeStep]
    ) throws -> MLModel {
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

    private static func selectedInputWindow(
        overrideText: String?,
        sequenceLength: ProbeSequenceLength
    ) throws -> TokenWindow {
        guard let rawValue = cleanedOverride(overrideText) ?? processInputIDsText() else {
            return TokenWindow(values: sequenceLength.defaultInputIDs, positionStart: 0)
        }

        let windowText = tokenWindowText(from: rawValue, sequenceLength: sequenceLength)
        let normalized = windowText
            .replacingOccurrences(of: "#", with: "")
            .replacingOccurrences(of: "[", with: " ")
            .replacingOccurrences(of: "]", with: " ")
            .replacingOccurrences(of: ",", with: " ")
        let values = normalized.split(whereSeparator: \.isWhitespace).map(String.init)
        guard values.count >= sequenceLength.rawValue else {
            throw ProbeError.invalidInputIDs("expected at least \(sequenceLength.rawValue) comma-separated token IDs, got \(values.count)")
        }

        let parsed = try values.map { value in
            guard let parsed = Int32(value) else {
                throw ProbeError.invalidInputIDs("not an Int32 token ID: \(value)")
            }
            return parsed
        }
        let startIndex = parsed.count - sequenceLength.rawValue
        let window = Array(parsed[startIndex...])
        return TokenWindow(values: window, positionStart: startIndex)
    }

    private static func tokenWindowText(from rawValue: String, sequenceLength: ProbeSequenceLength) -> String {
        let keys = [
            "input_ids_last\(sequenceLength.rawValue)=",
            "input_ids=",
            "input_ids_last4="
        ]
        for key in keys {
            guard let range = rawValue.range(of: key) else {
                continue
            }
            let tail = rawValue[range.upperBound...]
            return tail.split(whereSeparator: \.isNewline).first.map(String.init) ?? String(tail)
        }
        return rawValue
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

    private static func repeatedToken(in inputIDs: [Int32]) -> Int32? {
        guard let first = inputIDs.first,
              inputIDs.dropFirst().allSatisfy({ $0 == first }) else {
            return nil
        }
        return first
    }

    private static func makeInputIDs(values: [Int32]) throws -> MLMultiArray {
        guard !values.isEmpty else {
            throw ProbeError.invalidInputIDs("expected token IDs, got 0")
        }

        let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for index in 0..<values.count {
            pointer[index] = values[index]
        }
        return array
    }

    private static func makePositionIDs(seqLength: Int, start: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: seqLength)], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for index in 0..<array.count {
            pointer[index] = Int32(start + index)
        }
        return array
    }

    private static func makeCausalMask(seqLength: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: seqLength), NSNumber(value: seqLength)],
            dataType: .float16
        )
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        for index in 0..<array.count {
            pointer[index] = 0
        }
        for row in 0..<seqLength {
            for column in 0..<seqLength where column > row {
                pointer[row * seqLength + column] = Float16(-65504.0)
            }
        }
        return array
    }

    private static func copyLastToken(from decoded: MLMultiArray, sequenceLength: ProbeSequenceLength) throws -> MLMultiArray {
        guard decoded.dataType == .float16, decoded.count >= sequenceLength.rawValue * 3840 else {
            throw ProbeError.unexpectedShape("decoder output \(decoded.shape)")
        }

        let array = try MLMultiArray(shape: [1, 1, 3840], dataType: .float16)
        let source = decoded.dataPointer.bindMemory(to: Float16.self, capacity: decoded.count)
        let destination = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        let sourceOffset = (sequenceLength.rawValue - 1) * 3840
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

    private static func topLogitsSummary(_ logits: MLMultiArray, count: Int) -> String {
        guard let tokens = try? topLogits(logits, count: count) else {
            return "dtype \(logits.dataType.rawValue) shape \(logits.shape)"
        }

        return tokens
            .map { "#\($0.index) \(String(format: "%.3f", $0.logit))" }
            .joined(separator: ", ")
    }

    private static func topLogit(_ logits: MLMultiArray) throws -> (index: Int, logit: Float) {
        try topLogits(logits, count: 1)[0]
    }

    private static func topLogits(_ logits: MLMultiArray, count: Int) throws -> [(index: Int, logit: Float)] {
        guard logits.dataType == .float16 else {
            throw ProbeError.unexpectedShape("logits dtype \(logits.dataType.rawValue), shape \(logits.shape)")
        }

        let pointer = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
        var best: [(index: Int, logit: Float)] = []
        for index in 0..<logits.count {
            let value = Float(pointer[index])
            if best.count < count {
                best.append((index, value))
                best.sort { $0.logit > $1.logit }
            } else if let last = best.last, value > last.logit {
                best.removeLast()
                best.append((index, value))
                best.sort { $0.logit > $1.logit }
            }
        }
        return best
    }
}

enum ProbeError: LocalizedError {
    case missingModel(String)
    case missingOutput(String)
    case invalidInputIDs(String)
    case invalidTokenCount(String)
    case unexpectedShape(String)
    case unsafeComputeConfiguration(String)

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
        case .unsafeComputeConfiguration(let detail):
            "Unsafe compute configuration: \(detail)"
        }
    }
}
