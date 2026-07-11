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
        .cpuAndGPU,
        .all
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
    case seq32 = 32
    case seq64 = 64
    case seq320 = 320

    var id: Int { rawValue }

    var title: String {
        "Seq \(rawValue)"
    }

    var embeddingName: String {
        // Fall back to the int4 embedding when the requested variant asset is
        // not bundled. The pal4 embedding is deliberately not shipped: its ANE
        // compilation decompresses the 262k-vocab LUT table and the transient
        // spike jetsams the app, while the int4/CPU embedding predict is
        // already ~4ms and irrelevant to per-token latency.
        let variantName = "gemma4_12b_embedding_seq\(rawValue)_\(ProbeSequenceLength.endpointVariant)"
        if ProbeSequenceLength.modelExists(named: variantName) {
            return variantName
        }
        return "gemma4_12b_embedding_seq\(rawValue)_int4_block32"
    }

    var decoderName: String {
        "gemma4_12b_layer00_decoder_seq\(rawValue)_mask_\(ProbeSequenceLength.decoderVariant)"
    }

    var decoderSuffix: String {
        "_decoder_seq\(rawValue)_mask_\(ProbeSequenceLength.decoderVariant)"
    }

    /// Weight-compression variant tag in decoder bundle names. When the pal4
    /// (ANE-executable) decoder assets are bundled they are the default — this
    /// is the measured-fast configuration (all decoder layers resident on the
    /// ANE, ~1.8s/token vs 12s for int4/CPU). Env/argument overrides still win
    /// so the A/B automation keeps working.
    static let decoderVariant: String = {
        processVariant(
            environmentKey: "COREML_PROBE_DECODER_VARIANT",
            argumentPrefix: "--decoder-variant=",
            autoDetectName: "gemma4_12b_layers00_03_decoder_seq64_mask_pal4_g16"
        )
    }()

    /// Same idea for the embedding / norm+lm_head endpoints, independent of the
    /// decoder variant so mixed configurations (pal4 decoder + int4 endpoints)
    /// remain testable. The embedding itself always falls back to int4/CPU.
    static let endpointVariant: String = {
        processVariant(
            environmentKey: "COREML_PROBE_ENDPOINT_VARIANT",
            argumentPrefix: "--endpoint-variant=",
            autoDetectName: "gemma4_12b_norm_lm_head_1tok_pal4_g16"
        )
    }()

    private static func processVariant(environmentKey: String, argumentPrefix: String, autoDetectName: String) -> String {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment[environmentKey], !value.isEmpty {
            return value
        }
        if let argument = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix(argumentPrefix) }) {
            let value = String(argument.dropFirst(argumentPrefix.count))
            if !value.isEmpty {
                return value
            }
        }
        if modelExists(named: autoDetectName) {
            return "pal4_g16"
        }
        return "int4_block32"
    }

    /// True when the app is running the auto-detected pal4/ANE stack.
    static var usesPal4Stack: Bool {
        decoderVariant == "pal4_g16"
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
        case .seq32:
            [
                2, 105, 2364, 107, 85141, 236924, 238906, 237234, 7604, 31600,
                3335, 236924, 94951, 237007, 239309, 241910, 237000, 236951,
                239375, 221357, 117495, 109943, 236924, 106, 107, 105,
                4368, 107, 100, 45518, 107, 101
            ]
        case .seq64:
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
                2, 105, 2364, 107, 85141, 236924, 238906, 237234, 7604, 31600,
                3335, 236924, 94951, 237007, 239309, 241910, 18794, 36976,
                11914, 236924, 106, 107, 105, 4368, 107, 100, 45518, 107, 101
            ]
        case .seq320:
            [Int32](repeating: 0, count: 320 - 29) + [
                2, 105, 2364, 107, 85141, 236924, 238906, 237234, 7604, 31600,
                3335, 236924, 94951, 237007, 239309, 241910, 18794, 36976,
                11914, 236924, 106, 107, 105, 4368, 107, 100, 45518, 107, 101
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
        if modelExists(named: ProbeSequenceLength.seq64.embeddingName) {
            return .seq64
        }
        if modelExists(named: ProbeSequenceLength.seq32.embeddingName) {
            return .seq32
        }
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
        case "seq32", "s32": .seq32
        case "seq64", "s64": .seq64
        default: nil
        }
    }

    fileprivate static func modelExists(named name: String) -> Bool {
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
    case multimodalSmoke = "multimodal-smoke"
    case imageSmoke = "image-smoke"
    case audioSmoke = "audio-smoke"
    case memoryRamp = "memory-ramp"

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
        case .multimodalSmoke: "Multimodal smoke"
        case .imageSmoke: "Image smoke"
        case .audioSmoke: "Audio smoke"
        case .memoryRamp: "Memory ramp"
        }
    }

    var usesInputIDs: Bool {
        switch self {
        case .embeddingOnly, .fullSequential, .fullStackSequential, .generateOneToken, .generateTokenLoop:
            true
        case .loadEmbedding, .loadDecoder, .loadLMHead, .loadAllSequential, .loadDecoderStack, .decoderOnly, .decoderStack, .lmHeadOnly, .multimodalSmoke, .imageSmoke, .audioSmoke, .memoryRamp:
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
                .generateOneToken, .generateTokenLoop, .multimodalSmoke, .imageSmoke, .audioSmoke:
            true
        case .loadDecoder, .loadDecoderStack, .decoderOnly, .decoderStack, .memoryRamp:
            false
        }
    }

    var usesDecoderModels: Bool {
        switch self {
        case .loadDecoder, .loadAllSequential, .loadDecoderStack, .decoderOnly,
                .decoderStack, .fullSequential, .fullStackSequential,
                .generateOneToken, .generateTokenLoop, .multimodalSmoke, .imageSmoke, .audioSmoke, .memoryRamp:
            true
        case .loadEmbedding, .loadLMHead, .embeddingOnly, .lmHeadOnly:
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
    let startIndex: Int
    let endIndex: Int
    let name: String

    var layerCount: Int {
        endIndex - startIndex + 1
    }

    var title: String {
        if startIndex == endIndex {
            return "\(startIndex)"
        }
        return "\(startIndex)-\(endIndex)"
    }
}

struct DecoderLayerPlan {
    let availableCount: Int
    let layers: [DecoderLayerModel]
    let selection: ProbeLayerSelection

    var detail: String {
        let requested = selection.requestedCount.map(String.init) ?? "all"
        return "selected=\(selectedLayerCount), models=\(layers.count), available=\(availableCount), requested=\(requested)"
    }

    var selectedLayerCount: Int {
        layers.reduce(0) { $0 + $1.layerCount }
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
    var leftPadCount: Int = 0

    var seqLength: Int {
        values.count
    }
}

struct LoadedProbeModel {
    let model: MLModel
    let name: String
}

struct RetainedDecoderModels {
    let models: [String: MLModel]
    let names: [String]
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

    /// Bytes remaining before the process hits its jetsam (memory) limit,
    /// reported by the kernel via `os_proc_available_memory()`. Returns a
    /// negative value when the process is not memory-limited (e.g. attached to
    /// the debugger, or the entitlement/limit is not in effect), in which case
    /// the ceiling cannot be measured and the ramp should not trust it.
    static func availableMB() -> Double {
        let bytes = os_proc_available_memory()
        guard bytes > 0 else { return -1 }
        return Double(bytes) / 1_048_576.0
    }

    static func availableText() -> String {
        let value = availableMB()
        guard value >= 0 else { return "unknown" }
        return String(format: "%.1f MB", value)
    }
}

enum ProbeRunner {
    private static let decoderPrefix = "gemma4_12b_layer"
    private static let decoderChunkPrefix = "gemma4_12b_layers"
    private static let lmHeadName = "gemma4_12b_norm_lm_head_1tok_\(ProbeSequenceLength.endpointVariant)"
    private static let defaultLMHeadName = "gemma4_12b_norm_lm_head_1tok_int4_block32"
    private static let legacyLMHeadName = "gemma4_12b_lm_head_1tok_int4_block32"
    private static let lmHeadNames: [String] = {
        var names = [lmHeadName]
        if !names.contains(defaultLMHeadName) {
            names.append(defaultLMHeadName)
        }
        names.append(legacyLMHeadName)
        return names
    }()
    private static let imageEmbedderName = "gemma4_12b_image_embedder_patches32_int4_block32"
    private static let imageEmbedder256Name = "gemma4_12b_image_embedder_patches256_int4_block32"
    private static let imageSmokePatchCount = 32
    private static let imageSmokePatchDim = 48 * 48 * 3
    private static let audioEmbedderName = "gemma4_12b_audio_embedder_tokens32_int4_block32"
    private static let audioSmokeTokenCount = 32
    private static let audioSmokeFeatureDim = 640
    static let defaultGeneratedTokenCount = 2
    static let defaultChatGeneratedTokenCount = 32
    static let maxGeneratedTokenCount = 256
    /// Sentinel meaning "let the model decide": run until it emits <eos> /
    /// <end_of_turn>, bounded only by `autoGeneratedTokenCap` so a
    /// non-terminating generation cannot run forever (~2s/token).
    static let autoGeneratedTokenCount = 0
    // Kept short on purpose: with Seq64 and no KV cache the prompt slides out of
    // the window after a few dozen tokens and the model degenerates into
    // repetition. Capping auto near that boundary keeps replies readable until
    // KV cache + a longer window land.
    static let autoGeneratedTokenCap = 40
    static let defaultRetainedDecoderModelCount = 0
    // Raised from 1: keeping only a single chunk resident saved just one of the
    // ~12 per-token reloads (no measurable speedup, per earlier device runs).
    // The ramp probe showed each resident 4-layer chunk costs ~495 MB and that
    // resident prediction is ~14x cheaper than reloading, so the win only
    // appears when many chunks stay resident. loadRetainedDecoderModels stops
    // early if the process runs low on memory, so requesting more than fits
    // degrades gracefully. 36 covers the pal4 all-ANE plan: 4 sliding-only
    // 4-layer chunks + 32 single-layer models (full-attention layers only
    // compile for the ANE as singles).
    static let maxRetainedDecoderModelCount = 36
    static let defaultMemoryMarginMB = 500
    private static let maxSupportedDecoderBundleLayers = 4
    private static let generatedStopTokenIDs: Set<Int> = [1, 106]
    private static let repeatedGeneratedTokenLimit = 4

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

    static func selectedGeneratedTokenCountFromProcess(default fallback: Int = defaultGeneratedTokenCount) -> Int {
        guard let rawValue = processGeneratedTokenCountText(),
              let count = Int(rawValue),
              (autoGeneratedTokenCount...maxGeneratedTokenCount).contains(count) else {
            return fallback
        }
        return count
    }

    /// The stable measured configuration on iPhone 17 (8GB): 24 resident ANE
    /// decoder models + 12 transiently reloaded singles per token ≈ 1.8s/token.
    /// Retaining more coexists poorly with the lm_head under system-wide
    /// memory pressure; retaining fewer wastes reload time.
    static var recommendedRetainedDecoderModelCount: Int {
        guard ProbeSequenceLength.usesPal4Stack else { return defaultRetainedDecoderModelCount }
        // Residency only pays off on the newer, larger Neural Engines. On A19
        // (iPhone18,x) holding 24 pal4 models gives ~2s/token; on the A15
        // (iPhone14,x) the ANE saturates and per-token slows to ~14s, so
        // reload-per-token (retain 0, ~4.3s) is actually faster there. Devices
        // newer than the tested A19 default to residency; older/unknown ones
        // fall back to reload-per-token.
        return deviceMajorVersion >= 18 ? 24 : defaultRetainedDecoderModelCount
    }

    /// The numeric prefix of the hardware identifier ("iPhone18,3" → 18).
    static let deviceMajorVersion: Int = {
        var info = utsname()
        uname(&info)
        let identifier = withUnsafeBytes(of: &info.machine) { raw -> String in
            let bytes = raw.prefix { $0 != 0 }
            return String(decoding: bytes, as: UTF8.self)
        }
        guard identifier.hasPrefix("iPhone") else { return 0 }
        let digits = identifier.dropFirst("iPhone".count).prefix { $0.isNumber }
        return Int(digits) ?? 0
    }()

    static func selectedRetainedDecoderModelCountFromProcess() -> Int {
        guard let rawValue = processRetainedDecoderModelCountText(),
              let count = Int(rawValue),
              (defaultRetainedDecoderModelCount...maxRetainedDecoderModelCount).contains(count) else {
            return recommendedRetainedDecoderModelCount
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
        generatedTokenCount: Int? = nil,
        retainedDecoderModelCount: Int? = nil,
        persistent: Bool = false,
        onToken: ((Int) -> Void)? = nil
    ) -> Result<ProbeReport, ProbeFailure> {
        run(
            computePlan: .shared(computeSelection),
            mode: mode,
            layerSelection: layerSelection,
            cacheClearPolicy: cacheClearPolicy,
            sequenceLength: sequenceLength,
            inputIDsText: inputIDsText,
            generatedTokenCount: generatedTokenCount,
            retainedDecoderModelCount: retainedDecoderModelCount,
            persistent: persistent,
            onToken: onToken
        )
    }

    static func run(
        computePlan: ProbeComputePlan,
        mode: ProbeRunMode,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy = .afterEveryModel,
        sequenceLength: ProbeSequenceLength = selectedSequenceLengthFromProcess(),
        inputIDsText: String? = nil,
        generatedTokenCount: Int? = nil,
        retainedDecoderModelCount: Int? = nil,
        imageHidden: MLMultiArray? = nil,
        imageStartPosition: Int = -1,
        imageBlockBidirectional: Bool = false,
        persistent: Bool = false,
        onToken: ((Int) -> Void)? = nil
    ) -> Result<ProbeReport, ProbeFailure> {
        var steps: [ProbeStep] = []

        do {
            resetStepFile()
            if let pendingImageStatsLine {
                appendToStepFile(pendingImageStatsLine)
                Self.pendingImageStatsLine = nil
            }
            let startLine = "[CoreMLProbe] run started compute=\(computePlan.logDetail) mode=\(mode.rawValue) seq=\(sequenceLength.rawValue) layers=\(layerSelection.rawValue) cache=\(cacheClearPolicy.rawValue)"
            print(startLine)
            appendToStepFile(startLine)
            recordStep("Start", detail: "\(computePlan.logDetail), \(mode.title), \(sequenceLength.title), \(layerSelection.title), cache=\(cacheClearPolicy.title)", steps: &steps)
            clearCoreMLRuntimeCache(reason: "run start", steps: &steps)
            try validateComputePlan(computePlan, mode: mode)
            // The KV path uses its own prefill/decode bundles, not the
            // per-layer seq320 decoder singles the standard plan validates.
            let usesKVPath = mode == .generateTokenLoop && sequenceLength == .seq320 && kvChatAvailable
            if !usesKVPath {
                try validateDecoderBundlePlan(mode: mode, layerSelection: layerSelection, sequenceLength: sequenceLength)
            }

            let endpointConfig = makeConfig(computePlan.endpoint)
            let decoderConfig = makeConfig(computePlan.decoder)

            let summary: String
            switch mode {
            case .loadEmbedding:
                try runLoadOnly(
                    named: sequenceLength.embeddingName,
                    config: makeEmbeddingConfig(),
                    cacheClearReason: cacheClearPolicy.clearsAfterNonDecoderRelease ? "released \(sequenceLength.embeddingName)" : nil,
                    steps: &steps
                )
                summary = "OK: loaded embedding"
            case .loadDecoder:
                let decoderToLoad = try warmModelName() ?? warmChunkName(sequenceLength: sequenceLength) ?? sequenceLength.decoderName
                try runLoadOnly(
                    named: decoderToLoad,
                    config: decoderConfig,
                    cacheClearReason: cacheClearPolicy.decoderReleaseReason(layerPosition: 1, totalLayers: 1, layerName: decoderToLoad),
                    steps: &steps
                )
                summary = "OK: loaded \(decoderToLoad)"
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
                    config: makeEmbeddingConfig(),
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
                summary = "OK: loaded \(plan.selectedLayerCount) decoder layers in \(plan.layers.count) model(s)"
            case .embeddingOnly:
                let inputWindow = try selectedInputWindow(overrideText: inputIDsText, sequenceLength: sequenceLength)
                let hidden = try runEmbedding(inputIDs: inputWindow.values, sequenceLength: sequenceLength, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                summary = "OK: hidden \(hidden.shape)"
            case .decoderOnly:
                let hidden = try makeHidden(seqLength: sequenceLength.rawValue)
                let decoded = try runDecoder(
                    layer: DecoderLayerModel(startIndex: 0, endIndex: 0, name: sequenceLength.decoderName),
                    hidden: hidden,
                    config: decoderConfig,
                    sequenceLength: sequenceLength,
                    positionStart: 0,
                    leftPadCount: 0,
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
                    leftPadCount: 0,
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
                let hidden = try runEmbedding(inputIDs: inputWindow.values, sequenceLength: sequenceLength, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let decoded = try runDecoder(
                    layer: DecoderLayerModel(startIndex: 0, endIndex: 0, name: sequenceLength.decoderName),
                    hidden: hidden,
                    config: decoderConfig,
                    sequenceLength: sequenceLength,
                    positionStart: inputWindow.positionStart,
                    leftPadCount: inputWindow.leftPadCount,
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
                let hidden = try runEmbedding(inputIDs: inputWindow.values, sequenceLength: sequenceLength, cacheClearPolicy: cacheClearPolicy, steps: &steps)
                let decoded = try runDecoderStack(
                    hidden: hidden,
                    config: decoderConfig,
                    layerSelection: layerSelection,
                    sequenceLength: sequenceLength,
                    positionStart: inputWindow.positionStart,
                    leftPadCount: inputWindow.leftPadCount,
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
                // Prefer the KV-cache path when its 320-prefill / 512-capacity
                // assets are bundled and the run uses the Seq320 window (image
                // chat, or text explicitly routed to Seq320). It shares one
                // chunked prefill and then runs seq-1 decode steps against the
                // caches, unlocking longer replies than the fixed window.
                let predictions: [TokenPrediction]
                if sequenceLength == .seq320 && kvChatAvailable {
                    predictions = try runKVGeneration(
                        endpointConfig: endpointConfig,
                        decoderConfig: decoderConfig,
                        layerSelection: layerSelection,
                        inputIDsText: inputIDsText,
                        generatedTokenCount: generatedTokenCount,
                        retainedDecoderModelCount: retainedDecoderModelCount,
                        imageHidden: imageHidden,
                        imageStartPosition: imageStartPosition,
                        imageBlockBidirectional: imageBlockBidirectional,
                        persistent: persistent,
                        onToken: onToken,
                        steps: &steps
                    )
                } else {
                    predictions = try runGenerateTokenLoop(
                        endpointConfig: endpointConfig,
                        decoderConfig: decoderConfig,
                        layerSelection: layerSelection,
                        cacheClearPolicy: cacheClearPolicy,
                        sequenceLength: sequenceLength,
                        inputIDsText: inputIDsText,
                        generatedTokenCount: generatedTokenCount,
                        retainedDecoderModelCount: retainedDecoderModelCount,
                        imageHidden: imageHidden,
                        imageStartPosition: imageStartPosition,
                        imageBlockBidirectional: imageBlockBidirectional,
                        persistent: persistent,
                        onToken: onToken,
                        steps: &steps
                    )
                }
                let tokens = predictions.map { "#\($0.index)" }.joined(separator: ",")
                summary = "OK: generated \(predictions.count) tokens \(tokens)"
            case .multimodalSmoke:
                let token = try runMultimodalSmoke(
                    endpointConfig: endpointConfig,
                    decoderConfig: decoderConfig,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
                    steps: &steps
                )
                summary = "OK: multimodal smoke token #\(token.index) \(String(format: "%.3f", token.logit))"
            case .imageSmoke:
                let token = try runImageSmoke(
                    endpointConfig: endpointConfig,
                    decoderConfig: decoderConfig,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
                    steps: &steps
                )
                summary = "OK: image smoke token #\(token.index) \(String(format: "%.3f", token.logit))"
            case .audioSmoke:
                let token = try runAudioSmoke(
                    endpointConfig: endpointConfig,
                    decoderConfig: decoderConfig,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
                    steps: &steps
                )
                summary = "OK: audio smoke token #\(token.index) \(String(format: "%.3f", token.logit))"
            case .memoryRamp:
                summary = try runMemoryRamp(
                    decoderConfig: decoderConfig,
                    layerSelection: layerSelection,
                    sequenceLength: sequenceLength,
                    steps: &steps
                )
            }

            if cacheClearPolicy.clearsAtRunEnd {
                clearCoreMLRuntimeCache(reason: "run end policy=\(cacheClearPolicy.rawValue)", steps: &steps)
            }

            recordPeakMemory(steps: &steps)
            print("[CoreMLProbe] run finished \(summary)")
            appendToStepFile("[CoreMLProbe] run finished \(summary)")
            return .success(ProbeReport(steps: steps, summary: summary))
        } catch {
            clearCoreMLRuntimeCache(reason: "error cleanup", steps: &steps)
            recordStep("Error", detail: String(describing: error), steps: &steps)
            recordPeakMemory(steps: &steps)
            print("[CoreMLProbe] run failed: \(String(describing: error))")
            appendToStepFile("[CoreMLProbe] run failed: \(String(describing: error))")
            return .failure(ProbeFailure(steps: steps, message: String(describing: error)))
        }
    }

    private static func makeConfig(_ selection: ProbeComputeSelection) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = selection.units
        return config
    }

    /// The embedding endpoint is a token-id gather: ANE/GPU offer no speedup
    /// (CPU predict is ~4ms) and asking the runtime to compile the 262k-vocab
    /// table for the ANE spikes transient memory past even the entitled limit
    /// and jetsams the app. The embedding therefore always loads CPU-only,
    /// regardless of the selected endpoint compute units (which still apply to
    /// the norm+lm_head endpoint).
    private static func makeEmbeddingConfig() -> MLModelConfiguration {
        makeConfig(.cpuOnly)
    }

    /// Decoder chunks whose start layer is listed in `COREML_PROBE_GPU_CHUNKS`
    /// (comma-separated start indices, or `--gpu-chunks=`) load with CPU+GPU
    /// compute units instead of the selected decoder units. Purpose: Gemma4
    /// chunks containing a full_attention layer are rejected by the ANE
    /// compiler (ANECCompile() FAILED) — every load re-attempts that compile
    /// (~30s, ~4.3GB transient) before falling back. Forcing those chunks
    /// straight to the GPU skips the doomed ANE attempt; sliding-attention
    /// chunks stay on the ANE.
    private static let gpuChunkStartIndices: Set<Int> = {
        let environment = ProcessInfo.processInfo.environment
        let prefix = "--gpu-chunks="
        let raw = environment["COREML_PROBE_GPU_CHUNKS"]
            ?? ProcessInfo.processInfo.arguments
                .first(where: { $0.hasPrefix(prefix) })
                .map { String($0.dropFirst(prefix.count)) }
        guard let raw else { return [] }
        return Set(raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
    }()

    private static func decoderLoadConfig(for layer: DecoderLayerModel, base: MLModelConfiguration) -> MLModelConfiguration {
        guard gpuChunkStartIndices.contains(layer.startIndex) else { return base }
        return makeConfig(.cpuAndGPU)
    }

    private static func validateComputePlan(_ computePlan: ProbeComputePlan, mode: ProbeRunMode) throws {
        guard mode.usesEndpointModels else { return }
        // The high-water crash that motivated this guard was observed with the
        // int4 endpoints (BNNS/GPU compile ballooning during load). Palettized
        // endpoint variants target the ANE, so the guard only applies to the
        // default int4 assets.
        guard ProbeSequenceLength.endpointVariant == "int4_block32" else { return }
        switch computePlan.endpoint {
        case .cpuOnly, .cpuAndGPU:
            return
        case .all, .cpuAndNeuralEngine:
            throw ProbeError.unsafeComputeConfiguration(
                "Endpoint \(computePlan.endpoint.title) is disabled for modes that load embedding or LM head because endpoint All exceeded the iPhone high-water memory limit during MLModel load. Use endpoint CPU with decoder CPU+GPU."
            )
        }
    }

    private static func validateDecoderBundlePlan(
        mode: ProbeRunMode,
        layerSelection: ProbeLayerSelection,
        sequenceLength: ProbeSequenceLength
    ) throws {
        guard mode.usesDecoderModels else { return }
        let plan = try decoderLayerPlan(selection: layerSelection, sequenceLength: sequenceLength)
        guard let widest = plan.layers.max(by: { $0.layerCount < $1.layerCount }),
              widest.layerCount > maxSupportedDecoderBundleLayers else {
            return
        }

        throw ProbeError.unsafeComputeConfiguration(
            "Decoder bundle \(widest.name) contains \(widest.layerCount) layers. iPhone execution-plan compilation failed for 8-layer Seq \(sequenceLength.rawValue) bundles; rebuild decoder assets with chunk-size \(maxSupportedDecoderBundleLayers) or smaller."
        )
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
        let line = "[CoreMLProbe] \(step.name) duration=\(step.durationText) memory=\(step.memoryText) detail=\(step.detail)"
        print(line)
        appendToStepFile(line)
    }

    /// Mirrors every step line into Documents/probe-steps.log on device. The
    /// devicectl console stream (Mercury) drops mid-run often enough that
    /// benchmarks lose their tail; the file survives and is fetched afterwards
    /// with `devicectl device copy from`. Truncated at each run start.
    private static let stepFileURL: URL? = FileManager.default
        .urls(for: .documentDirectory, in: .userDomainMask).first?
        .appendingPathComponent("probe-steps.log")

    private static var stepFileHandle: FileHandle?

    /// Tail of the on-device step log, for the HTTP /log endpoint and
    /// external debugging (coding agents fetch this instead of the flaky
    /// devicectl console stream).
    static func stepFileTail(maxLines: Int) -> String {
        guard let url = stepFileURL,
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            return ""
        }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        return lines.suffix(max(1, maxLines)).joined(separator: "\n")
    }

    static func resetStepFile() {
        guard let url = stepFileURL else { return }
        try? stepFileHandle?.close()
        FileManager.default.createFile(atPath: url.path, contents: nil)
        stepFileHandle = try? FileHandle(forWritingTo: url)
    }

    private static func appendToStepFile(_ line: String) {
        guard let data = (line + "\n").data(using: .utf8) else { return }
        if stepFileHandle == nil, let url = stepFileURL {
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            stepFileHandle = try? FileHandle(forWritingTo: url)
            _ = try? stepFileHandle?.seekToEnd()
        }
        try? stepFileHandle?.write(contentsOf: data)
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
        inputIDs: [Int32],
        sequenceLength: ProbeSequenceLength,
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        recordStep("Prompt IDs", detail: inputIDs.map(String.init).joined(separator: ","), steps: &steps)

        let hidden = try autoreleasepool {
            let embedding = try loadModel(named: sequenceLength.embeddingName, config: makeEmbeddingConfig(), steps: &steps)
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
        let hidden = try runEmbedding(inputIDs: inputWindow.values, sequenceLength: sequenceLength, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let decoded = try runDecoderStack(
            hidden: hidden,
            config: decoderConfig,
            layerSelection: layerSelection,
            sequenceLength: sequenceLength,
            positionStart: inputWindow.positionStart,
            leftPadCount: inputWindow.leftPadCount,
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

    /// Embedding + norm/lm_head + retained decoder models for a chat turn.
    struct LoadedChatModels {
        let embedding: MLModel
        let lmHead: LoadedProbeModel
        let retained: RetainedDecoderModels
    }

    /// Models kept resident between chat turns (A: persistent residency). The
    /// pal4 decoders execute on the ANE, so holding them costs almost no app
    /// footprint (~300MB total) while removing the ~18s (warm) / ~150s (cold)
    /// per-turn reload. Rebuilt only when the signature (variant/seq/retain/
    /// compute) changes, and released on background/memory pressure.
    private static var residentChatModels: LoadedChatModels?
    private static var residentChatSignature: String?

    static func releaseResidentChatModels() {
        residentChatModels = nil
        residentChatSignature = nil
    }

    private static func chatModelSignature(
        sequenceLength: ProbeSequenceLength,
        retainCount: Int,
        layerSelection: ProbeLayerSelection,
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration
    ) -> String {
        [
            ProbeSequenceLength.decoderVariant,
            ProbeSequenceLength.endpointVariant,
            "seq\(sequenceLength.rawValue)",
            "retain\(retainCount)",
            "layers\(layerSelection.rawValue)",
            "ep\(endpointConfig.computeUnits.rawValue)",
            "dec\(decoderConfig.computeUnits.rawValue)"
        ].joined(separator: "|")
    }

    private static func ensureResidentChatModels(
        signature: String,
        plan: DecoderLayerPlan,
        retainCount: Int,
        sequenceLength: ProbeSequenceLength,
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        steps: inout [ProbeStep]
    ) throws -> LoadedChatModels {
        if let models = residentChatModels, residentChatSignature == signature {
            recordStep("Resident chat models", detail: "reused signature=\(signature)", steps: &steps)
            return models
        }
        // Drop the previous set first so its allocations free before reloading.
        releaseResidentChatModels()
        recordStep("Resident chat models", detail: "loading signature=\(signature)", steps: &steps)
        let models = LoadedChatModels(
            embedding: try loadModel(named: sequenceLength.embeddingName, config: makeEmbeddingConfig(), steps: &steps),
            lmHead: try loadLMHead(config: endpointConfig, steps: &steps),
            retained: try loadRetainedDecoderModels(
                plan: plan,
                requestedCount: retainCount,
                config: decoderConfig,
                sequenceLength: sequenceLength,
                steps: &steps
            )
        )
        residentChatModels = models
        residentChatSignature = signature
        return models
    }

    private static func runGenerateTokenLoop(
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        sequenceLength: ProbeSequenceLength,
        inputIDsText: String?,
        generatedTokenCount: Int?,
        retainedDecoderModelCount: Int?,
        imageHidden: MLMultiArray? = nil,
        imageStartPosition: Int = -1,
        imageBlockBidirectional: Bool = false,
        persistent: Bool = false,
        onToken: ((Int) -> Void)? = nil,
        steps: inout [ProbeStep]
    ) throws -> [TokenPrediction] {
        var inputWindow = try selectedInputWindow(overrideText: inputIDsText, sequenceLength: sequenceLength)
        let tokenCount = try selectedGeneratedTokenCount(override: generatedTokenCount)
        let retainedDecoderCount = try selectedRetainedDecoderModelCount(override: retainedDecoderModelCount)
        if retainedDecoderCount > 0 && cacheClearPolicy != .runEndOnly {
            throw ProbeError.unsafeComputeConfiguration("Retain Decoders requires Cache = Run end so retained MLModel instances are not mixed with cache deletion.")
        }
        let decoderPlan = try decoderLayerPlan(selection: layerSelection, sequenceLength: sequenceLength)
        recordStep(
            "Generation loop",
            detail: "tokens=\(tokenCount) seq=\(sequenceLength.rawValue) strategy=reuse embedding+lm-head retain decoders=\(retainedDecoderCount) cache=\(cacheClearPolicy.rawValue) persistent=\(persistent)",
            steps: &steps
        )

        // Persistent path (chat): reuse resident models across turns, keep alive.
        if persistent {
            let signature = chatModelSignature(
                sequenceLength: sequenceLength,
                retainCount: retainedDecoderCount,
                layerSelection: layerSelection,
                endpointConfig: endpointConfig,
                decoderConfig: decoderConfig
            )
            let models = try ensureResidentChatModels(
                signature: signature,
                plan: decoderPlan,
                retainCount: retainedDecoderCount,
                sequenceLength: sequenceLength,
                endpointConfig: endpointConfig,
                decoderConfig: decoderConfig,
                steps: &steps
            )
            let predictions = try runTokenLoop(
                models: models,
                decoderPlan: decoderPlan,
                inputWindow: &inputWindow,
                tokenCount: tokenCount,
                imageHidden: imageHidden,
                imageStartPosition: imageStartPosition,
                imageBlockBidirectional: imageBlockBidirectional,
                sequenceLength: sequenceLength,
                decoderConfig: decoderConfig,
                layerSelection: layerSelection,
                cacheClearPolicy: cacheClearPolicy,
                onToken: onToken,
                steps: &steps
            )
            recordStep("Kept resident chat models", detail: "embedding+lm-head+\(models.retained.names.count) decoders resident", steps: &steps)
            recordGeneratedTokens(predictions, steps: &steps)
            return predictions
        }

        // One-shot path (benchmarks): load, run, release — bounded lifetime.
        let predictions = try autoreleasepool { () throws -> [TokenPrediction] in
            let models = LoadedChatModels(
                embedding: try loadModel(named: sequenceLength.embeddingName, config: makeEmbeddingConfig(), steps: &steps),
                lmHead: try loadLMHead(config: endpointConfig, steps: &steps),
                retained: try loadRetainedDecoderModels(
                    plan: decoderPlan,
                    requestedCount: retainedDecoderCount,
                    config: decoderConfig,
                    sequenceLength: sequenceLength,
                    steps: &steps
                )
            )
            let localPredictions = try runTokenLoop(
                models: models,
                decoderPlan: decoderPlan,
                inputWindow: &inputWindow,
                tokenCount: tokenCount,
                imageHidden: imageHidden,
                imageStartPosition: imageStartPosition,
                imageBlockBidirectional: imageBlockBidirectional,
                sequenceLength: sequenceLength,
                decoderConfig: decoderConfig,
                layerSelection: layerSelection,
                cacheClearPolicy: cacheClearPolicy,
                onToken: onToken,
                steps: &steps
            )
            recordStep("Released generation endpoints", detail: "\(sequenceLength.embeddingName), \(models.lmHead.name)", steps: &steps)
            if !models.retained.names.isEmpty {
                recordStep("Released retained decoders", detail: models.retained.names.joined(separator: ","), steps: &steps)
            }
            return localPredictions
        }

        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released generation endpoints", steps: &steps)
        }
        recordGeneratedTokens(predictions, steps: &steps)
        return predictions
    }

    private static func recordGeneratedTokens(_ predictions: [TokenPrediction], steps: inout [ProbeStep]) {
        let detail = predictions
            .map { "\($0.step):#\($0.index)=\(String(format: "%.3f", $0.logit))" }
            .joined(separator: ", ")
        recordStep("Generated tokens", detail: detail, steps: &steps)
    }

    private static func runTokenLoop(
        models: LoadedChatModels,
        decoderPlan: DecoderLayerPlan,
        inputWindow: inout TokenWindow,
        tokenCount: Int,
        imageHidden: MLMultiArray?,
        imageStartPosition: Int,
        imageBlockBidirectional: Bool = false,
        sequenceLength: ProbeSequenceLength,
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        onToken: ((Int) -> Void)? = nil,
        steps: inout [ProbeStep]
    ) throws -> [TokenPrediction] {
        var localPredictions: [TokenPrediction] = []
        // Window-relative start of the image-hidden block; slides left as
        // generated tokens push the window forward, until it falls off.
        var imageWindowStart = imageStartPosition

        for step in 1...tokenCount {
            let tokenStart = Date()
            let prediction = try autoreleasepool { () throws -> TokenPrediction in
                recordStep(
                    "Prompt IDs \(step)",
                    detail: "\(formatInputIDs(inputWindow.values)) positionStart=\(inputWindow.positionStart) leftPad=\(inputWindow.leftPadCount)",
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
                    model: models.embedding,
                    inputIDs: inputWindow.values,
                    name: "Embedding token \(step)",
                    steps: &steps
                )
                if let imageHidden {
                    let overlaid = try overlayImageHidden(
                        imageHidden,
                        into: hidden,
                        windowStart: imageWindowStart,
                        seqLength: sequenceLength.rawValue
                    )
                    if overlaid > 0 {
                        recordStep(
                            "Image hidden overlay \(step)",
                            detail: "patches=\(overlaid) windowStart=\(imageWindowStart)",
                            steps: &steps
                        )
                    }
                }
                let decoded = try runDecoderStack(
                    hidden: hidden,
                    config: decoderConfig,
                    layerSelection: layerSelection,
                    sequenceLength: sequenceLength,
                    positionStart: inputWindow.positionStart,
                    leftPadCount: inputWindow.leftPadCount,
                    cacheClearPolicy: cacheClearPolicy,
                    providedPlan: decoderPlan,
                    retainedDecoders: models.retained.models,
                    bidirectionalBlock: (imageBlockBidirectional && imageHidden != nil)
                        ? imageWindowStart..<(imageWindowStart + imageHidden!.count / 3840)
                        : nil,
                    steps: &steps
                )
                let lastHidden = try copyLastToken(from: decoded, sequenceLength: sequenceLength)
                let logits = try predictLMHead(
                    model: models.lmHead.model,
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

            localPredictions.append(prediction)
            onToken?(prediction.index)

            if let stopReason = generationStopReason(predictions: localPredictions, latestTokenID: prediction.index) {
                recordStep("Stop generation", detail: stopReason, steps: &steps)
                break
            }

            guard let nextToken = Int32(exactly: prediction.index) else {
                throw ProbeError.invalidInputIDs("generated token does not fit Int32: \(prediction.index)")
            }
            inputWindow.values.removeFirst()
            inputWindow.values.append(nextToken)
            imageWindowStart -= 1
            if inputWindow.leftPadCount > 0 {
                inputWindow.leftPadCount -= 1
            } else {
                inputWindow.positionStart += 1
            }
        }

        return localPredictions
    }

    private static func runMultimodalSmoke(
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        sequenceLength: ProbeSequenceLength,
        steps: inout [ProbeStep]
    ) throws -> TokenPrediction {
        let tokenStart = Date()
        recordStep(
            "Multimodal smoke",
            detail: "source=synthetic-placeholder seq=\(sequenceLength.rawValue) hidden=[1,\(sequenceLength.rawValue),3840]",
            steps: &steps
        )
        let hidden = try makeMultimodalSmokeHidden(seqLength: sequenceLength.rawValue)
        let decoded = try runDecoderStack(
            hidden: hidden,
            config: decoderConfig,
            layerSelection: layerSelection,
            sequenceLength: sequenceLength,
            positionStart: 0,
            leftPadCount: 0,
            cacheClearPolicy: cacheClearPolicy,
            steps: &steps
        )
        let lastHidden = try copyLastToken(from: decoded, sequenceLength: sequenceLength)
        let logits = try runLMHead(hidden: lastHidden, config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let token = try topLogit(logits)
        recordStep("Top logits token 1", detail: topLogitsSummary(logits, count: 5), steps: &steps)
        recordStep(
            "Generated token 1",
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))",
            steps: &steps
        )
        appendStep(ProbeStep(
            name: "Token 1 total",
            seconds: Date().timeIntervalSince(tokenStart),
            memoryMB: ProbeMemory.currentMB(),
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))"
        ), to: &steps)
        recordStep("Generated tokens", detail: "1:#\(token.index)=\(String(format: "%.3f", token.logit))", steps: &steps)
        return TokenPrediction(step: 1, index: token.index, logit: token.logit)
    }

    private static func runImageSmoke(
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        sequenceLength: ProbeSequenceLength,
        steps: inout [ProbeStep]
    ) throws -> TokenPrediction {
        let tokenStart = Date()
        recordStep(
            "Image smoke",
            detail: "source=raw-patch-fixture patches=\(imageSmokePatchCount) patch_dim=\(imageSmokePatchDim) hidden=[1,\(sequenceLength.rawValue),3840]",
            steps: &steps
        )
        let imageHidden = try runImageEmbedder(config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let hidden = try makeHidden(seqLength: sequenceLength.rawValue)
        try copyImageHidden(imageHidden, intoSequenceHidden: hidden, sequenceLength: sequenceLength.rawValue)
        recordStep(
            "Image hidden inserted",
            detail: "patches=\(imageSmokePatchCount) positions=\(max(0, sequenceLength.rawValue - imageSmokePatchCount))...\(sequenceLength.rawValue - 1)",
            steps: &steps
        )
        let decoded = try runDecoderStack(
            hidden: hidden,
            config: decoderConfig,
            layerSelection: layerSelection,
            sequenceLength: sequenceLength,
            positionStart: 0,
            leftPadCount: 0,
            cacheClearPolicy: cacheClearPolicy,
            steps: &steps
        )
        let lastHidden = try copyLastToken(from: decoded, sequenceLength: sequenceLength)
        let logits = try runLMHead(hidden: lastHidden, config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let token = try topLogit(logits)
        recordStep("Top logits token 1", detail: topLogitsSummary(logits, count: 5), steps: &steps)
        recordStep(
            "Generated token 1",
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))",
            steps: &steps
        )
        appendStep(ProbeStep(
            name: "Token 1 total",
            seconds: Date().timeIntervalSince(tokenStart),
            memoryMB: ProbeMemory.currentMB(),
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))"
        ), to: &steps)
        recordStep("Generated tokens", detail: "1:#\(token.index)=\(String(format: "%.3f", token.logit))", steps: &steps)
        return TokenPrediction(step: 1, index: token.index, logit: token.logit)
    }

    private static func runImageEmbedder(
        config: MLModelConfiguration,
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let pixelValues = try makeImageSmokePixelValues()
        let positionIDs = try makeImageSmokePositionIDs()
        let imageHidden = try autoreleasepool {
            let imageEmbedder = try loadModel(named: imageEmbedderName, config: config, steps: &steps)
            let output = try timedPrediction(
                name: "Image embedder",
                model: imageEmbedder,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "pixel_values": MLFeatureValue(multiArray: pixelValues),
                    "image_position_ids": MLFeatureValue(multiArray: positionIDs)
                ]),
                steps: &steps
            )
            return try requireArray(named: "image_hidden", output: output)
        }
        recordStep("Released \(imageEmbedderName)", detail: "image_hidden retained", steps: &steps)
        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released \(imageEmbedderName)", steps: &steps)
        }
        return imageHidden
    }

    private static func runAudioSmoke(
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        cacheClearPolicy: ProbeCacheClearPolicy,
        sequenceLength: ProbeSequenceLength,
        steps: inout [ProbeStep]
    ) throws -> TokenPrediction {
        let tokenStart = Date()
        recordStep(
            "Audio smoke",
            detail: "source=feature-fixture tokens=\(audioSmokeTokenCount) feature_dim=\(audioSmokeFeatureDim) hidden=[1,\(sequenceLength.rawValue),3840]",
            steps: &steps
        )
        let audioHidden = try runAudioEmbedder(config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let hidden = try makeHidden(seqLength: sequenceLength.rawValue)
        try copyAudioHidden(audioHidden, intoSequenceHidden: hidden, sequenceLength: sequenceLength.rawValue)
        recordStep(
            "Audio hidden inserted",
            detail: "tokens=\(audioSmokeTokenCount) positions=\(max(0, sequenceLength.rawValue - audioSmokeTokenCount))...\(sequenceLength.rawValue - 1)",
            steps: &steps
        )
        let decoded = try runDecoderStack(
            hidden: hidden,
            config: decoderConfig,
            layerSelection: layerSelection,
            sequenceLength: sequenceLength,
            positionStart: 0,
            leftPadCount: 0,
            cacheClearPolicy: cacheClearPolicy,
            steps: &steps
        )
        let lastHidden = try copyLastToken(from: decoded, sequenceLength: sequenceLength)
        let logits = try runLMHead(hidden: lastHidden, config: endpointConfig, cacheClearPolicy: cacheClearPolicy, steps: &steps)
        let token = try topLogit(logits)
        recordStep("Top logits token 1", detail: topLogitsSummary(logits, count: 5), steps: &steps)
        recordStep(
            "Generated token 1",
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))",
            steps: &steps
        )
        appendStep(ProbeStep(
            name: "Token 1 total",
            seconds: Date().timeIntervalSince(tokenStart),
            memoryMB: ProbeMemory.currentMB(),
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))"
        ), to: &steps)
        recordStep("Generated tokens", detail: "1:#\(token.index)=\(String(format: "%.3f", token.logit))", steps: &steps)
        return TokenPrediction(step: 1, index: token.index, logit: token.logit)
    }

    private static func runAudioEmbedder(
        config: MLModelConfiguration,
        cacheClearPolicy: ProbeCacheClearPolicy,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let inputFeatures = try makeAudioSmokeInputFeatures()
        let audioHidden = try autoreleasepool {
            let audioEmbedder = try loadModel(named: audioEmbedderName, config: config, steps: &steps)
            let output = try timedPrediction(
                name: "Audio embedder",
                model: audioEmbedder,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "input_features": MLFeatureValue(multiArray: inputFeatures)
                ]),
                steps: &steps
            )
            return try requireArray(named: "audio_hidden", output: output)
        }
        recordStep("Released \(audioEmbedderName)", detail: "audio_hidden retained", steps: &steps)
        if cacheClearPolicy.clearsAfterNonDecoderRelease {
            clearCoreMLRuntimeCache(reason: "released \(audioEmbedderName)", steps: &steps)
        }
        return audioHidden
    }

    /// Loads decoder chunks one at a time and keeps every one resident (no
    /// release, no cache clear), running one forward per chunk to force the
    /// weights fully into memory. After each chunk it logs the process
    /// footprint and the kernel-reported headroom, stopping before the
    /// available budget drops under the safety margin so the OS never jetsams
    /// the app. The result answers "how many 4-layer decoder chunks fit
    /// resident with margin on this device" — the measurement needed before
    /// switching the generation loop from reload-every-token to resident.
    private static func runMemoryRamp(
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        sequenceLength: ProbeSequenceLength,
        steps: inout [ProbeStep]
    ) throws -> String {
        let plan = try decoderLayerPlan(selection: layerSelection, sequenceLength: sequenceLength)
        let marginMB = selectedMemoryMarginMB()
        recordStep(
            "Memory ramp",
            detail: "chunks=\(plan.layers.count) variant=\(ProbeSequenceLength.decoderVariant) marginMB=\(marginMB) strategy=load+predict+retain compute=\(decoderConfig.computeUnits.rawValue)",
            steps: &steps
        )
        recordMemory("Ramp baseline", steps: &steps)

        // Definitive backend evidence for the first chunk: MLComputePlan reports
        // the preferred compute device per op, so "did this variant escape
        // BNNS/CPU" is answered directly instead of inferred from timings.
        if let firstLayer = plan.layers.first, let url = modelURL(named: firstLayer.name) {
            let start = Date()
            let summary = computePlanDeviceSummary(url: url, config: decoderConfig)
            appendStep(ProbeStep(
                name: "Compute plan \(firstLayer.title)",
                seconds: Date().timeIntervalSince(start),
                memoryMB: ProbeMemory.currentMB(),
                detail: summary
            ), to: &steps)
        }

        // Reused decoder inputs; the payload is synthetic because we only care
        // about the memory footprint of holding the weights resident, not the
        // logits.
        let hidden = try makeHidden(seqLength: sequenceLength.rawValue)
        let positionIDs = try makePositionIDs(seqLength: sequenceLength.rawValue, start: 0, leftPadCount: 0)
        let mask = try makeCausalMask(seqLength: sequenceLength.rawValue, leftPadCount: 0)

        var resident: [MLModel] = []
        var stopReason = "loaded all \(plan.layers.count) chunks resident"

        for layer in plan.layers {
            let available = ProbeMemory.availableMB()
            if available >= 0, available < Double(marginMB) {
                stopReason = "stopped before \(layer.title): available \(String(format: "%.1f", available))MB < margin \(marginMB)MB"
                recordStep("Ramp stop", detail: stopReason, steps: &steps)
                break
            }

            let model = try loadModel(named: layer.name, config: decoderConfig, steps: &steps)
            let output = try timedPrediction(
                name: "Ramp predict \(layer.title)",
                model: model,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "x": MLFeatureValue(multiArray: hidden),
                    "position_ids": MLFeatureValue(multiArray: positionIDs),
                    "attention_mask": MLFeatureValue(multiArray: mask)
                ]),
                steps: &steps
            )
            _ = try requireArray(named: "y", output: output)
            resident.append(model)
            recordMemory("Resident \(resident.count)/\(plan.layers.count) layers \(layer.title)", steps: &steps)
        }

        let residentCount = resident.count
        // Keep the models alive across the measurement, then release together so
        // the footprint numbers above reflect true concurrent residency.
        withExtendedLifetime(resident) {}
        resident.removeAll()
        recordStep("Released ramp decoders", detail: "released \(residentCount) resident chunk(s)", steps: &steps)

        let summary = "OK: resident \(residentCount)/\(plan.layers.count) chunks (\(residentCount * 4) layers) — \(stopReason)"
        recordStep("Memory ramp result", detail: summary, steps: &steps)
        return summary
    }

    /// Loads the MLComputePlan for a compiled model and counts operations per
    /// preferred compute device (CPU/GPU/ANE). Blocks the calling probe thread
    /// until the async plan load finishes; acceptable for a one-shot diagnostic.
    private static func computePlanDeviceSummary(url: URL, config: MLModelConfiguration) -> String {
        let semaphore = DispatchSemaphore(value: 0)
        var summary = "unavailable"
        Task {
            defer { semaphore.signal() }
            do {
                let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
                guard case .program(let program) = plan.modelStructure,
                      let mainFunction = program.functions["main"] else {
                    summary = "unsupported model structure"
                    return
                }
                var counts: [String: Int] = [:]
                for operation in mainFunction.block.operations {
                    guard let usage = plan.deviceUsage(for: operation) else { continue }
                    let device: String
                    switch usage.preferred {
                    case .cpu: device = "CPU"
                    case .gpu: device = "GPU"
                    case .neuralEngine: device = "ANE"
                    @unknown default: device = "other"
                    }
                    counts[device, default: 0] += 1
                }
                summary = counts.isEmpty
                    ? "no device usage reported"
                    : "preferred ops " + counts
                        .sorted { $0.value > $1.value }
                        .map { "\($0.key)=\($0.value)" }
                        .joined(separator: " ")
            } catch {
                summary = "compute plan failed: \(error.localizedDescription)"
            }
        }
        semaphore.wait()
        return summary
    }

    private static func recordMemory(_ name: String, steps: inout [ProbeStep]) {
        let available = ProbeMemory.availableMB()
        let detail = available >= 0
            ? "available \(String(format: "%.1f", available)) MB before jetsam"
            : "available unknown (process not memory-limited; run untethered)"
        recordStep(name, detail: detail, steps: &steps)
    }

    /// `COREML_PROBE_WARM_MODEL=<name>` (or `--warm-model=`) makes the
    /// load-decoder mode load an arbitrary bundled model by exact name,
    /// bypassing the contiguous-coverage decoder plan. Used for one-off
    /// diagnostics such as "does a single full-attention layer compile for the
    /// ANE" where the asset does not participate in a full 0..47 plan.
    private static func warmModelName() throws -> String? {
        let environment = ProcessInfo.processInfo.environment
        let prefix = "--warm-model="
        let raw = environment["COREML_PROBE_WARM_MODEL"]
            ?? ProcessInfo.processInfo.arguments
                .first(where: { $0.hasPrefix(prefix) })
                .map { String($0.dropFirst(prefix.count)) }
        guard let raw, !raw.isEmpty else { return nil }
        guard modelURL(named: raw) != nil else {
            throw ProbeError.missingModel(raw)
        }
        return raw
    }

    /// `COREML_PROBE_WARM_CHUNK=<startLayer>` (or `--warm-chunk=`) makes the
    /// load-decoder mode load exactly the decoder bundle starting at that layer
    /// index instead of layer 0. Used to warm the e5rt ANE-compilation cache
    /// one model per app launch, because compiling a second large ANE model in
    /// the same launch gets the process jetsammed.
    private static func warmChunkName(sequenceLength: ProbeSequenceLength) throws -> String? {
        let environment = ProcessInfo.processInfo.environment
        let prefix = "--warm-chunk="
        let raw = environment["COREML_PROBE_WARM_CHUNK"]
            ?? ProcessInfo.processInfo.arguments
                .first(where: { $0.hasPrefix(prefix) })
                .map { String($0.dropFirst(prefix.count)) }
        guard let raw, let startIndex = Int(raw) else { return nil }

        let plan = try decoderLayerPlan(selection: .all, sequenceLength: sequenceLength)
        guard let layer = plan.layers.first(where: { $0.startIndex == startIndex }) else {
            throw ProbeError.missingModel("decoder chunk starting at layer \(startIndex) for seq\(sequenceLength.rawValue)")
        }
        return layer.name
    }

    private static func selectedMemoryMarginMB() -> Int {
        let environment = ProcessInfo.processInfo.environment
        let prefix = "--memory-margin="
        let raw = environment["COREML_PROBE_MEMORY_MARGIN_MB"]
            ?? ProcessInfo.processInfo.arguments
                .first(where: { $0.hasPrefix(prefix) })
                .map { String($0.dropFirst(prefix.count)) }
        guard let raw, let value = Int(raw), value >= 0 else {
            return defaultMemoryMarginMB
        }
        return value
    }

    private static func generationStopReason(predictions: [TokenPrediction], latestTokenID: Int) -> String? {
        if generatedStopTokenIDs.contains(latestTokenID) {
            return "stop token #\(latestTokenID)"
        }

        guard predictions.count >= repeatedGeneratedTokenLimit else {
            return nil
        }

        let recent = predictions.suffix(repeatedGeneratedTokenLimit)
        if recent.allSatisfy({ $0.index == latestTokenID }) {
            return "token #\(latestTokenID) repeated \(repeatedGeneratedTokenLimit)x"
        }
        return nil
    }

    private static func runDecoderStack(
        hidden: MLMultiArray,
        config: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        sequenceLength: ProbeSequenceLength,
        positionStart: Int,
        leftPadCount: Int,
        cacheClearPolicy: ProbeCacheClearPolicy,
        providedPlan: DecoderLayerPlan? = nil,
        retainedDecoders: [String: MLModel] = [:],
        bidirectionalBlock: Range<Int>? = nil,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let plan: DecoderLayerPlan
        if let providedPlan {
            plan = providedPlan
        } else {
            plan = try decoderLayerPlan(selection: layerSelection, sequenceLength: sequenceLength)
        }
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
                leftPadCount: leftPadCount,
                cacheClearReason: cacheClearPolicy.decoderReleaseReason(
                    layerPosition: layerPosition,
                    totalLayers: plan.layers.count,
                    layerName: layer.name
                ),
                retainedModel: retainedDecoders[layer.name],
                bidirectionalBlock: bidirectionalBlock,
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
        leftPadCount: Int,
        cacheClearReason: String?,
        retainedModel: MLModel? = nil,
        bidirectionalBlock: Range<Int>? = nil,
        steps: inout [ProbeStep]
    ) throws -> MLMultiArray {
        let usesRetainedModel = retainedModel != nil
        let decoded = try autoreleasepool {
            let decoder: MLModel
            if let retainedModel {
                decoder = retainedModel
            } else {
                decoder = try loadModel(named: layer.name, config: decoderLoadConfig(for: layer, base: config), steps: &steps)
            }
            let positionIDs = try makePositionIDs(seqLength: sequenceLength.rawValue, start: positionStart, leftPadCount: leftPadCount)
            let mask = try makeCausalMask(seqLength: sequenceLength.rawValue, leftPadCount: leftPadCount, bidirectionalBlock: bidirectionalBlock)
            let output = try timedPrediction(
                name: "Decoder layer \(layer.title)",
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
        if usesRetainedModel {
            recordStep("Kept \(layer.name)", detail: "retained decoder model", steps: &steps)
        } else {
            recordStep("Released \(layer.name)", detail: "decoded retained", steps: &steps)
        }
        if let cacheClearReason, !usesRetainedModel {
            clearCoreMLRuntimeCache(reason: cacheClearReason, steps: &steps)
        }
        return decoded
    }

    private static func loadRetainedDecoderModels(
        plan: DecoderLayerPlan,
        requestedCount: Int,
        config: MLModelConfiguration,
        sequenceLength: ProbeSequenceLength,
        steps: inout [ProbeStep]
    ) throws -> RetainedDecoderModels {
        guard requestedCount > 0 else {
            return RetainedDecoderModels(models: [:], names: [])
        }

        let targetCount = min(requestedCount, plan.layers.count)
        let marginMB = selectedMemoryMarginMB()
        recordStep(
            "Retain decoder models",
            detail: "requested=\(requestedCount) retaining=\(targetCount) of \(plan.layers.count) marginMB=\(marginMB)",
            steps: &steps
        )

        // Warmup inputs: on the A15 Neural Engine, loading ~24+ pal4 models
        // back-to-back WITHOUT exercising them stalls the ANE loader (observed
        // hang on iPhone 14). Running one synthetic prediction right after each
        // load commits the model on the ANE and avoids the stall — the same
        // load→predict cadence the memory-ramp probe uses to reach 36 resident.
        let warmHidden = try makeHidden(seqLength: sequenceLength.rawValue)
        let warmPositions = try makePositionIDs(seqLength: sequenceLength.rawValue, start: 0, leftPadCount: 0)
        let warmMask = try makeCausalMask(seqLength: sequenceLength.rawValue, leftPadCount: 0)
        let warmProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: warmHidden),
            "position_ids": MLFeatureValue(multiArray: warmPositions),
            "attention_mask": MLFeatureValue(multiArray: warmMask)
        ])

        var models: [String: MLModel] = [:]
        var names: [String] = []
        for layer in plan.layers.prefix(targetCount) {
            // Stop retaining before the process crosses its jetsam margin. Each
            // resident chunk is ~495 MB, so over-requesting on a device without
            // the increased-memory-limit entitlement would otherwise get the app
            // killed. Chunks left unretained simply reload per token as before.
            let available = ProbeMemory.availableMB()
            if available >= 0, available < Double(marginMB) {
                recordStep(
                    "Retain stop",
                    detail: "retained \(names.count)/\(targetCount): available \(String(format: "%.1f", available))MB < margin \(marginMB)MB; remaining chunks reload per token",
                    steps: &steps
                )
                break
            }
            let model = try loadModel(named: layer.name, config: decoderLoadConfig(for: layer, base: config), steps: &steps)
            try autoreleasepool {
                _ = try model.prediction(from: warmProvider)
            }
            models[layer.name] = model
            names.append(layer.name)
        }

        recordStep(
            "Retained decoder models",
            detail: names.isEmpty ? "none (memory margin reached before any retained)" : "\(names.count) warmed+resident",
            steps: &steps
        )
        return RetainedDecoderModels(models: models, names: names)
    }

    private static func decoderLayerPlan(selection: ProbeLayerSelection, sequenceLength: ProbeSequenceLength) throws -> DecoderLayerPlan {
        var candidatesByStart: [Int: [DecoderLayerModel]] = [:]
        var coveredIndices: Set<Int> = []
        let urls = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: "Models") ?? []

        for url in urls {
            let name = url.deletingPathExtension().lastPathComponent
            let model: DecoderLayerModel?
            if name.hasPrefix(decoderChunkPrefix), name.hasSuffix(sequenceLength.decoderSuffix) {
                let start = name.index(name.startIndex, offsetBy: decoderChunkPrefix.count)
                let end = name.index(name.endIndex, offsetBy: -sequenceLength.decoderSuffix.count)
                let rangeText = String(name[start..<end])
                let parts = rangeText.split(separator: "_", maxSplits: 1).map(String.init)
                if parts.count == 2,
                   let startIndex = Int(parts[0]),
                   let endIndex = Int(parts[1]),
                   startIndex <= endIndex {
                    model = DecoderLayerModel(startIndex: startIndex, endIndex: endIndex, name: name)
                } else {
                    model = nil
                }
            } else if name.hasPrefix(decoderPrefix), name.hasSuffix(sequenceLength.decoderSuffix) {
                let start = name.index(name.startIndex, offsetBy: decoderPrefix.count)
                let end = name.index(name.endIndex, offsetBy: -sequenceLength.decoderSuffix.count)
                let numberText = String(name[start..<end])
                if let index = Int(numberText) {
                    model = DecoderLayerModel(startIndex: index, endIndex: index, name: name)
                } else {
                    model = nil
                }
            } else {
                model = nil
            }

            guard let model else {
                continue
            }
            candidatesByStart[model.startIndex, default: []].append(model)
            for index in model.startIndex...model.endIndex {
                coveredIndices.insert(index)
            }
        }

        guard !coveredIndices.isEmpty else {
            throw ProbeError.missingModel("\(decoderPrefix)*\(sequenceLength.decoderSuffix).mlmodelc or \(decoderChunkPrefix)*\(sequenceLength.decoderSuffix).mlmodelc")
        }

        let availableCount = coveredIndices.count
        let targetCount = selection.requestedCount ?? availableCount
        var selectedLayers: [DecoderLayerModel] = []
        var cursor = 0
        while cursor < targetCount {
            guard let candidates = candidatesByStart[cursor] else {
                throw ProbeError.missingModel("decoder layer \(cursor) for seq\(sequenceLength.rawValue)")
            }
            let validCandidates = candidates.filter { $0.endIndex < targetCount }
            guard let selected = validCandidates.max(by: { lhs, rhs in
                if lhs.layerCount == rhs.layerCount {
                    return lhs.name < rhs.name
                }
                return lhs.layerCount < rhs.layerCount
            }) else {
                throw ProbeError.missingModel("decoder layer \(cursor) for seq\(sequenceLength.rawValue)")
            }
            selectedLayers.append(selected)
            cursor = selected.endIndex + 1
        }

        return DecoderLayerPlan(
            availableCount: availableCount,
            layers: selectedLayers,
            selection: selection
        )
    }

    // MARK: - KV-cache generation

    /// Fixed KV geometry matching the converted assets. Prompt prefill fills
    /// slots [0, 320); generation writes slots [320, 512) — up to 192 tokens.
    static let kvPrefillLength = 320
    static let kvCacheCapacity = 512

    private static func kvPrefillName(layer: Int) -> String {
        String(format: "gemma4_12b_layer%02d_prefill_seq%d_kv_pal4_g16", layer, kvPrefillLength)
    }

    private static func kvDecodeName(layer: Int) -> String {
        String(format: "gemma4_12b_layer%02d_decode_kv%d_pal4_g16", layer, kvCacheCapacity)
    }

    private static let kvEmbeddingSeq1Name = "gemma4_12b_embedding_seq1_int4_block32"

    /// True when the whole KV stack is bundled (checked at the endpoints and
    /// both layer families' first/last members). `COREML_PROBE_DISABLE_KV=1`
    /// keeps the cache-free path selectable for A/B runs.
    static var kvChatAvailable: Bool {
        if ProcessInfo.processInfo.environment["COREML_PROBE_DISABLE_KV"] == "1" { return false }
        return ProbeSequenceLength.modelExists(named: kvEmbeddingSeq1Name)
            && ProbeSequenceLength.modelExists(named: ProbeSequenceLength.seq320.embeddingName)
            && ProbeSequenceLength.modelExists(named: kvPrefillName(layer: 0))
            && ProbeSequenceLength.modelExists(named: kvPrefillName(layer: 47))
            && ProbeSequenceLength.modelExists(named: kvDecodeName(layer: 0))
            && ProbeSequenceLength.modelExists(named: kvDecodeName(layer: 47))
    }

    /// Per-layer KV cache geometry. Gemma 4 Unified alternates 5 sliding
    /// (GQA, 8 heads x 256) + 1 full-attention (MQA with K=V projection,
    /// 1 head x 512) layers; mirrors the conversion-side `layer_kv_shape`.
    private static func kvLayerGeometry(layer: Int) -> (heads: Int, headDim: Int) {
        layer % 6 == 5 ? (1, 512) : (8, 256)
    }

    private struct KVLayerCache {
        let kBuffer: MLMultiArray
        let vBuffer: MLMultiArray
        let heads: Int
        let headDim: Int
    }

    private static func makeKVCaches(layerCount: Int) throws -> [KVLayerCache] {
        try (0..<layerCount).map { layer in
            let geometry = kvLayerGeometry(layer: layer)
            let shape: [NSNumber] = [1, NSNumber(value: geometry.heads), NSNumber(value: kvCacheCapacity), NSNumber(value: geometry.headDim)]
            let k = try MLMultiArray(shape: shape, dataType: .float16)
            let v = try MLMultiArray(shape: shape, dataType: .float16)
            // Zero-fill so unwritten slots hold defined values (they are also
            // masked with -inf, but NaN garbage would still poison 0*NaN paths).
            for buffer in [k, v] {
                let pointer = buffer.dataPointer.bindMemory(to: Float16.self, capacity: buffer.count)
                for index in 0..<buffer.count { pointer[index] = 0 }
            }
            return KVLayerCache(kBuffer: k, vBuffer: v, heads: geometry.heads, headDim: geometry.headDim)
        }
    }

    /// Copies a prefill K/V block [1,H,block,D] into cache slots [0, block).
    private static func copyKVBlock(_ block: MLMultiArray, into buffer: MLMultiArray, cache: KVLayerCache) throws {
        guard block.dataType == .float16, buffer.dataType == .float16 else {
            throw ProbeError.unexpectedShape("KV block dtype \(block.dataType)")
        }
        let blockLength = block.count / (cache.heads * cache.headDim)
        let source = block.dataPointer.bindMemory(to: Float16.self, capacity: block.count)
        let destination = buffer.dataPointer.bindMemory(to: Float16.self, capacity: buffer.count)
        for head in 0..<cache.heads {
            let sourceBase = head * blockLength * cache.headDim
            let destinationBase = head * kvCacheCapacity * cache.headDim
            for index in 0..<(blockLength * cache.headDim) {
                destination[destinationBase + index] = source[sourceBase + index]
            }
        }
    }

    /// Writes a decode-step K/V [1,H,1,D] into cache slot `slot`.
    private static func writeKVSlot(_ new: MLMultiArray, into buffer: MLMultiArray, cache: KVLayerCache, slot: Int) {
        let source = new.dataPointer.bindMemory(to: Float16.self, capacity: new.count)
        let destination = buffer.dataPointer.bindMemory(to: Float16.self, capacity: buffer.count)
        for head in 0..<cache.heads {
            let sourceBase = head * cache.headDim
            let destinationBase = (head * kvCacheCapacity + slot) * cache.headDim
            for index in 0..<cache.headDim {
                destination[destinationBase + index] = source[sourceBase + index]
            }
        }
    }

    /// Decode-step mask [1,1,1,capacity+1]: slots [leftPad, written) and the
    /// new token itself (last column) are visible; pads and unwritten slots
    /// stay at -inf.
    private static func makeKVDecodeMask(leftPadCount: Int, written: Int) throws -> MLMultiArray {
        let width = kvCacheCapacity + 1
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: width)], dataType: .float16)
        let pointer = mask.dataPointer.bindMemory(to: Float16.self, capacity: width)
        for index in 0..<width { pointer[index] = Float16(-65504.0) }
        for index in leftPadCount..<min(written, kvCacheCapacity) { pointer[index] = 0 }
        pointer[width - 1] = 0
        return mask
    }

    private struct KVChatModels {
        let embeddingSeq1: MLModel
        let lmHead: LoadedProbeModel
        let decoders: [String: MLModel]
    }

    private static var residentKVModels: KVChatModels?
    private static var residentKVSignature: String?

    static func releaseResidentKVModels() {
        residentKVModels = nil
        residentKVSignature = nil
    }

    private static func ensureResidentKVModels(
        layerCount: Int,
        retainCount: Int,
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        steps: inout [ProbeStep]
    ) throws -> KVChatModels {
        let signature = "kv|\(layerCount)|retain\(retainCount)|ep\(endpointConfig.computeUnits.rawValue)|dec\(decoderConfig.computeUnits.rawValue)"
        if let models = residentKVModels, residentKVSignature == signature {
            recordStep("Resident KV models", detail: "reused signature=\(signature)", steps: &steps)
            return models
        }
        releaseResidentKVModels()
        releaseResidentChatModels()
        recordStep("Resident KV models", detail: "loading signature=\(signature)", steps: &steps)
        var decoders: [String: MLModel] = [:]
        for layer in 0..<min(retainCount, layerCount) {
            let name = kvDecodeName(layer: layer)
            decoders[name] = try loadModel(named: name, config: decoderConfig, steps: &steps)
        }
        let models = KVChatModels(
            embeddingSeq1: try loadModel(named: kvEmbeddingSeq1Name, config: makeEmbeddingConfig(), steps: &steps),
            lmHead: try loadLMHead(config: endpointConfig, steps: &steps),
            decoders: decoders
        )
        residentKVModels = models
        residentKVSignature = signature
        return models
    }

    /// KV-cache generation: one chunked prefill over the 320-token window
    /// fills the caches and produces token 1; each further token runs the
    /// seq-1 decode stack against the caches (attention over [cache ; new]).
    private static func runKVGeneration(
        endpointConfig: MLModelConfiguration,
        decoderConfig: MLModelConfiguration,
        layerSelection: ProbeLayerSelection,
        inputIDsText: String?,
        generatedTokenCount: Int?,
        retainedDecoderModelCount: Int?,
        imageHidden: MLMultiArray?,
        imageStartPosition: Int,
        imageBlockBidirectional: Bool,
        persistent: Bool,
        onToken: ((Int) -> Void)?,
        steps: inout [ProbeStep]
    ) throws -> [TokenPrediction] {
        let sequenceLength = ProbeSequenceLength.seq320
        let inputWindow = try selectedInputWindow(overrideText: inputIDsText, sequenceLength: sequenceLength)
        // The cache holds capacity-prefill generation slots; clamp rather than
        // fail so the 256-token chat preset still runs (just capped).
        let tokenCount = min(try selectedGeneratedTokenCount(override: generatedTokenCount), kvCacheCapacity - kvPrefillLength)
        let layerCount = min(layerSelection.requestedCount ?? 48, 48)
        let retainCount = retainedDecoderModelCount ?? 0
        recordStep(
            "KV generation",
            detail: "prefill=\(kvPrefillLength) capacity=\(kvCacheCapacity) tokens=\(tokenCount) layers=\(layerCount) retain=\(retainCount) persistent=\(persistent)",
            steps: &steps
        )

        let models = try ensureResidentKVModels(
            layerCount: layerCount,
            retainCount: persistent ? retainCount : 0,
            endpointConfig: endpointConfig,
            decoderConfig: decoderConfig,
            steps: &steps
        )
        let caches = try makeKVCaches(layerCount: layerCount)
        var predictions: [TokenPrediction] = []
        let imageBlock: Range<Int>? = (imageBlockBidirectional && imageHidden != nil)
            ? imageStartPosition..<(imageStartPosition + imageHidden!.count / 3840)
            : nil

        // ---- Prefill ----
        let prefillStart = Date()
        var hidden: MLMultiArray = try autoreleasepool {
            let embedding = try loadModel(named: sequenceLength.embeddingName, config: makeEmbeddingConfig(), steps: &steps)
            return try predictEmbedding(model: embedding, inputIDs: inputWindow.values, name: "KV prefill embedding", steps: &steps)
        }
        if let imageHidden {
            let overlaid = try overlayImageHidden(imageHidden, into: hidden, windowStart: imageStartPosition, seqLength: kvPrefillLength)
            recordStep("Image hidden overlay prefill", detail: "patches=\(overlaid) windowStart=\(imageStartPosition)", steps: &steps)
        }
        let prefillMask = try makeCausalMask(
            seqLength: kvPrefillLength,
            leftPadCount: inputWindow.leftPadCount,
            bidirectionalBlock: imageBlock
        )
        let prefillPositions = try makePositionIDs(seqLength: kvPrefillLength, start: inputWindow.positionStart, leftPadCount: inputWindow.leftPadCount)
        for layer in 0..<layerCount {
            hidden = try autoreleasepool { () throws -> MLMultiArray in
                let model = try loadModel(named: kvPrefillName(layer: layer), config: decoderConfig, steps: &steps)
                let output = try timedPrediction(
                    name: "KV prefill layer \(layer)",
                    model: model,
                    provider: MLDictionaryFeatureProvider(dictionary: [
                        "x": MLFeatureValue(multiArray: hidden),
                        "position_ids": MLFeatureValue(multiArray: prefillPositions),
                        "attention_mask": MLFeatureValue(multiArray: prefillMask)
                    ]),
                    steps: &steps
                )
                let cache = caches[layer]
                try copyKVBlock(try requireArray(named: "k_block", output: output), into: cache.kBuffer, cache: cache)
                try copyKVBlock(try requireArray(named: "v_block", output: output), into: cache.vBuffer, cache: cache)
                return try requireArray(named: "y", output: output)
            }
        }
        var written = kvPrefillLength
        let lastHidden = try copyLastToken(from: hidden, sequenceLength: sequenceLength)
        var logits = try predictLMHead(model: models.lmHead.model, hidden: lastHidden, name: "KV LM head token 1", steps: &steps)
        var token = try topLogit(logits)
        recordStep("Top logits token 1", detail: topLogitsSummary(logits, count: 5), steps: &steps)
        appendStep(ProbeStep(
            name: "Token 1 total",
            seconds: Date().timeIntervalSince(prefillStart),
            memoryMB: ProbeMemory.currentMB(),
            detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit)) (prefill)"
        ), to: &steps)
        predictions.append(TokenPrediction(step: 1, index: token.index, logit: token.logit))
        onToken?(token.index)

        // ---- Decode loop ----
        var step = 2
        while step <= tokenCount {
            if let stopReason = generationStopReason(predictions: predictions, latestTokenID: token.index) {
                recordStep("Stop generation", detail: stopReason, steps: &steps)
                break
            }
            let tokenStart = Date()
            let currentToken = token
            token = try autoreleasepool { () throws -> (index: Int, logit: Float) in
                var x = try predictEmbedding(
                    model: models.embeddingSeq1,
                    inputIDs: [Int32(currentToken.index)],
                    name: "KV embedding token \(step)",
                    steps: &steps
                )
                let position = try MLMultiArray(shape: [1, 1], dataType: .int32)
                // RoPE position is the real-token-relative index (prefill numbers
                // real tokens 0,1,2,... after the left pad), while the cache slot
                // is the absolute buffer index. They differ by leftPadCount.
                position[0] = NSNumber(value: written - inputWindow.leftPadCount)
                let mask = try makeKVDecodeMask(leftPadCount: inputWindow.leftPadCount, written: written)
                for layer in 0..<layerCount {
                    let name = kvDecodeName(layer: layer)
                    let model: MLModel
                    if let resident = models.decoders[name] {
                        model = resident
                    } else {
                        model = try loadModel(named: name, config: decoderConfig, steps: &steps)
                    }
                    let cache = caches[layer]
                    let output = try timedPrediction(
                        name: "KV decode layer \(layer) token \(step)",
                        model: model,
                        provider: MLDictionaryFeatureProvider(dictionary: [
                            "x": MLFeatureValue(multiArray: x),
                            "position_ids": MLFeatureValue(multiArray: position),
                            "attention_mask": MLFeatureValue(multiArray: mask),
                            "k_cache": MLFeatureValue(multiArray: cache.kBuffer),
                            "v_cache": MLFeatureValue(multiArray: cache.vBuffer)
                        ]),
                        steps: &steps
                    )
                    writeKVSlot(try requireArray(named: "k_new", output: output), into: cache.kBuffer, cache: cache, slot: written)
                    writeKVSlot(try requireArray(named: "v_new", output: output), into: cache.vBuffer, cache: cache, slot: written)
                    x = try requireArray(named: "y", output: output)
                }
                written += 1
                logits = try predictLMHead(model: models.lmHead.model, hidden: x, name: "KV LM head token \(step)", steps: &steps)
                return try topLogit(logits)
            }
            recordStep("Top logits token \(step)", detail: topLogitsSummary(logits, count: 5), steps: &steps)
            appendStep(ProbeStep(
                name: "Token \(step) total",
                seconds: Date().timeIntervalSince(tokenStart),
                memoryMB: ProbeMemory.currentMB(),
                detail: "#\(token.index) logit=\(String(format: "%.3f", token.logit))"
            ), to: &steps)
            predictions.append(TokenPrediction(step: step, index: token.index, logit: token.logit))
            onToken?(token.index)
            step += 1
        }
        if !persistent {
            releaseResidentKVModels()
            recordStep("Released KV models", detail: "persistent=false", steps: &steps)
        }
        recordGeneratedTokens(predictions, steps: &steps)
        return predictions
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

    /// When set (env `COREML_PROBE_KEEP_E5_CACHE=1` or `--keep-e5-cache`), the
    /// e5rt ANE-compilation cache survives across runs. Required for the pal4
    /// ANE assets: recompiling a dozen ~450MB models in one launch spikes
    /// transient memory during ANECompilerService handoff and gets the app
    /// jetsammed (SIGKILL) long before phys_footprint shows pressure. With the
    /// cache kept, each model compiles once and later launches reuse binaries.
    static let keepsE5Cache: Bool = {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["COREML_PROBE_KEEP_E5_CACHE"] {
            return ["1", "true", "yes", "on"].contains(value.lowercased())
        }
        if ProcessInfo.processInfo.arguments.contains("--keep-e5-cache") {
            return true
        }
        // Default on for the pal4/ANE stack: clearing the cache forces a full
        // ANE recompilation of every model on the next launch.
        return ProbeSequenceLength.usesPal4Stack
    }()

    private static func clearCoreMLRuntimeCache(reason: String, steps: inout [ProbeStep]) {
        if keepsE5Cache {
            recordStep("Keep Core ML cache", detail: "skipped clear (reason=\(reason)) keep-e5-cache enabled", steps: &steps)
            return
        }
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

    private static func makeMultimodalSmokeHidden(seqLength: Int) throws -> MLMultiArray {
        let hiddenSize = 3840
        let array = try MLMultiArray(shape: [1, NSNumber(value: seqLength), NSNumber(value: hiddenSize)], dataType: .float16)
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        for index in 0..<array.count {
            pointer[index] = 0
        }

        let markerTokenIDs = [255999, 258880, 258882]
        let startPosition = max(0, seqLength - markerTokenIDs.count)
        for (offset, tokenID) in markerTokenIDs.enumerated() {
            let position = startPosition + offset
            guard position < seqLength else { continue }
            let base = position * hiddenSize
            pointer[base] = Float16(0.125)
            pointer[base + 1] = Float16(Float(position + 1) / Float(max(seqLength, 1)))
            pointer[base + (tokenID % hiddenSize)] = Float16(0.25)
            pointer[base + ((tokenID / hiddenSize) % hiddenSize)] = Float16(-0.125)
        }

        return array
    }

    private static func makeImageSmokePixelValues() throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: imageSmokePatchCount), NSNumber(value: imageSmokePatchDim)],
            dataType: .float32
        )
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let side = Int(ceil(sqrt(Double(imageSmokePatchCount))))
        let patchSide = 48
        let channelCount = 3

        for patchIndex in 0..<imageSmokePatchCount {
            let patchX = patchIndex % side
            let patchY = patchIndex / side
            let patchBase = patchIndex * imageSmokePatchDim
            for pixelIndex in 0..<(patchSide * patchSide) {
                let x = pixelIndex % patchSide
                let y = pixelIndex / patchSide
                let gradient = Float((patchX + x) % patchSide) / Float(patchSide - 1)
                let vertical = Float((patchY + y) % patchSide) / Float(patchSide - 1)
                let mixed = Float((patchIndex + x + y) % patchSide) / Float(patchSide - 1)
                let pixelBase = patchBase + pixelIndex * channelCount
                pointer[pixelBase] = gradient
                pointer[pixelBase + 1] = vertical
                pointer[pixelBase + 2] = mixed
            }
        }

        return array
    }

    private static func makeImageSmokePositionIDs(patchCount: Int = imageSmokePatchCount) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: patchCount), 2],
            dataType: .int32
        )
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        let side = Int32(ceil(sqrt(Double(patchCount))))
        for patchIndex in 0..<patchCount {
            let base = patchIndex * 2
            pointer[base] = Int32(patchIndex) % side
            pointer[base + 1] = Int32(patchIndex) / side
        }
        return array
    }

    /// Writes image-hidden patch vectors into a sequence-hidden buffer at a
    /// window-relative start offset (which may be negative once the block has
    /// partially slid out of the window). Returns the number of patches that
    /// landed inside the window.
    @discardableResult
    private static func overlayImageHidden(
        _ imageHidden: MLMultiArray,
        into hidden: MLMultiArray,
        windowStart: Int,
        seqLength: Int
    ) throws -> Int {
        let hiddenSize = 3840
        guard imageHidden.dataType == .float16, hidden.dataType == .float16 else {
            throw ProbeError.unexpectedShape("image overlay requires float16 hidden buffers")
        }
        let patchCount = imageHidden.count / hiddenSize
        let source = imageHidden.dataPointer.bindMemory(to: Float16.self, capacity: imageHidden.count)
        let destination = hidden.dataPointer.bindMemory(to: Float16.self, capacity: hidden.count)
        // Experimental scale knob: the merged image tokens must sit at the same
        // magnitude as the ×sqrt(hidden) text embeddings. scale=1 uses the raw
        // embedder output; other values let us A/B the scale hypothesis on
        // device without reconverting the embedder.
        let scale = imageHiddenScale
        var written = 0
        for patch in 0..<patchCount {
            let position = windowStart + patch
            guard position >= 0, position < seqLength else { continue }
            let sourceBase = patch * hiddenSize
            let destinationBase = position * hiddenSize
            if scale == 1 {
                for index in 0..<hiddenSize {
                    destination[destinationBase + index] = source[sourceBase + index]
                }
            } else {
                for index in 0..<hiddenSize {
                    destination[destinationBase + index] = Float16(Float(source[sourceBase + index]) * scale)
                }
            }
            written += 1
        }
        return written
    }

    /// Runs the bundled Gemma4 image embedder over caller-supplied pixel
    /// patches ([1, 32, 6912] fp32, values normalized to [-1, 1]) and returns
    /// `image_hidden` [1, 32, 3840] fp16 for overlay into the decoder input.
    /// Loads the embedder on CPU (it is small: ~39MB, predict ~0.02s).
    static func encodeImage(pixelValues: MLMultiArray) throws -> MLMultiArray {
        var steps: [ProbeStep] = []
        let patchCount = pixelValues.shape.count > 1 ? pixelValues.shape[1].intValue : imageSmokePatchCount
        let embedderName = patchCount == 256 ? imageEmbedder256Name : imageEmbedderName
        let positionIDs = try makeImageSmokePositionIDs(patchCount: patchCount)
        return try autoreleasepool {
            let imageEmbedder = try loadModel(named: embedderName, config: makeConfig(.cpuOnly), steps: &steps)
            let output = try timedPrediction(
                name: "Image embedder (chat)",
                model: imageEmbedder,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "pixel_values": MLFeatureValue(multiArray: pixelValues),
                    "image_position_ids": MLFeatureValue(multiArray: positionIDs)
                ]),
                steps: &steps
            )
            // The fp32-compute embedder can hand back a float32 array even
            // though the spec declares fp16; normalize so the overlay (which
            // binds Float16) works either way.
            let imageHidden = try float16Copy(of: requireArray(named: "image_hidden", output: output))
            logHiddenStats(imageHidden, label: "Image")
            return imageHidden
        }
    }

    /// Runs the bundled Gemma4 audio embedder over caller-supplied raw
    /// waveform frames ([1, 32, 640] fp32, 16 kHz PCM in [-1, 1], 640 samples
    /// = 40ms per token) and returns `audio_hidden` [1, 32, 3840] fp16 for
    /// overlay into the decoder input, exactly like the image path.
    static func encodeAudio(inputFeatures: MLMultiArray) throws -> MLMultiArray {
        var steps: [ProbeStep] = []
        return try autoreleasepool {
            let audioEmbedder = try loadModel(named: audioEmbedderName, config: makeConfig(.cpuOnly), steps: &steps)
            let output = try timedPrediction(
                name: "Audio embedder (chat)",
                model: audioEmbedder,
                provider: MLDictionaryFeatureProvider(dictionary: [
                    "input_features": MLFeatureValue(multiArray: inputFeatures)
                ]),
                steps: &steps
            )
            let audioHidden = try float16Copy(of: requireArray(named: "audio_hidden", output: output))
            logHiddenStats(audioHidden, label: "Audio")
            return audioHidden
        }
    }

    private static func float16Copy(of array: MLMultiArray) throws -> MLMultiArray {
        if array.dataType == .float16 { return array }
        guard array.dataType == .float32 else {
            throw ProbeError.unexpectedShape("expected float16/float32 image_hidden, got \(array.dataType)")
        }
        let copy = try MLMultiArray(shape: array.shape, dataType: .float16)
        let source = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let destination = copy.dataPointer.bindMemory(to: Float16.self, capacity: copy.count)
        for index in 0..<array.count {
            destination[index] = Float16(source[index])
        }
        return copy
    }

    /// One-line scale diagnostic for the modality paths: exploded hiddens
    /// saturate the softcapped lm_head (every top logit pinned at ~29.97), so
    /// the magnitude here tells us whether input normalization matches training.
    private static func logHiddenStats(_ array: MLMultiArray, label: String) {
        guard array.dataType == .float16 else { return }
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        var maxAbs: Float = 0
        var sum: Float = 0
        for index in 0..<array.count {
            let value = Float(pointer[index])
            maxAbs = max(maxAbs, abs(value))
            sum += value
        }
        let line = "[CoreMLProbe] \(label) hidden stats maxAbs=\(maxAbs) mean=\(sum / Float(array.count)) count=\(array.count) scale=\(imageHiddenScale)"
        print(line)
        appendToStepFile(line)
        // encodeImage/encodeAudio run before run() calls resetStepFile(), so
        // also stash the line and let run() re-append it after the reset.
        pendingImageStatsLine = line
    }

    static let imagePatchCount = imageSmokePatchCount
    static let imagePatchDim = imageSmokePatchDim
    static let audioTokenCount = audioSmokeTokenCount
    static let audioFeatureDim = audioSmokeFeatureDim

    /// True when the full-fidelity image chat stack is bundled: Seq320 text
    /// embedding, at least the first Seq320 decoder layer, and the 256-patch
    /// (16x16 grid = 768x768 px) image embedder. 256 image tokens matches the
    /// real Gemma 4 processor budget; the Seq64/32-patch path stays as the
    /// fallback micro smoke.
    static var imageChatSeq320Available: Bool {
        guard ProbeSequenceLength.modelExists(named: "gemma4_12b_embedding_seq320_int4_block32"),
              ProbeSequenceLength.modelExists(named: imageEmbedder256Name) else { return false }
        // The Seq320 window can be driven either by the non-KV per-layer
        // decoders or by the KV prefill stack; require at least one.
        return kvChatAvailable
            || ProbeSequenceLength.modelExists(named: ProbeSequenceLength.seq320.decoderName)
    }

    /// Multiplier applied to image_hidden vectors before they overlay the text
    /// embedding sequence. Defaults from `COREML_PROBE_IMAGE_SCALE` (else 1.0)
    /// and can be overridden per request via the `/generate` API `image_scale`.
    static var imageHiddenScale: Float = {
        ProcessInfo.processInfo.environment["COREML_PROBE_IMAGE_SCALE"].flatMap { Float($0) } ?? 1.0
    }()

    /// Image-hidden diagnostic line, stashed so it survives `resetStepFile()`
    /// (encodeImage runs before run() resets the step log).
    static var pendingImageStatsLine: String?

    private static func copyImageHidden(
        _ imageHidden: MLMultiArray,
        intoSequenceHidden sequenceHidden: MLMultiArray,
        sequenceLength: Int
    ) throws {
        let hiddenSize = 3840
        guard imageHidden.dataType == .float16 else {
            throw ProbeError.unexpectedShape("image_hidden expected float16, got \(imageHidden.dataType)")
        }
        guard imageHidden.count == imageSmokePatchCount * hiddenSize else {
            throw ProbeError.unexpectedShape("image_hidden expected \(imageSmokePatchCount * hiddenSize) values, got \(imageHidden.count)")
        }
        guard sequenceHidden.dataType == .float16 else {
            throw ProbeError.unexpectedShape("sequence hidden expected float16, got \(sequenceHidden.dataType)")
        }
        guard sequenceLength >= imageSmokePatchCount else {
            throw ProbeError.unexpectedShape("sequence length \(sequenceLength) is smaller than image patches \(imageSmokePatchCount)")
        }

        let source = imageHidden.dataPointer.bindMemory(to: Float16.self, capacity: imageHidden.count)
        let destination = sequenceHidden.dataPointer.bindMemory(to: Float16.self, capacity: sequenceHidden.count)
        let startPosition = sequenceLength - imageSmokePatchCount
        for patchIndex in 0..<imageSmokePatchCount {
            let sourceBase = patchIndex * hiddenSize
            let destinationBase = (startPosition + patchIndex) * hiddenSize
            for hiddenIndex in 0..<hiddenSize {
                destination[destinationBase + hiddenIndex] = source[sourceBase + hiddenIndex]
            }
        }
    }

    private static func makeAudioSmokeInputFeatures() throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: audioSmokeTokenCount), NSNumber(value: audioSmokeFeatureDim)],
            dataType: .float32
        )
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        for tokenIndex in 0..<audioSmokeTokenCount {
            for featureIndex in 0..<audioSmokeFeatureDim {
                let phase = Float((tokenIndex * 17 + featureIndex * 3) % 127) / 126.0
                let envelope = Float(tokenIndex + 1) / Float(max(audioSmokeTokenCount, 1))
                pointer[tokenIndex * audioSmokeFeatureDim + featureIndex] = (phase - 0.5) * envelope
            }
        }
        return array
    }

    private static func copyAudioHidden(
        _ audioHidden: MLMultiArray,
        intoSequenceHidden sequenceHidden: MLMultiArray,
        sequenceLength: Int
    ) throws {
        let hiddenSize = 3840
        guard audioHidden.dataType == .float16 else {
            throw ProbeError.unexpectedShape("audio_hidden expected float16, got \(audioHidden.dataType)")
        }
        guard audioHidden.count == audioSmokeTokenCount * hiddenSize else {
            throw ProbeError.unexpectedShape("audio_hidden expected \(audioSmokeTokenCount * hiddenSize) values, got \(audioHidden.count)")
        }
        guard sequenceHidden.dataType == .float16 else {
            throw ProbeError.unexpectedShape("sequence hidden expected float16, got \(sequenceHidden.dataType)")
        }
        guard sequenceLength >= audioSmokeTokenCount else {
            throw ProbeError.unexpectedShape("sequence length \(sequenceLength) is smaller than audio tokens \(audioSmokeTokenCount)")
        }

        let source = audioHidden.dataPointer.bindMemory(to: Float16.self, capacity: audioHidden.count)
        let destination = sequenceHidden.dataPointer.bindMemory(to: Float16.self, capacity: sequenceHidden.count)
        let startPosition = sequenceLength - audioSmokeTokenCount
        for tokenIndex in 0..<audioSmokeTokenCount {
            let sourceBase = tokenIndex * hiddenSize
            let destinationBase = (startPosition + tokenIndex) * hiddenSize
            for hiddenIndex in 0..<hiddenSize {
                destination[destinationBase + hiddenIndex] = source[sourceBase + hiddenIndex]
            }
        }
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
        let leftPadCount = startIndex == 0 ? window.prefix { $0 == 0 }.count : 0
        return TokenWindow(values: window, positionStart: startIndex, leftPadCount: leftPadCount)
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
        var count: Int
        if let override {
            count = override
        } else if let rawValue = processGeneratedTokenCountText(), let parsed = Int(rawValue) {
            count = parsed
        } else {
            count = defaultGeneratedTokenCount
        }

        // 0 = auto: run to the safety cap and stop early on <eos>.
        if count <= autoGeneratedTokenCount {
            count = autoGeneratedTokenCap
        }

        guard (1...maxGeneratedTokenCount).contains(count) else {
            throw ProbeError.invalidTokenCount("expected \(autoGeneratedTokenCount) (auto) or 1...\(maxGeneratedTokenCount), got \(count)")
        }
        return count
    }

    private static func selectedRetainedDecoderModelCount(override: Int?) throws -> Int {
        let count: Int
        if let override {
            count = override
        } else if let rawValue = processRetainedDecoderModelCountText(), let parsed = Int(rawValue) {
            count = parsed
        } else {
            count = defaultRetainedDecoderModelCount
        }

        guard (defaultRetainedDecoderModelCount...maxRetainedDecoderModelCount).contains(count) else {
            throw ProbeError.invalidTokenCount("retained decoder models expected \(defaultRetainedDecoderModelCount)...\(maxRetainedDecoderModelCount), got \(count)")
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

    private static func processRetainedDecoderModelCountText() -> String? {
        let environment = ProcessInfo.processInfo.environment
        let prefix = "--retain-decoders="
        return environment["COREML_PROBE_RETAIN_DECODERS"]
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

    private static func makePositionIDs(seqLength: Int, start: Int, leftPadCount: Int = 0) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: seqLength)], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for index in 0..<array.count {
            if index < leftPadCount {
                pointer[index] = 0
            } else {
                pointer[index] = Int32(start + index - leftPadCount)
            }
        }
        return array
    }

    private static func makeCausalMask(
        seqLength: Int,
        leftPadCount: Int = 0,
        bidirectionalBlock: Range<Int>? = nil
    ) throws -> MLMultiArray {
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
        // Gemma 4 trains image soft tokens with bidirectional attention inside
        // the block (get_block_sequence_ids_for_mask), so unmask block-internal
        // positions before re-applying pad masking.
        if let block = bidirectionalBlock {
            let clamped = max(0, block.lowerBound)..<min(seqLength, block.upperBound)
            for row in clamped {
                for column in clamped {
                    pointer[row * seqLength + column] = 0
                }
            }
        }
        for row in 0..<seqLength {
            for column in 0..<min(leftPadCount, seqLength) {
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
