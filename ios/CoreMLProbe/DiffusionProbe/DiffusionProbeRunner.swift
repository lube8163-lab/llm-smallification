import Darwin
import AVFoundation
import Foundation
import ReplayKit

struct DiffusionModelCandidate: Identifiable, Hashable {
    let quant: String
    let fileName: String
    let path: String?

    var id: String { quant }
    var isPresent: Bool { path != nil }
}

struct DiffusionChatMessage: Identifiable, Hashable, Codable {
    enum Role: String, Codable {
        case user
        case assistant
    }

    let id: UUID
    let role: Role
    let content: String

    init(id: UUID = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
}

private struct DiffusionProbePersistedState: Codable {
    let prompt: String
    let seqLen: Int
    let steps: Int
    let blockLength: Int
    let temperature: Double
    let seed: Int
    let selectedQuant: String
    let chatInput: String
    let chatMessages: [DiffusionChatMessage]
    let adaptiveQualityBoost: Bool?
}

private struct DiffusionProbeFileAutorunRequest: Codable {
    let model: String?
    let prompt: String?
    let chatTurns: [String]?
    let seqLen: Int?
    let steps: Int?
    let blockLength: Int?
    let temperature: Double?
    let seed: Int?
    let repeatCount: Int?
    let persistChanges: Bool?
    let useExistingChat: Bool?
    let adaptiveQualityBoost: Bool?
}

private struct DiffusionProbeFileAutorunResult: Codable {
    let success: Bool
    let status: String
    let output: String
    let log: String
    let finishedAt: String
}

private final class DiffusionDemoScreenRecorder {
    private let outputURL: URL
    private let queue = DispatchQueue(label: "DiffusionProbe.demoScreenRecorder")
    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var didStartSession = false

    init(outputURL: URL) {
        self.outputURL = outputURL
    }

    func start(completion: @escaping (Error?) -> Void) {
        try? FileManager.default.removeItem(at: outputURL)
        let recorder = RPScreenRecorder.shared()
        recorder.isMicrophoneEnabled = false
        guard recorder.isAvailable else {
            completion(NSError(domain: "DiffusionProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "ReplayKit is not available"]))
            return
        }

        recorder.startCapture(handler: { [weak self] sampleBuffer, sampleType, error in
            guard let self else { return }
            if let error {
                print("[DiffusionProbe] ReplayKit sample error: \(error)")
                return
            }
            guard sampleType == .video, CMSampleBufferDataIsReady(sampleBuffer) else {
                return
            }
            self.append(sampleBuffer)
        }, completionHandler: completion)
    }

    func stop(completion: @escaping (Error?) -> Void) {
        RPScreenRecorder.shared().stopCapture { [weak self] captureError in
            guard let self else {
                completion(captureError)
                return
            }
            self.queue.async {
                guard let writer = self.writer, let videoInput = self.videoInput else {
                    DispatchQueue.main.async { completion(captureError) }
                    return
                }
                videoInput.markAsFinished()
                writer.finishWriting {
                    let finalError = captureError ?? writer.error
                    DispatchQueue.main.async { completion(finalError) }
                }
            }
        }
    }

    private func append(_ sampleBuffer: CMSampleBuffer) {
        let retained = sampleBuffer
        queue.async {
            do {
                if self.writer == nil {
                    try self.prepareWriter(for: retained)
                }
                guard let writer = self.writer,
                      let videoInput = self.videoInput,
                      writer.status == .writing || writer.status == .unknown else {
                    return
                }
                let presentationTime = CMSampleBufferGetPresentationTimeStamp(retained)
                if !self.didStartSession {
                    writer.startWriting()
                    writer.startSession(atSourceTime: presentationTime)
                    self.didStartSession = true
                }
                if videoInput.isReadyForMoreMediaData {
                    videoInput.append(retained)
                }
            } catch {
                print("[DiffusionProbe] ReplayKit writer error: \(error)")
            }
        }
    }

    private func prepareWriter(for sampleBuffer: CMSampleBuffer) throws {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else {
            throw NSError(domain: "DiffusionProbe", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing video format"])
        }
        let dimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
        let width = Int(dimensions.width)
        let height = Int(dimensions.height)
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        videoInput.expectsMediaDataInRealTime = true
        if writer.canAdd(videoInput) {
            writer.add(videoInput)
        }
        self.writer = writer
        self.videoInput = videoInput
    }
}

@MainActor
final class DiffusionProbeViewModel: ObservableObject {
    @Published var prompt = "Write one short sentence about an iPhone running a diffusion language model." {
        didSet { persistNow() }
    }
    @Published var seqLen = 64 {
        didSet { persistNow() }
    }
    @Published var steps = 8 {
        didSet { persistNow() }
    }
    @Published var blockLength = 32 {
        didSet { persistNow() }
    }
    @Published var temperature = 0.0 {
        didSet { persistNow() }
    }
    @Published var seed = 1234 {
        didSet { persistNow() }
    }
    @Published var selectedQuant = "IQ4_XS" {
        didSet { persistNow() }
    }
    @Published var adaptiveQualityBoost = true {
        didSet { persistNow() }
    }
    @Published private(set) var candidates: [DiffusionModelCandidate] = []
    @Published private(set) var isRunning = false
    @Published private(set) var status = "Ready"
    @Published private(set) var output = ""
    @Published private(set) var logText = ""
    @Published var demoInputFocused = false
    @Published var demoKeyboardVisible = false
    @Published var demoKeyboardHighlightedKey: String?
    @Published var chatInput = "Can an iPhone run a diffusion language model locally?" {
        didSet { persistNow() }
    }
    @Published private(set) var chatMessages: [DiffusionChatMessage] = [] {
        didSet { persistNow() }
    }

    private static let persistedStateKey = "DiffusionProbe.persistedState.v1"
    private static let autorunRequestFileName = "diffusion-autorun-request.json"
    private static let autorunResultFileName = "diffusion-autorun-result.json"
    private static let demoRecordingFileName = "diffusion-demo-chat-recording.mp4"
    private static let demoRecordingDoneFileName = "diffusion-demo-chat-recording-done.txt"
    private static let persistedMessageLimit = 40
    private var isRestoringState = false
    private var demoRecorder: DiffusionDemoScreenRecorder?
    private var demoUsesInAppKeyboard = false

    init() {
        restorePersistedState()
        reloadCandidates()
        if !chatMessages.isEmpty {
            status = "Ready - restored chat"
        }
    }

    private static func exitAutorun(_ code: Int32) -> Never {
        fflush(nil)
        Darwin._exit(code)
    }

    var selectedCandidate: DiffusionModelCandidate? {
        candidates.first { $0.quant == selectedQuant }
    }

    func reloadCandidates() {
        candidates = ["IQ4_XS", "Q3_K_M"].map { quant in
            let fileName = "LLaDA-MoE-7B-A1B-Instruct-TD.\(quant).gguf"
            return DiffusionModelCandidate(quant: quant, fileName: fileName, path: Self.findModel(named: fileName))
        }
        if candidates.first(where: { $0.quant == selectedQuant })?.path == nil,
           let present = candidates.first(where: { $0.isPresent }) {
            selectedQuant = present.quant
        }
    }

    func autorunIfRequested() {
        let process = ProcessInfo.processInfo
        let args = process.arguments
        let env = process.environment
        let argValue: (String) -> String? = { prefix in
            args.first { $0.hasPrefix(prefix) }.map { String($0.dropFirst(prefix.count)) }
        }

        if let quant = argValue("--model=") {
            selectedQuant = quant
        }
        if let rawPrompt = argValue("--prompt=") ?? env["DIFFUSION_PROBE_PROMPT"] {
            prompt = rawPrompt
        }
        if let rawSeqLen = argValue("--seq-len="),
           let value = Int(rawSeqLen) {
            seqLen = value
        }
        if let rawSteps = argValue("--steps="),
           let value = Int(rawSteps) {
            steps = value
        }
        if let rawBlock = argValue("--block-length="),
           let value = Int(rawBlock) {
            blockLength = value
        }
        if let rawTemperature = argValue("--temperature=") ?? env["DIFFUSION_PROBE_TEMPERATURE"],
           let value = Double(rawTemperature) {
            temperature = value
        }
        if let rawSeed = argValue("--seed=") ?? env["DIFFUSION_PROBE_SEED"],
           let value = Int(rawSeed) {
            seed = value
        }
        let repeatCount = argValue("--repeat=").flatMap { Int($0) }
            ?? env["DIFFUSION_PROBE_REPEAT"].flatMap { Int($0) }
            ?? 1
        let chatTurns = argValue("--chat-turns=")
            ?? env["DIFFUSION_PROBE_CHAT_TURNS"]

        if autorunFromFileIfRequested() {
            return
        }

        if args.contains("--demo-chat") || env["DIFFUSION_PROBE_DEMO_CHAT"] == "1" {
            let recordDemo = args.contains("--demo-record") || env["DIFFUSION_PROBE_DEMO_RECORD"] == "1"
            let demoTurns = (argValue("--demo-chat-turns=") ?? env["DIFFUSION_PROBE_DEMO_CHAT_TURNS"])
                .map { rawTurns in
                    rawTurns
                        .components(separatedBy: "|||")
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                }
            runDemoChat(turns: demoTurns, record: recordDemo)
            return
        }

        if args.contains("--autorun") || env["DIFFUSION_PROBE_AUTORUN"] == "1" {
            let autoExit = env["DIFFUSION_PROBE_AUTO_EXIT"] == "1" || args.contains("--auto-exit")
            if let chatTurns, !chatTurns.isEmpty {
                let turns = chatTurns
                    .components(separatedBy: "|||")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                runChat(turns: turns, autoExit: autoExit, persistChanges: !autoExit, useExistingChat: !autoExit)
            } else {
                run(autoExit: autoExit, repeatCount: repeatCount)
            }
        }
    }

    func runDemoChat(turns: [String]? = nil, record: Bool = false) {
        guard !isRunning else { return }
        selectedQuant = "IQ4_XS"
        seqLen = 64
        steps = 8
        blockLength = 32
        temperature = 0
        seed = 1234
        adaptiveQualityBoost = true
        clearChat()
        chatInput = ""
        demoUsesInAppKeyboard = record
        demoKeyboardVisible = false
        demoKeyboardHighlightedKey = nil
        status = "Demo starting"

        let demoTurns = turns ?? [
            "Hi, I am testing LLaDA on this iPhone.",
            "Please remember that my favorite city is Kyoto and my favorite food is ramen.",
            "What am I testing on this phone?",
            "What are my favorite city and food?"
        ]

        if record, let documentsURL = Self.documentsURL() {
            let recordingURL = documentsURL.appendingPathComponent(Self.demoRecordingFileName)
            let doneURL = documentsURL.appendingPathComponent(Self.demoRecordingDoneFileName)
            try? FileManager.default.removeItem(at: doneURL)
            let recorder = DiffusionDemoScreenRecorder(outputURL: recordingURL)
            demoRecorder = recorder
            status = "Demo recording"
            recorder.start { [weak self] error in
                guard let self else { return }
                if let error {
                    self.status = "Demo recording unavailable"
                    print("[DiffusionProbe] ReplayKit start failed: \(error)")
                }
                self.runDemoTurn(demoTurns, index: 0)
            }
        } else {
            runDemoTurn(demoTurns, index: 0)
        }
    }

    private func runDemoTurn(_ turns: [String], index: Int) {
        guard index < turns.count else {
            finishDemo()
            return
        }

        let turn = turns[index]
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: index == 0 ? 900_000_000 : 1_200_000_000)
            demoInputFocused = !demoUsesInAppKeyboard
            demoKeyboardVisible = demoUsesInAppKeyboard
            chatInput = ""
            for character in turn {
                chatInput.append(character)
                demoKeyboardHighlightedKey = Self.demoKeyboardKey(for: character)
                try? await Task.sleep(nanoseconds: 45_000_000)
            }
            demoKeyboardHighlightedKey = "RETURN"
            try? await Task.sleep(nanoseconds: 180_000_000)
            demoKeyboardHighlightedKey = nil
            demoKeyboardVisible = false
            try? await Task.sleep(nanoseconds: 350_000_000)
            runChat(turns: [turn], persistChanges: true, useExistingChat: true) { [weak self] _, _, _, _ in
                guard let self else { return }
                Task { @MainActor in
                    self.demoInputFocused = false
                    self.demoKeyboardVisible = false
                    self.demoKeyboardHighlightedKey = nil
                    self.chatInput = ""
                    try? await Task.sleep(nanoseconds: 1_200_000_000)
                    self.runDemoTurn(turns, index: index + 1)
                }
            }
        }
    }

    private func finishDemo() {
        demoInputFocused = false
        demoKeyboardVisible = false
        demoKeyboardHighlightedKey = nil
        chatInput = ""
        guard let recorder = demoRecorder else {
            status = "Demo complete"
            return
        }
        status = "Saving demo"
        recorder.stop { [weak self] error in
            guard let self else { return }
            if let error {
                self.status = "Demo recording failed"
                Self.writeDemoRecordingDone("failed: \(error)")
            } else {
                self.status = "Demo recording saved"
                Self.writeDemoRecordingDone("ok")
            }
            self.demoRecorder = nil
        }
    }

    func run(autoExit: Bool = false,
             repeatCount: Int = 1,
             completion: ((Bool, String, String, String) -> Void)? = nil) {
        reloadCandidates()
        guard !isRunning else { return }
        guard let model = selectedCandidate, let modelPath = model.path else {
            status = "Missing \(selectedQuant) model"
            output = ""
            logText = modelSearchSummary()
            completion?(false, status, output, logText)
            if autoExit {
                Self.exitAutorun(2)
            }
            return
        }

        isRunning = true
        status = "Running \(model.quant)"
        output = ""
        logText = ""

        let prompt = prompt
        let seqLen = Int32(seqLen)
        let steps = Int32(steps)
        let blockLength = Int32(blockLength)
        let temperature = Float(temperature)
        let seed = Int32(seed)
        let repeatCount = max(1, repeatCount)

        Task {
            var combinedLog = ""
            var combinedOutput: [String] = []
            var finalStatus = ""
            var success = true

            for runIndex in 1...repeatCount {
                print("[DiffusionProbe] run \(runIndex)/\(repeatCount)")
                let result = await Task.detached(priority: .userInitiated) {
                    DiffusionBridge.run(modelPath: modelPath,
                                        prompt: prompt,
                                        seqLen: seqLen,
                                        steps: steps,
                                        blockLength: blockLength,
                                        temperature: temperature,
                                        seed: seed,
                                        formattedPrompt: false)
                }.value

                combinedLog += "[DiffusionProbe] run \(runIndex)/\(repeatCount)\n"
                combinedLog += result.log
                if !combinedLog.hasSuffix("\n") {
                    combinedLog += "\n"
                }
                if !result.output.isEmpty {
                    combinedOutput.append("Run \(runIndex): \(result.output)")
                }
                finalStatus = result.summary
                success = result.success

                if !result.success {
                    break
                }
            }

            status = finalStatus
            let finalOutput = combinedOutput.joined(separator: "\n\n")
            output = finalOutput
            logText = combinedLog
            isRunning = false
            completion?(success, finalStatus, finalOutput, combinedLog)

            if autoExit {
                Self.exitAutorun(success ? 0 : 1)
            }
        }
    }

    func sendChat() {
        let turn = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !turn.isEmpty else { return }
        chatInput = ""
        runChat(turns: [turn])
    }

    func clearChat() {
        chatMessages = []
        output = ""
        logText = ""
        status = "Ready"
    }

    func runChat(turns: [String],
                 autoExit: Bool = false,
                 persistChanges: Bool = true,
                 useExistingChat: Bool = true,
                 completion: ((Bool, String, String, String) -> Void)? = nil) {
        reloadCandidates()
        guard !isRunning else { return }
        guard let model = selectedCandidate, let modelPath = model.path else {
            status = "Missing \(selectedQuant) model"
            output = ""
            logText = modelSearchSummary()
            completion?(false, status, output, logText)
            if autoExit {
                Self.exitAutorun(2)
            }
            return
        }

        let cleanTurns = turns
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !cleanTurns.isEmpty else {
            completion?(true, status, output, logText)
            if autoExit {
                Self.exitAutorun(0)
            }
            return
        }

        isRunning = true
        status = "Running chat"
        output = ""
        logText = ""

        let requestedSeqLen = Int32(seqLen)
        let steps = Int32(steps)
        let blockLength = Int32(blockLength)
        let temperature = Float(temperature)
        let seed = Int32(seed)
        let adaptiveQualityBoost = adaptiveQualityBoost
        var localMessages = useExistingChat ? chatMessages : []

        Task {
            var combinedLog = ""
            var combinedOutput: [String] = []
            var success = true
            var finalStatus = ""

            for (index, turn) in cleanTurns.enumerated() {
                localMessages.append(DiffusionChatMessage(role: .user, content: turn))
                let formattedPrompt = Self.formattedChatPrompt(messages: localMessages)
                print("[DiffusionProbe] chat turn \(index + 1)/\(cleanTurns.count)")

                var result = await Self.runFormattedPrompt(modelPath: modelPath,
                                                           prompt: formattedPrompt,
                                                           seqLen: requestedSeqLen,
                                                           steps: steps,
                                                           blockLength: blockLength,
                                                           temperature: temperature,
                                                           seed: seed)
                if !result.success,
                   requestedSeqLen < 128,
                   (result.summary.localizedCaseInsensitiveContains("Prompt is too long")
                    || result.summary.localizedCaseInsensitiveContains("too little room")) {
                    let retrySteps = adaptiveQualityBoost ? max(steps, 12) : steps
                    combinedLog += "[DiffusionProbe] retrying chat turn \(index + 1) with seqLen=128\n"
                    result = await Self.runFormattedPrompt(modelPath: modelPath,
                                                           prompt: formattedPrompt,
                                                           seqLen: 128,
                                                           steps: retrySteps,
                                                           blockLength: blockLength,
                                                           temperature: temperature,
                                                           seed: seed)
                }

                combinedLog += "[DiffusionProbe] chat turn \(index + 1)/\(cleanTurns.count)\n"
                combinedLog += result.log
                if !combinedLog.hasSuffix("\n") {
                    combinedLog += "\n"
                }
                finalStatus = result.summary
                success = result.success

                if result.success {
                    let cleanedOutput = Self.cleanedOutput(result.output)
                    localMessages.append(DiffusionChatMessage(role: .assistant, content: cleanedOutput))
                    combinedOutput.append("User: \(turn)\nAssistant: \(cleanedOutput)")
                } else {
                    combinedOutput.append("User: \(turn)\nAssistant: \(result.summary)")
                    break
                }
            }

            if persistChanges {
                chatMessages = localMessages
            }
            status = finalStatus
            let finalOutput = combinedOutput.joined(separator: "\n\n")
            output = finalOutput
            logText = combinedLog
            isRunning = false
            completion?(success, finalStatus, finalOutput, combinedLog)

            if autoExit {
                Self.exitAutorun(success ? 0 : 1)
            }
        }
    }

    func persistNow() {
        guard !isRestoringState else { return }
        let persistedMessages = Array(chatMessages.suffix(Self.persistedMessageLimit))
        let state = DiffusionProbePersistedState(prompt: prompt,
                                                 seqLen: seqLen,
                                                 steps: steps,
                                                 blockLength: blockLength,
                                                 temperature: temperature,
                                                 seed: seed,
                                                 selectedQuant: selectedQuant,
                                                 chatInput: chatInput,
                                                 chatMessages: persistedMessages,
                                                 adaptiveQualityBoost: adaptiveQualityBoost)
        guard let data = try? JSONEncoder().encode(state) else { return }
        UserDefaults.standard.set(data, forKey: Self.persistedStateKey)
    }

    private func autorunFromFileIfRequested() -> Bool {
        guard let requestURL = Self.documentsURL()?.appendingPathComponent(Self.autorunRequestFileName) else {
            return false
        }
        let fm = FileManager.default
        guard fm.fileExists(atPath: requestURL.path) else {
            return false
        }

        do {
            let data = try Data(contentsOf: requestURL)
            let request = try JSONDecoder().decode(DiffusionProbeFileAutorunRequest.self, from: data)
            try? fm.removeItem(at: requestURL)

            if let model = request.model {
                selectedQuant = model
            }
            if let prompt = request.prompt {
                self.prompt = prompt
            }
            if let seqLen = request.seqLen {
                self.seqLen = min(max(seqLen, 64), 256)
            }
            if let steps = request.steps {
                self.steps = min(max(steps, 4), 64)
            }
            if let blockLength = request.blockLength {
                self.blockLength = min(max(blockLength, 16), 128)
            }
            if let temperature = request.temperature {
                self.temperature = min(max(temperature, 0), 2)
            }
            if let seed = request.seed {
                self.seed = min(max(seed, 0), 999_999)
            }
            if let adaptiveQualityBoost = request.adaptiveQualityBoost {
                self.adaptiveQualityBoost = adaptiveQualityBoost
            }

            let completion: (Bool, String, String, String) -> Void = { success, status, output, log in
                Self.writeAutorunResult(success: success, status: status, output: output, log: log)
            }

            if let chatTurns = request.chatTurns?.filter({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
               !chatTurns.isEmpty {
                runChat(turns: chatTurns,
                        persistChanges: request.persistChanges ?? false,
                        useExistingChat: request.useExistingChat ?? false,
                        completion: completion)
            } else {
                run(repeatCount: max(1, request.repeatCount ?? 1), completion: completion)
            }
        } catch {
            Self.writeAutorunResult(success: false,
                                    status: "Failed to read autorun request",
                                    output: "",
                                    log: String(describing: error))
        }

        return true
    }

    private func restorePersistedState() {
        guard let data = UserDefaults.standard.data(forKey: Self.persistedStateKey),
              let state = try? JSONDecoder().decode(DiffusionProbePersistedState.self, from: data) else {
            return
        }

        isRestoringState = true
        prompt = state.prompt
        seqLen = min(max(state.seqLen, 64), 256)
        steps = min(max(state.steps, 4), 64)
        blockLength = min(max(state.blockLength, 16), 128)
        temperature = min(max(state.temperature, 0), 2)
        seed = min(max(state.seed, 0), 999_999)
        selectedQuant = state.selectedQuant
        chatInput = state.chatInput
        chatMessages = Array(state.chatMessages.suffix(Self.persistedMessageLimit))
        adaptiveQualityBoost = state.adaptiveQualityBoost ?? true
        isRestoringState = false
    }

    func modelSearchSummary() -> String {
        var lines: [String] = ["Model search:"]
        for candidate in candidates {
            lines.append("- \(candidate.fileName): \(candidate.path ?? "not found")")
        }
        lines.append("")
        lines.append("Set DIFFUSION_PROBE_MODEL_PATH or place a GGUF in the app container root or Documents directory.")
        return lines.joined(separator: "\n")
    }

    private static func findModel(named fileName: String) -> String? {
        let fm = FileManager.default
        if let override = ProcessInfo.processInfo.environment["DIFFUSION_PROBE_MODEL_PATH"],
           fm.fileExists(atPath: override) {
            return override
        }

        var roots: [URL] = [
            URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        ]
        let directoryKinds: [FileManager.SearchPathDirectory] = [
            .documentDirectory,
            .applicationSupportDirectory,
            .cachesDirectory
        ]
        for kind in directoryKinds {
            roots.append(contentsOf: fm.urls(for: kind, in: .userDomainMask))
        }
        if let resourceURL = Bundle.main.resourceURL {
            roots.append(resourceURL)
        }

        for root in roots {
            let path = root.appendingPathComponent(fileName).path
            if fm.fileExists(atPath: path) {
                return path
            }
        }
        return nil
    }

    private static func documentsURL() -> URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    }

    private static func writeAutorunResult(success: Bool, status: String, output: String, log: String) {
        guard let resultURL = documentsURL()?.appendingPathComponent(autorunResultFileName) else {
            return
        }
        let result = DiffusionProbeFileAutorunResult(success: success,
                                                     status: status,
                                                     output: output,
                                                     log: log,
                                                     finishedAt: ISO8601DateFormatter().string(from: Date()))
        guard let data = try? JSONEncoder().encode(result) else {
            return
        }
        try? data.write(to: resultURL, options: .atomic)
    }

    private static func writeDemoRecordingDone(_ message: String) {
        guard let doneURL = documentsURL()?.appendingPathComponent(demoRecordingDoneFileName) else {
            return
        }
        try? message.write(to: doneURL, atomically: true, encoding: .utf8)
    }

    private static func formattedChatPrompt(messages: [DiffusionChatMessage]) -> String {
        let userMessages = messages
            .filter { $0.role == .user }
            .suffix(6)
        guard let latest = userMessages.last else {
            return "<role>SYSTEM</role>Reply shortly.\ndetailed thinking off<|role_end|><role>ASSISTANT</role>"
        }

        var userPrompt = ""
        let previousMessages = userMessages.dropLast()
        if !previousMessages.isEmpty {
            let earlier = previousMessages
                .map(\.content)
                .joined(separator: " ")
            userPrompt += "Earlier: \(earlier)\n"
        }
        userPrompt += "Now: \(latest.content)\n"
        userPrompt += previousMessages.isEmpty
            ? "Answer in one short sentence."
            : "Use Earlier. Answer in one short sentence."

        return "<role>SYSTEM</role>Reply shortly.\ndetailed thinking off<|role_end|><role>HUMAN</role>\(userPrompt)<|role_end|><role>ASSISTANT</role>"
    }

    private static func cleanedOutput(_ output: String) -> String {
        var cleaned = output.trimmingCharacters(in: .whitespacesAndNewlines)
        cleaned = cleaned.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: #"(\b[\p{L}\p{N}']+\b)(\s+\1\b)+"#,
                                               with: "$1",
                                               options: [.regularExpression, .caseInsensitive])
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func demoKeyboardKey(for character: Character) -> String {
        if character == " " {
            return "SPACE"
        }
        let key = String(character).uppercased()
        switch key {
        case ",", ";", ":":
            return "."
        case "!":
            return "?"
        default:
            return key
        }
    }

    private static func runFormattedPrompt(modelPath: String,
                                           prompt: String,
                                           seqLen: Int32,
                                           steps: Int32,
                                           blockLength: Int32,
                                           temperature: Float,
                                           seed: Int32) async -> DiffusionBridgeResult {
        await Task.detached(priority: .userInitiated) {
            DiffusionBridge.run(modelPath: modelPath,
                                prompt: prompt,
                                seqLen: seqLen,
                                steps: steps,
                                blockLength: blockLength,
                                temperature: temperature,
                                seed: seed,
                                formattedPrompt: true)
        }.value
    }
}
