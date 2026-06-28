import SwiftUI

enum AppTab: Hashable {
    case chat
    case probe
}

struct ContentView: View {
    @StateObject private var probeViewModel: ProbeViewModel
    @StateObject private var chatViewModel: ChatViewModel
    @State private var selectedTab: AppTab
    private let autoRun: Bool

    init() {
        let shouldAutoRun = ProcessInfo.processInfo.arguments.contains("--autorun")
            || ProcessInfo.processInfo.environment["COREML_PROBE_AUTORUN"] == "1"
        autoRun = shouldAutoRun
        _selectedTab = State(initialValue: shouldAutoRun ? .probe : .chat)
        _probeViewModel = StateObject(wrappedValue: ProbeViewModel())
        _chatViewModel = StateObject(wrappedValue: ChatViewModel())
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            ChatScreen(viewModel: chatViewModel)
                .tabItem {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right.fill")
                }
                .tag(AppTab.chat)

            ProbeScreen(viewModel: probeViewModel)
                .tabItem {
                    Label("Probe", systemImage: "waveform.path.ecg.rectangle")
                }
                .tag(AppTab.probe)
        }
        .task {
            probeViewModel.runIfRequested(autoRun)
        }
    }
}

struct ChatScreen: View {
    @ObservedObject var viewModel: ChatViewModel

    var body: some View {
        NavigationStack {
            List {
                Section("Generation") {
                    LabeledContent("Mode", value: ProbeRunMode.generateTokenLoop.title)

                    Picker("Endpoints", selection: $viewModel.endpointComputeSelection) {
                        ForEach(ProbeComputeSelection.endpointCases) { selection in
                            Text(selection.title).tag(selection)
                        }
                    }

                    Picker("Decoder", selection: $viewModel.decoderComputeSelection) {
                        ForEach(ProbeComputeSelection.allCases) { selection in
                            Text(selection.title).tag(selection)
                        }
                    }

                    Picker("Layers", selection: $viewModel.layerSelection) {
                        ForEach(ProbeLayerSelection.allCases) { selection in
                            Text(selection.title).tag(selection)
                        }
                    }

                    Picker("Cache", selection: $viewModel.cacheClearPolicy) {
                        ForEach(ProbeCacheClearPolicy.allCases) { policy in
                            Text(policy.title).tag(policy)
                        }
                    }

                    Stepper(value: $viewModel.generatedTokenCount, in: 1...ProbeRunner.maxGeneratedTokenCount) {
                        LabeledContent("Tokens", value: "\(viewModel.generatedTokenCount)")
                    }

                    Picker("Window", selection: $viewModel.sequenceLength) {
                        ForEach(ProbeSequenceLength.allCases) { sequenceLength in
                            Text(sequenceLength.title).tag(sequenceLength)
                        }
                    }
                    .disabled(viewModel.isGenerating)

                    TextField("Token window", text: $viewModel.inputIDsText)
                        .keyboardType(.numbersAndPunctuation)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Button {
                        viewModel.resetInputWindow()
                    } label: {
                        Label("Reset Window", systemImage: "arrow.counterclockwise")
                    }
                    .disabled(viewModel.isGenerating)
                }

                Section("Conversation") {
                    if viewModel.messages.isEmpty {
                        Label("Ready", systemImage: "text.bubble")
                            .foregroundStyle(.secondary)
                    }

                    ForEach(viewModel.messages) { message in
                        ChatMessageRow(message: message)
                            .listRowSeparator(.hidden)
                    }

                    if viewModel.isGenerating {
                        HStack {
                            ProgressView()
                            Text("Generating")
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                    }
                }

                Section("Status") {
                    LabeledContent("Memory", value: viewModel.currentMemoryText)
                    LabeledContent("Result", value: viewModel.summary)
                    if !viewModel.generatedTokenText.isEmpty {
                        LabeledContent("Tokens", value: viewModel.generatedTokenText)
                    }
                }

                if !viewModel.steps.isEmpty {
                    Section("Recent Steps") {
                        ForEach(viewModel.recentSteps) { step in
                            ProbeStepRow(step: step)
                        }
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Gemma Chat")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        viewModel.clear()
                    } label: {
                        Label("Clear", systemImage: "trash")
                    }
                    .disabled(viewModel.isGenerating)
                }
            }
            .safeAreaInset(edge: .bottom) {
                ChatComposer(
                    text: $viewModel.messageText,
                    isGenerating: viewModel.isGenerating,
                    send: viewModel.send
                )
            }
        }
    }
}

struct ChatComposer: View {
    @Binding var text: String
    let isGenerating: Bool
    let send: () -> Void
    @FocusState private var isFocused: Bool

    private var canSend: Bool {
        !isGenerating && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            Menu {
                Button {
                } label: {
                    Label("Photo", systemImage: "photo")
                }
                .disabled(true)

                Button {
                } label: {
                    Label("Audio", systemImage: "waveform")
                }
                .disabled(true)
            } label: {
                Image(systemName: "plus.circle.fill")
                    .font(.title3)
                    .frame(width: 36, height: 36)
            }
            .disabled(isGenerating)

            TextField("Message", text: $text, axis: .vertical)
                .lineLimit(1...4)
                .textInputAutocapitalization(.sentences)
                .focused($isFocused)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
                .onSubmit {
                    if canSend {
                        send()
                    }
                }

            Button {
                send()
            } label: {
                Image(systemName: "paperplane.fill")
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canSend)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.bar)
    }
}

struct ChatMessageRow: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .bottom) {
            if message.role == .user {
                Spacer(minLength: 48)
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                Text(message.role.title)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 8) {
                    Text(message.text)
                        .font(.body)
                        .textSelection(.enabled)

                    if !message.tokens.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                ForEach(Array(message.tokens.enumerated()), id: \.offset) { _, token in
                                    Text("#\(token)")
                                        .font(.caption.monospacedDigit())
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(tokenBackground, in: RoundedRectangle(cornerRadius: 8))
                                }
                            }
                        }
                    }

                    if let detail = message.detail, !detail.isEmpty {
                        Text(detail)
                            .font(.caption)
                            .foregroundStyle(message.isError ? .red : .secondary)
                            .textSelection(.enabled)
                    }
                }
                .padding(10)
                .foregroundStyle(foregroundStyle)
                .background(bubbleBackground, in: RoundedRectangle(cornerRadius: 8))
            }
            .frame(maxWidth: 330, alignment: message.role == .user ? .trailing : .leading)

            if message.role == .assistant {
                Spacer(minLength: 48)
            }
        }
        .padding(.vertical, 3)
    }

    private var bubbleBackground: Color {
        if message.isError {
            return Color.red.opacity(0.12)
        }
        return message.role == .user ? Color.accentColor : Color(.secondarySystemGroupedBackground)
    }

    private var tokenBackground: Color {
        message.role == .user ? Color.white.opacity(0.18) : Color(.tertiarySystemGroupedBackground)
    }

    private var foregroundStyle: Color {
        message.role == .user ? .white : .primary
    }
}

struct ProbeScreen: View {
    @ObservedObject var viewModel: ProbeViewModel

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Picker("Mode", selection: $viewModel.runMode) {
                        ForEach(ProbeRunMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    Picker("Endpoints", selection: $viewModel.endpointComputeSelection) {
                        ForEach(ProbeComputeSelection.endpointCases) { selection in
                            Text(selection.title).tag(selection)
                        }
                    }

                    Picker("Decoder", selection: $viewModel.decoderComputeSelection) {
                        ForEach(ProbeComputeSelection.allCases) { selection in
                            Text(selection.title).tag(selection)
                        }
                    }
                    Picker("Layers", selection: $viewModel.layerSelection) {
                        ForEach(ProbeLayerSelection.allCases) { selection in
                            Text(selection.title).tag(selection)
                        }
                    }
                    Picker("Cache", selection: $viewModel.cacheClearPolicy) {
                        ForEach(ProbeCacheClearPolicy.allCases) { policy in
                            Text(policy.title).tag(policy)
                        }
                    }
                    if viewModel.runMode.usesInputIDs {
                        Picker("Window", selection: $viewModel.sequenceLength) {
                            ForEach(ProbeSequenceLength.allCases) { sequenceLength in
                                Text(sequenceLength.title).tag(sequenceLength)
                            }
                        }
                        .disabled(viewModel.isRunning)

                        TextField("Input IDs", text: $viewModel.inputIDsText)
                            .keyboardType(.numbersAndPunctuation)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    }
                    if viewModel.runMode.usesGeneratedTokenCount {
                        Stepper(value: $viewModel.generatedTokenCount, in: 1...ProbeRunner.maxGeneratedTokenCount) {
                            LabeledContent("Tokens", value: "\(viewModel.generatedTokenCount)")
                        }
                    }
                    Button {
                        viewModel.run()
                    } label: {
                        Label(viewModel.isRunning ? "Running" : "Run Probe", systemImage: "play.fill")
                    }
                    .disabled(viewModel.isRunning)

                    Button {
                        viewModel.clear()
                    } label: {
                        Label("Clear", systemImage: "trash")
                    }
                    .disabled(viewModel.isRunning && viewModel.steps.isEmpty)
                }

                Section("Status") {
                    LabeledContent("Memory", value: viewModel.currentMemoryText)
                    LabeledContent("Result", value: viewModel.summary)
                }

                Section("Steps") {
                    ForEach(viewModel.steps) { step in
                        ProbeStepRow(step: step)
                    }
                }
            }
            .navigationTitle("Core ML Probe")
        }
    }
}

struct ProbeStepRow: View {
    let step: ProbeStep

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(step.name)
                    .font(.headline)
                Spacer()
                Text(step.durationText)
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            HStack {
                Text(step.memoryText)
                Spacer()
                Text(step.detail)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

enum ChatRole {
    case user
    case assistant

    var title: String {
        switch self {
        case .user: "User"
        case .assistant: "Assistant"
        }
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: ChatRole
    let text: String
    let tokens: [Int]
    let detail: String?
    let isError: Bool
}

enum TokenDisplay {
    private static let knownTokens: [Int: String] = [
        2: "<bos>",
        255999: "<boi>",
        256000: "<boa>",
        258880: "<image>",
        258881: "<audio>",
        258882: "<eoi>",
        258883: "<eoa>"
    ]

    static func label(for tokenID: Int) -> String {
        knownTokens[tokenID] ?? "#\(tokenID)"
    }

    static func joinedLabels(for tokenIDs: [Int]) -> String {
        tokenIDs.map(label(for:)).joined(separator: " ")
    }
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var endpointComputeSelection = ProbeComputeSelection.selectedEndpointFromProcess(default: .cpuOnly)
    @Published var decoderComputeSelection = ProbeComputeSelection.selectedDecoderFromProcess(default: .cpuAndGPU)
    @Published var layerSelection = ProbeLayerSelection.selectedFromProcess(default: .first48)
    @Published var cacheClearPolicy = ProbeCacheClearPolicy.selectedFromProcess(default: .runEndOnly)
    @Published var sequenceLength: ProbeSequenceLength {
        didSet {
            guard oldValue != sequenceLength else { return }
            inputIDsText = ProbeRunner.defaultInputIDsText(sequenceLength: sequenceLength)
        }
    }
    @Published var inputIDsText: String
    @Published var generatedTokenCount = ProbeRunner.selectedGeneratedTokenCountFromProcess()
    @Published var messageText = ""
    @Published var isGenerating = false
    @Published var messages: [ChatMessage] = []
    @Published var steps: [ProbeStep] = []
    @Published var summary = "Idle"
    @Published var generatedTokenText = ""
    @Published var currentMemoryText = ProbeMemory.currentText()

    var recentSteps: [ProbeStep] {
        Array(steps.suffix(16))
    }

    init() {
        let sequenceLength = ProbeRunner.selectedSequenceLengthFromProcess()
        self.sequenceLength = sequenceLength
        self.inputIDsText = ProbeRunner.selectedInputIDsTextFromProcess(sequenceLength: sequenceLength)
    }

    func clear() {
        messages.removeAll()
        steps.removeAll()
        summary = "Idle"
        generatedTokenText = ""
        currentMemoryText = ProbeMemory.currentText()
    }

    func resetInputWindow() {
        inputIDsText = ProbeRunner.defaultInputIDsText(sequenceLength: sequenceLength)
        summary = "Input window reset"
        currentMemoryText = ProbeMemory.currentText()
    }

    func send() {
        let text = messageText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating else { return }

        let resolvedWindow: TokenWindowResolution
        do {
            resolvedWindow = try Self.resolveTokenWindow(
                messageText: text,
                fallbackText: inputIDsText,
                sequenceLength: sequenceLength
            )
        } catch {
            messageText = ""
            summary = error.localizedDescription
            generatedTokenText = ""
            currentMemoryText = ProbeMemory.currentText()
            messages.append(ChatMessage(
                role: .user,
                text: text,
                tokens: [],
                detail: nil,
                isError: false
            ))
            messages.append(ChatMessage(
                role: .assistant,
                text: "Invalid token window",
                tokens: [],
                detail: error.localizedDescription,
                isError: true
            ))
            return
        }

        let inputIDsText = resolvedWindow.text
        let generatedTokenCount = generatedTokenCount
        let computePlan = ProbeComputePlan(endpoint: endpointComputeSelection, decoder: decoderComputeSelection)
        let layerSelection = layerSelection
        let cacheClearPolicy = cacheClearPolicy
        let sequenceLength = self.sequenceLength

        if resolvedWindow.shouldUpdateInputField {
            self.inputIDsText = inputIDsText
        }
        messageText = ""
        isGenerating = true
        summary = "Generating"
        generatedTokenText = ""
        currentMemoryText = ProbeMemory.currentText()
        messages.append(ChatMessage(
            role: .user,
            text: text,
            tokens: [],
            detail: resolvedWindow.detail,
            isError: false
        ))

        Task {
            let result = await Task.detached(priority: .userInitiated) {
                ProbeRunner.run(
                    computePlan: computePlan,
                    mode: .generateTokenLoop,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
                    inputIDsText: inputIDsText,
                    generatedTokenCount: generatedTokenCount
                )
            }.value

            switch result {
            case .success(let report):
                let tokens = Self.generatedTokenIDs(from: report)
                let tokenText = tokens.map { "#\($0)" }.joined(separator: ", ")
                let displayText = TokenDisplay.joinedLabels(for: tokens)
                steps = report.steps
                summary = report.summary
                generatedTokenText = tokenText
                messages.append(ChatMessage(
                    role: .assistant,
                    text: displayText.isEmpty ? report.summary : displayText,
                    tokens: tokens,
                    detail: Self.generatedTokenDetail(from: report),
                    isError: false
                ))
                if let nextWindow = Self.slidTokenWindow(
                    from: inputIDsText,
                    appending: tokens,
                    sequenceLength: sequenceLength
                ) {
                    self.inputIDsText = nextWindow
                }
            case .failure(let error):
                steps = error.steps
                summary = error.message
                generatedTokenText = ""
                messages.append(ChatMessage(
                    role: .assistant,
                    text: "Generation failed",
                    tokens: [],
                    detail: error.message,
                    isError: true
                ))
            }

            currentMemoryText = ProbeMemory.currentText()
            isGenerating = false
        }
    }

    private static func generatedTokenIDs(from report: ProbeReport) -> [Int] {
        tokenIDs(from: generatedTokenDetail(from: report) ?? report.summary)
    }

    private static func generatedTokenDetail(from report: ProbeReport) -> String? {
        report.steps.last(where: { $0.name == "Generated tokens" })?.detail
    }

    private static func slidTokenWindow(
        from rawWindow: String,
        appending tokenIDs: [Int],
        sequenceLength: ProbeSequenceLength
    ) -> String? {
        guard !tokenIDs.isEmpty else { return nil }
        guard var window = try? parseTokenWindow(rawWindow, sequenceLength: sequenceLength, allowLong: true) else { return nil }
        guard window.count == sequenceLength.rawValue else { return nil }

        for tokenID in tokenIDs {
            guard let value = Int32(exactly: tokenID) else { return nil }
            window.removeFirst()
            window.append(value)
        }

        return window.map(String.init).joined(separator: ",")
    }

    private struct TokenWindowResolution {
        let text: String
        let detail: String
        let shouldUpdateInputField: Bool
    }

    private static func resolveTokenWindow(
        messageText: String,
        fallbackText: String,
        sequenceLength: ProbeSequenceLength
    ) throws -> TokenWindowResolution {
        if let ids = try tokenWindowFromMessage(messageText, sequenceLength: sequenceLength) {
            let text = formatTokenWindow(ids)
            return TokenWindowResolution(
                text: text,
                detail: tokenWindowDetail(text: text, source: "from message IDs", ids: ids, sequenceLength: sequenceLength),
                shouldUpdateInputField: true
            )
        }

        let ids = try parseTokenWindow(fallbackText, sequenceLength: sequenceLength, allowLong: false)
        let text = formatTokenWindow(ids)
        return TokenWindowResolution(
            text: text,
            detail: tokenWindowDetail(
                text: text,
                source: "from Token window; message text is not tokenized yet",
                ids: ids,
                sequenceLength: sequenceLength
            ),
            shouldUpdateInputField: false
        )
    }

    private static func tokenWindowDetail(
        text: String,
        source: String,
        ids: [Int32],
        sequenceLength: ProbeSequenceLength
    ) -> String {
        var parts = ["\(sequenceLength.title) \(text)", source]
        if let repeatedToken = repeatedToken(in: ids) {
            parts.append("repeated #\(repeatedToken)")
        }
        return parts.joined(separator: " | ")
    }

    private static func tokenWindowFromMessage(_ text: String, sequenceLength: ProbeSequenceLength) throws -> [Int32]? {
        let keys = ["input_ids_last\(sequenceLength.rawValue)=", "input_ids_last4=", "input_ids="]
        for key in keys {
            if let keyedValue = valueAfterKey(key, in: text) {
                return try parseTokenWindow(
                    keyedValue,
                    sequenceLength: sequenceLength,
                    allowLong: key == "input_ids="
                )
            }
        }

        let hashIDs = tokenIDs(from: text)
        if hashIDs.count >= sequenceLength.rawValue {
            return try int32Window(from: hashIDs, sequenceLength: sequenceLength)
        }

        return try parseBareTokenWindow(text, sequenceLength: sequenceLength)
    }

    private static func valueAfterKey(_ key: String, in text: String) -> String? {
        guard let range = text.range(of: key) else { return nil }
        let remainder = text[range.upperBound...]
        let line = remainder.split(whereSeparator: \.isNewline).first.map(String.init) ?? String(remainder)
        return line.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func parseBareTokenWindow(_ rawWindow: String, sequenceLength: ProbeSequenceLength) throws -> [Int32]? {
        let trimmed = rawWindow.trimmingCharacters(in: .whitespacesAndNewlines)
        let allowed = CharacterSet(charactersIn: "0123456789,#[] \t\r\n")
        guard !trimmed.isEmpty,
              trimmed.unicodeScalars.allSatisfy({ allowed.contains($0) }) else {
            return nil
        }

        let normalized = trimmed
            .replacingOccurrences(of: "#", with: "")
            .replacingOccurrences(of: "[", with: " ")
            .replacingOccurrences(of: "]", with: " ")
        let values = normalized.split { character in
            character == "," || character.isWhitespace
        }
        guard values.count == sequenceLength.rawValue else {
            return nil
        }
        return try parseTokenWindow(trimmed, sequenceLength: sequenceLength, allowLong: false)
    }

    private static func parseTokenWindow(
        _ rawWindow: String,
        sequenceLength: ProbeSequenceLength,
        allowLong: Bool
    ) throws -> [Int32] {
        let normalized = rawWindow
            .replacingOccurrences(of: "#", with: "")
            .replacingOccurrences(of: "[", with: " ")
            .replacingOccurrences(of: "]", with: " ")
        let values = normalized
            .split { character in
                character == "," || character.isWhitespace
            }
            .map(String.init)
        if allowLong {
            guard values.count >= sequenceLength.rawValue else {
                throw ProbeError.invalidInputIDs("expected at least \(sequenceLength.rawValue) token IDs, got \(values.count)")
            }
        } else {
            guard values.count == sequenceLength.rawValue else {
                throw ProbeError.invalidInputIDs("expected exactly \(sequenceLength.rawValue) token IDs, got \(values.count)")
            }
        }

        let parsed = try values.map { value in
            guard let parsed = Int32(value) else {
                throw ProbeError.invalidInputIDs("not an Int32 token ID: \(value)")
            }
            return parsed
        }
        return Array(parsed.suffix(sequenceLength.rawValue))
    }

    private static func int32Window(from values: [Int], sequenceLength: ProbeSequenceLength) throws -> [Int32] {
        guard values.count >= sequenceLength.rawValue else {
            throw ProbeError.invalidInputIDs("expected at least \(sequenceLength.rawValue) token IDs, got \(values.count)")
        }

        let parsed = try values.map { value in
            guard let parsed = Int32(exactly: value) else {
                throw ProbeError.invalidInputIDs("not an Int32 token ID: \(value)")
            }
            return parsed
        }
        return Array(parsed.suffix(sequenceLength.rawValue))
    }

    private static func formatTokenWindow(_ values: [Int32]) -> String {
        values.map(String.init).joined(separator: ",")
    }

    private static func repeatedToken(in values: [Int32]) -> Int32? {
        guard let first = values.first,
              values.dropFirst().allSatisfy({ $0 == first }) else {
            return nil
        }
        return first
    }

    private static func tokenIDs(from text: String) -> [Int] {
        var tokens: [Int] = []
        var scanner = text.startIndex

        while scanner < text.endIndex {
            guard text[scanner] == "#" else {
                scanner = text.index(after: scanner)
                continue
            }

            var digitIndex = text.index(after: scanner)
            var digits = ""
            while digitIndex < text.endIndex, text[digitIndex].isNumber {
                digits.append(text[digitIndex])
                digitIndex = text.index(after: digitIndex)
            }

            if let token = Int(digits) {
                tokens.append(token)
            }
            scanner = digitIndex
        }

        return tokens
    }
}

@MainActor
final class ProbeViewModel: ObservableObject {
    @Published var runMode = ProbeRunMode.selectedFromProcess(default: .loadEmbedding)
    @Published var endpointComputeSelection = ProbeComputeSelection.selectedEndpointFromProcess(default: .cpuOnly)
    @Published var decoderComputeSelection = ProbeComputeSelection.selectedDecoderFromProcess(default: .cpuOnly)
    @Published var layerSelection = ProbeLayerSelection.selectedFromProcess(default: .first8)
    @Published var cacheClearPolicy = ProbeCacheClearPolicy.selectedFromProcess(default: .afterEveryModel)
    @Published var sequenceLength: ProbeSequenceLength {
        didSet {
            guard oldValue != sequenceLength else { return }
            inputIDsText = ProbeRunner.defaultInputIDsText(sequenceLength: sequenceLength)
        }
    }
    @Published var inputIDsText: String
    @Published var generatedTokenCount = ProbeRunner.selectedGeneratedTokenCountFromProcess()
    @Published var isRunning = false
    @Published var steps: [ProbeStep] = []
    @Published var summary = "Idle"
    @Published var currentMemoryText = ProbeMemory.currentText()
    private var didAutoRun = false

    init() {
        let sequenceLength = ProbeRunner.selectedSequenceLengthFromProcess()
        self.sequenceLength = sequenceLength
        self.inputIDsText = ProbeRunner.selectedInputIDsTextFromProcess(sequenceLength: sequenceLength)
    }

    func clear() {
        steps.removeAll()
        summary = "Idle"
        currentMemoryText = ProbeMemory.currentText()
    }

    func runIfRequested(_ shouldRun: Bool) {
        guard shouldRun, !didAutoRun else { return }
        didAutoRun = true
        run()
    }

    func run() {
        guard !isRunning else { return }
        isRunning = true
        steps.removeAll()
        summary = "Running"
        currentMemoryText = ProbeMemory.currentText()

        let computePlan = ProbeComputePlan(endpoint: endpointComputeSelection, decoder: decoderComputeSelection)
        let runMode = runMode
        let layerSelection = layerSelection
        let cacheClearPolicy = cacheClearPolicy
        let sequenceLength = self.sequenceLength
        let inputIDsText = inputIDsText
        let generatedTokenCount = generatedTokenCount
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                ProbeRunner.run(
                    computePlan: computePlan,
                    mode: runMode,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
                    inputIDsText: inputIDsText,
                    generatedTokenCount: generatedTokenCount
                )
            }.value

            switch result {
            case .success(let report):
                steps = report.steps
                summary = report.summary
            case .failure(let error):
                steps = error.steps
                summary = error.message
            }
            currentMemoryText = ProbeMemory.currentText()
            isRunning = false
        }
    }
}
