import CoreML
import Darwin
import PhotosUI
import SwiftUI
import UIKit

enum AppTab: Hashable {
    case simple
    case chat
    case probe
}

enum AutomationRunKind: String {
    case probe
    case speedSweep = "speed-sweep"
    case stabilitySweep = "stability-sweep"
    case retentionSweep = "retention-sweep"
    case multimodalSmoke = "multimodal-smoke"
    case imageSmoke = "image-smoke"
    case audioSmoke = "audio-smoke"
    case memoryRamp = "memory-ramp"
    case chat

    var usesChat: Bool {
        self == .chat
    }

    static func parse(_ rawValue: String?) -> AutomationRunKind? {
        guard let rawValue else { return nil }
        let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch value {
        case "", "0", "false", "no", "off", "none":
            return nil
        case "1", "true", "yes", "on", "probe", "run", "single":
            return .probe
        case "speed", "speed-sweep":
            return .speedSweep
        case "stability", "stability-sweep":
            return .stabilitySweep
        case "retention", "retain", "retention-sweep", "retain-sweep":
            return .retentionSweep
        case "multimodal", "image", "image-smoke", "vision-smoke":
            return .imageSmoke
        case "audio", "audio-smoke":
            return .audioSmoke
        case "multimodal-smoke", "synthetic-multimodal":
            return .multimodalSmoke
        case "memory-ramp", "memory", "mem-ramp", "ramp":
            return .memoryRamp
        case "chat", "chat-smoke":
            return .chat
        default:
            return nil
        }
    }
}

struct AutomationConfig {
    let runKind: AutomationRunKind?
    let autoExit: Bool
    let chatPrompt: String

    var shouldRun: Bool {
        runKind != nil
    }

    static func current() -> AutomationConfig {
        let arguments = ProcessInfo.processInfo.arguments
        let environment = ProcessInfo.processInfo.environment
        let runKind = AutomationRunKind.parse(argumentValue(arguments, prefix: "--automation="))
            ?? AutomationRunKind.parse(argumentValue(arguments, prefix: "--autorun="))
            ?? AutomationRunKind.parse(environment["COREML_PROBE_AUTORUN"])
            ?? (arguments.contains("--autorun") ? .probe : nil)
        let autoExit = boolValue(environment["COREML_PROBE_AUTO_EXIT"])
            || arguments.contains("--auto-exit")
            || arguments.contains("--exit-after-run")
        let chatPrompt = argumentValue(arguments, prefix: "--chat-prompt=")
            ?? environment["COREML_PROBE_CHAT_PROMPT"]
            ?? "こんにちは。短く答えてください。"
        return AutomationConfig(runKind: runKind, autoExit: autoExit, chatPrompt: chatPrompt)
    }

    private static func argumentValue(_ arguments: [String], prefix: String) -> String? {
        guard let argument = arguments.first(where: { $0.hasPrefix(prefix) }) else { return nil }
        return String(argument.dropFirst(prefix.count))
    }

    private static func boolValue(_ rawValue: String?) -> Bool {
        guard let rawValue else { return false }
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }
}

enum AutomationExit {
    static func completeIfRequested(_ config: AutomationConfig, success: Bool, summary: String) {
        guard config.autoExit else { return }
        let status = success ? "success" : "failure"
        let code: Int32 = success ? 0 : 1
        print("[CoreMLProbe] automation finished status=\(status) exit_code=\(code) summary=\(summary)")
        Task {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            Darwin.exit(code)
        }
    }
}

struct ContentView: View {
    @StateObject private var probeViewModel: ProbeViewModel
    @StateObject private var chatViewModel: ChatViewModel
    @State private var selectedTab: AppTab
    private let automationConfig: AutomationConfig

    init() {
        let config = AutomationConfig.current()
        automationConfig = config
        _selectedTab = State(initialValue: config.runKind?.usesChat == true ? .chat : (config.shouldRun ? .probe : .simple))
        _probeViewModel = StateObject(wrappedValue: ProbeViewModel())
        _chatViewModel = StateObject(wrappedValue: ChatViewModel())
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            SimpleChatScreen(viewModel: chatViewModel)
                .tabItem {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right.fill")
                }
                .tag(AppTab.simple)

            ChatScreen(viewModel: chatViewModel)
                .tabItem {
                    Label("Debug", systemImage: "wrench.and.screwdriver")
                }
                .tag(AppTab.chat)

            ProbeScreen(viewModel: probeViewModel)
                .tabItem {
                    Label("Probe", systemImage: "waveform.path.ecg.rectangle")
                }
                .tag(AppTab.probe)
        }
        .task {
            chatViewModel.configureControlServerFromEnvironment()
            if automationConfig.runKind?.usesChat == true {
                chatViewModel.runIfRequested(automationConfig)
            } else {
                probeViewModel.runIfRequested(automationConfig)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
            chatViewModel.handleDidEnterBackground()
        }
    }
}

/// Plain consumer-style chat: bubbles with decoded text and images only.
/// Token IDs, logits, compute settings, and step logs all live in the Debug
/// tab; both screens share the same ChatViewModel and generation pipeline.
struct SimpleChatScreen: View {
    @ObservedObject var viewModel: ChatViewModel

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 10) {
                        if viewModel.messages.isEmpty {
                            VStack(spacing: 8) {
                                Image(systemName: "sparkles")
                                    .font(.largeTitle)
                                    .foregroundStyle(.secondary)
                                Text("Gemma 4 12B（オンデバイス）")
                                    .font(.headline)
                                Text("テキストまたは画像付きで話しかけてください")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.top, 80)
                        }

                        ForEach(viewModel.messages) { message in
                            SimpleChatBubble(
                                message: message,
                                showTyping: message.role == .assistant
                                    && message.text.isEmpty
                                    && !message.isError
                                    && viewModel.isGenerating,
                                phase: viewModel.generationPhase
                            )
                            .id(message.id)
                        }

                        if viewModel.isGenerating && !viewModel.generationPhase.isEmpty {
                            // Prefill / encode phase: the empty assistant bubble
                            // already shows the phase + typing dots, so keep the
                            // footer quiet to avoid a duplicate spinner.
                            EmptyView()
                        } else if viewModel.isGenerating {
                            HStack(spacing: 8) {
                                ProgressView()
                                Text("生成中…")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }
                            .padding(.horizontal, 16)
                        } else if !viewModel.lastStatText.isEmpty {
                            Text(viewModel.lastStatText)
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .center)
                                .padding(.top, 2)
                        }
                    }
                    .padding(.vertical, 12)
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    if let last = viewModel.messages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: viewModel.streamTick) { _, _ in
                    // Keep the growing reply pinned to the bottom while streaming.
                    if let last = viewModel.messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
                .scrollDismissesKeyboard(.interactively)
            }
            .navigationTitle("Gemma Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        viewModel.clear()
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                    .disabled(viewModel.isGenerating)
                }
            }
            .safeAreaInset(edge: .bottom) {
                ChatComposer(
                    text: $viewModel.messageText,
                    photoItem: $viewModel.photoItem,
                    attachedImage: $viewModel.attachedImage,
                    isGenerating: viewModel.isGenerating,
                    send: {
                        viewModel.send()
                    }
                )
            }
            .onChange(of: viewModel.photoItem) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        viewModel.attachedImage = image
                    }
                }
            }
        }
    }
}

struct SimpleChatBubble: View {
    let message: ChatMessage
    var showTyping: Bool = false
    var phase: String = ""

    var body: some View {
        HStack {
            if message.role == .user {
                Spacer(minLength: 56)
            }

            VStack(alignment: .leading, spacing: 6) {
                if let image = message.image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: 200, maxHeight: 200)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                if showTyping {
                    HStack(spacing: 8) {
                        TypingDots()
                        if !phase.isEmpty {
                            Text(phase)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                } else {
                    Text(message.isError ? (message.detail ?? message.text) : message.text)
                        .font(.body)
                        .textSelection(.enabled)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .foregroundStyle(message.role == .user ? Color.white : Color.primary)
            .background(
                message.isError
                    ? Color.red.opacity(0.15)
                    : (message.role == .user ? Color.accentColor : Color(.secondarySystemBackground)),
                in: RoundedRectangle(cornerRadius: 16)
            )
            .frame(maxWidth: 320, alignment: message.role == .user ? .trailing : .leading)

            if message.role == .assistant {
                Spacer(minLength: 56)
            }
        }
        .padding(.horizontal, 12)
    }
}

/// Three dots that fade in sequence — a lightweight "thinking" indicator for
/// the gap before the first decoded token streams in.
struct TypingDots: View {
    @State private var phase = 0.0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .frame(width: 7, height: 7)
                    .foregroundStyle(.secondary)
                    .opacity(opacity(for: index))
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: false)) {
                phase = 3
            }
        }
    }

    private func opacity(for index: Int) -> Double {
        let distance = (phase - Double(index)).truncatingRemainder(dividingBy: 3)
        let wrapped = distance < 0 ? distance + 3 : distance
        return 0.3 + 0.7 * max(0, 1 - wrapped)
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

                    VStack(alignment: .leading, spacing: 8) {
                        Text("応答の長さ（自動 = EOSまで、上限\(ProbeRunner.autoGeneratedTokenCap)）")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        Picker("Max Tokens", selection: $viewModel.generatedTokenCount) {
                            ForEach(ChatViewModel.generatedTokenPresets, id: \.self) { count in
                                Text(count == ProbeRunner.autoGeneratedTokenCount ? "自動" : "\(count)").tag(count)
                            }
                        }
                        .pickerStyle(.segmented)
                    }
                    .disabled(viewModel.isGenerating)

                    Stepper(value: $viewModel.retainedDecoderModelCount, in: ProbeRunner.defaultRetainedDecoderModelCount...ProbeRunner.maxRetainedDecoderModelCount) {
                        LabeledContent("Retain Decoders", value: "\(viewModel.retainedDecoderModelCount)")
                    }
                    .disabled(viewModel.isGenerating)

                    Toggle("Image norm [-1,1]", isOn: $viewModel.imageNormSigned)
                        .disabled(viewModel.isGenerating)

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

                Section("Local API") {
                    Toggle(
                        "Enable LAN API",
                        isOn: Binding(
                            get: { viewModel.isAPIEnabled },
                            set: { viewModel.setAPIEnabled($0) }
                        )
                    )

                    if viewModel.isAPIEnabled {
                        LabeledContent("Address", value: viewModel.apiAddress)

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Session token")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(viewModel.apiToken)
                                .font(.caption2.monospaced())
                                .textSelection(.enabled)
                            Button {
                                UIPasteboard.general.string = viewModel.apiToken
                            } label: {
                                Label("Copy Token", systemImage: "doc.on.doc")
                            }
                        }

#if DEBUG
                        Toggle(
                            "Allow diagnostic logs",
                            isOn: Binding(
                                get: { viewModel.apiDiagnosticsEnabled },
                                set: { viewModel.setAPIDiagnosticsEnabled($0) }
                            )
                        )
#endif

                        Text("同じLAN上から操作できます。共有Wi-Fiでは有効にしないでください。バックグラウンドへ移ると自動停止します。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("デフォルトは無効です。必要なセッションだけ明示的に有効化してください。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Status") {
                    LabeledContent("Memory", value: viewModel.currentMemoryText)
                    LabeledContent("Result", value: viewModel.summary)
                    if !viewModel.generatedTokenText.isEmpty {
                        LabeledContent("Tokens", value: viewModel.generatedTokenText)
                    }
                    if let statistics = ProbeRunStatistics(steps: viewModel.steps) {
                        ProbeRunStatisticsRows(statistics: statistics)
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
            .scrollDismissesKeyboard(.interactively)
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
                    photoItem: $viewModel.photoItem,
                    attachedImage: $viewModel.attachedImage,
                    isGenerating: viewModel.isGenerating,
                    send: {
                        viewModel.send()
                    }
                )
            }
            .onChange(of: viewModel.photoItem) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        viewModel.attachedImage = image
                    }
                }
            }
        }
    }
}

struct ChatComposer: View {
    @Binding var text: String
    @Binding var photoItem: PhotosPickerItem?
    @Binding var attachedImage: UIImage?
    let isGenerating: Bool
    let send: () -> Void
    @FocusState private var isFocused: Bool

    private var canSend: Bool {
        !isGenerating && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 8) {
            if let attachedImage {
                HStack(spacing: 8) {
                    Image(uiImage: attachedImage)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    Text("画像を添付済み（32パッチとしてモデルへ入力）")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        self.attachedImage = nil
                        self.photoItem = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 2)
            }

            composerRow
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var composerRow: some View {
        HStack(alignment: .bottom, spacing: 10) {
            PhotosPicker(selection: $photoItem, matching: .images) {
                Image(systemName: attachedImage == nil ? "photo.badge.plus" : "photo.fill")
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
                        isFocused = false
                        send()
                    }
                }
                .toolbar {
                    // A vertical-axis TextField turns Return into a newline, so
                    // without this the software keyboard has no dismiss
                    // affordance. Keep the dismiss button left-aligned so it
                    // does not sit directly above the send button.
                    ToolbarItemGroup(placement: .keyboard) {
                        Button {
                            isFocused = false
                        } label: {
                            Label("閉じる", systemImage: "keyboard.chevron.compact.down")
                        }
                        Spacer()
                    }
                }

            Button {
                isFocused = false
                send()
            } label: {
                Image(systemName: "paperplane.fill")
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canSend)
        }
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
                    if let image = message.image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(maxWidth: 180, maxHeight: 180)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

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
                        .disabled(viewModel.isRunning)

                        Stepper(value: $viewModel.retainedDecoderModelCount, in: ProbeRunner.defaultRetainedDecoderModelCount...ProbeRunner.maxRetainedDecoderModelCount) {
                            LabeledContent("Retain Decoders", value: "\(viewModel.retainedDecoderModelCount)")
                        }
                        .disabled(viewModel.isRunning)
                    }
                    Button {
                        viewModel.run()
                    } label: {
                        Label(viewModel.isRunning ? "Running" : "Run Probe", systemImage: "play.fill")
                    }
                    .disabled(viewModel.isRunning)

                    Button {
                        viewModel.runSpeedSweep()
                    } label: {
                        Label(viewModel.isRunning ? "Running" : "Run Speed Sweep", systemImage: "speedometer")
                    }
                    .disabled(viewModel.isRunning)

                    Button {
                        viewModel.runStabilitySweep()
                    } label: {
                        Label(viewModel.isRunning ? "Running" : "Run Stability Sweep", systemImage: "timer")
                    }
                    .disabled(viewModel.isRunning)

                    Button {
                        viewModel.runRetentionSweep()
                    } label: {
                        Label(viewModel.isRunning ? "Running" : "Run Retain Sweep", systemImage: "rectangle.stack")
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
                    if let statistics = ProbeRunStatistics(steps: viewModel.steps) {
                        ProbeRunStatisticsRows(statistics: statistics)
                    }
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

struct ProbeRunStatistics {
    let generatedTokenCount: Int
    let requestedTokenCount: Int?
    let warmAverageSeconds: Double?
    let totalAverageSeconds: Double?
    let peakMemoryMB: Double?
    let stopReason: String
    let decoderLoadAverageSeconds: Double?
    let decoderPredictAverageSeconds: Double?
    let lmHeadAverageSeconds: Double?

    init?(steps: [ProbeStep]) {
        guard !steps.isEmpty else { return nil }

        let tokenTotals = steps.compactMap { step -> Double? in
            guard step.name.hasPrefix("Token "), step.name.hasSuffix(" total") else {
                return nil
            }
            return step.seconds
        }
        let requested = Self.requestedTokenCount(from: steps)
        let peak = steps.last(where: { $0.name == "Peak memory" })?.memoryMB
            ?? steps.map(\.memoryMB).filter { $0 >= 0 }.max()
        let warmTotals = tokenTotals.dropFirst()
        let stopStep = steps.last(where: { $0.name == "Stop generation" })

        generatedTokenCount = tokenTotals.count
        requestedTokenCount = requested
        warmAverageSeconds = warmTotals.isEmpty ? nil : Self.average(warmTotals)
        totalAverageSeconds = tokenTotals.isEmpty ? nil : Self.average(tokenTotals)
        peakMemoryMB = peak
        stopReason = Self.stopReason(
            generatedTokenCount: tokenTotals.count,
            requestedTokenCount: requested,
            explicitStopReason: stopStep?.detail
        )
        decoderLoadAverageSeconds = Self.averageDuration(
            in: steps,
            matching: { $0.name.hasPrefix("Load gemma4_12b_layers") && $0.name.contains("_decoder_") },
            divisor: tokenTotals.count
        )
        decoderPredictAverageSeconds = Self.averageDuration(
            in: steps,
            matching: { $0.name.hasPrefix("Decoder layer ") },
            divisor: tokenTotals.count
        )
        lmHeadAverageSeconds = Self.averageDuration(
            in: steps,
            matching: { $0.name.hasPrefix("LM head token ") },
            divisor: tokenTotals.count
        )
    }

    var tokenProgressText: String {
        if let requestedTokenCount {
            return "\(generatedTokenCount)/\(requestedTokenCount)"
        }
        return "\(generatedTokenCount)"
    }

    var warmAverageText: String {
        warmAverageSeconds.map { String(format: "%.2fs/token", $0) } ?? "-"
    }

    var totalAverageText: String {
        totalAverageSeconds.map { String(format: "%.2fs/token", $0) } ?? "-"
    }

    var peakMemoryText: String {
        peakMemoryMB.map { String(format: "%.1f MB", $0) } ?? "-"
    }

    var timingBreakdownText: String {
        let decoderLoad = decoderLoadAverageSeconds.map { String(format: "load %.2fs", $0) } ?? "load -"
        let decoderPredict = decoderPredictAverageSeconds.map { String(format: "decode %.2fs", $0) } ?? "decode -"
        let lmHead = lmHeadAverageSeconds.map { String(format: "lm %.2fs", $0) } ?? "lm -"
        return "\(decoderLoad), \(decoderPredict), \(lmHead)"
    }

    private static func requestedTokenCount(from steps: [ProbeStep]) -> Int? {
        guard let generationStep = steps.last(where: { $0.name == "Generation loop" }) else {
            return nil
        }
        let marker = "tokens="
        guard let range = generationStep.detail.range(of: marker) else {
            return nil
        }
        let tail = generationStep.detail[range.upperBound...]
        let digits = tail.prefix { $0.isNumber }
        return Int(String(digits))
    }

    private static func stopReason(
        generatedTokenCount: Int,
        requestedTokenCount: Int?,
        explicitStopReason: String?
    ) -> String {
        if let explicitStopReason, !explicitStopReason.isEmpty {
            return explicitStopReason
        }
        if let requestedTokenCount, generatedTokenCount >= requestedTokenCount {
            return "max tokens reached"
        }
        if generatedTokenCount > 0 {
            return "completed"
        }
        return "-"
    }

    private static func average<S: Sequence>(_ values: S) -> Double where S.Element == Double {
        let values = Array(values)
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func averageDuration(
        in steps: [ProbeStep],
        matching predicate: (ProbeStep) -> Bool,
        divisor: Int
    ) -> Double? {
        guard divisor > 0 else { return nil }
        let sum = steps.reduce(0.0) { partial, step in
            guard predicate(step), let seconds = step.seconds else {
                return partial
            }
            return partial + seconds
        }
        return sum / Double(divisor)
    }
}

struct ProbeRunStatisticsRows: View {
    let statistics: ProbeRunStatistics

    var body: some View {
        LabeledContent("Generated", value: statistics.tokenProgressText)
        LabeledContent("Warm Avg", value: statistics.warmAverageText)
        LabeledContent("All Avg", value: statistics.totalAverageText)
        LabeledContent("Peak", value: statistics.peakMemoryText)
        LabeledContent("Stop", value: statistics.stopReason)
        LabeledContent("Timing", value: statistics.timingBreakdownText)
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
    let id: UUID
    let role: ChatRole
    // Mutable so a streaming assistant reply can grow in place under a stable id.
    var text: String
    var tokens: [Int]
    var detail: String?
    var isError: Bool
    var image: UIImage? = nil

    init(
        id: UUID = UUID(),
        role: ChatRole,
        text: String,
        tokens: [Int],
        detail: String?,
        isError: Bool,
        image: UIImage? = nil
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.tokens = tokens
        self.detail = detail
        self.isError = isError
        self.image = image
    }
}

struct ProbeSweepCase {
    let title: String
    let endpoint: ProbeComputeSelection
    let decoder: ProbeComputeSelection
    let tokens: Int
    let retainedDecoders: Int
}

enum ProbeSweep {
    static let speedCases: [ProbeSweepCase] = [
        ProbeSweepCase(
            title: "Decoder CPU+GPU",
            endpoint: .cpuOnly,
            decoder: .cpuAndGPU,
            tokens: 4,
            retainedDecoders: 0
        ),
        ProbeSweepCase(
            title: "Decoder All",
            endpoint: .cpuOnly,
            decoder: .all,
            tokens: 4,
            retainedDecoders: 0
        ),
        ProbeSweepCase(
            title: "Decoder CPU+ANE",
            endpoint: .cpuOnly,
            decoder: .cpuAndNeuralEngine,
            tokens: 4,
            retainedDecoders: 0
        )
    ]

    static let stabilityCases: [ProbeSweepCase] = [
        ProbeSweepCase(
            title: "Decoder All 8",
            endpoint: .cpuOnly,
            decoder: .all,
            tokens: 8,
            retainedDecoders: 0
        ),
        ProbeSweepCase(
            title: "Decoder All 16",
            endpoint: .cpuOnly,
            decoder: .all,
            tokens: 16,
            retainedDecoders: 0
        )
    ]

    static let retentionCases: [ProbeSweepCase] = [
        ProbeSweepCase(
            title: "Retain 0",
            endpoint: .cpuOnly,
            decoder: .all,
            tokens: 8,
            retainedDecoders: 0
        ),
        ProbeSweepCase(
            title: "Retain 2",
            endpoint: .cpuOnly,
            decoder: .all,
            tokens: 8,
            retainedDecoders: 2
        ),
        ProbeSweepCase(
            title: "Retain 4",
            endpoint: .cpuOnly,
            decoder: .all,
            tokens: 8,
            retainedDecoders: 4
        ),
        // Under the default (no-entitlement) ~3.3 GB ceiling this may hit the
        // memory-margin valve and retain fewer than 6; that is expected and
        // safe. With the increased-memory-limit entitlement it should hold all 6.
        ProbeSweepCase(
            title: "Retain 6",
            endpoint: .cpuOnly,
            decoder: .all,
            tokens: 8,
            retainedDecoders: 6
        )
    ]

    static func summaryLine(for sweepCase: ProbeSweepCase, result: Result<ProbeReport, ProbeFailure>) -> String {
        let steps: [ProbeStep]
        let status: String
        switch result {
        case .success(let report):
            steps = report.steps
            status = "OK"
        case .failure(let failure):
            steps = failure.steps
            status = "FAIL"
        }

        let tokenTotals = steps.compactMap { step -> Double? in
            guard step.name.hasPrefix("Token "), step.name.hasSuffix(" total") else {
                return nil
            }
            return step.seconds
        }
        let warmTotals = tokenTotals.dropFirst()
        let warmAverage = warmTotals.isEmpty ? nil : warmTotals.reduce(0, +) / Double(warmTotals.count)
        let peak = steps.map(\.memoryMB).filter { $0 >= 0 }.max()
        let generated = steps.last(where: { $0.name == "Generated tokens" })?.detail ?? "-"
        let warmText = warmAverage.map { String(format: "%.2fs", $0) } ?? "-"
        let peakText = peak.map { String(format: "%.1f MB", $0) } ?? "-"
        return "\(sweepCase.title): \(status), tokens=\(tokenTotals.count)/\(sweepCase.tokens), warm=\(warmText), peak=\(peakText), \(generated)"
    }
}

enum TokenDisplay {
    private static let knownTokens: [Int: String] = [
        0: "<pad>",
        1: "<eos>",
        2: "<bos>",
        105: "<|turn>",
        106: "<turn|>",
        107: "\\n",
        108: "\\n\\n",
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
        if let decoded = GemmaTokenizerStore.shared.tokenizer?.decode(tokenIDs: tokenIDs),
           !decoded.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return decoded
        }
        return tokenIDs.map(label(for:)).joined(separator: " ")
    }
}

final class GemmaBPETokenizer {
    private let vocab: [String: Int32]
    private let tokenByID: [Int: String]
    private let vocabPiecesByFirstCharacter: [Character: [String]]
    private let padTokenID: Int32

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let rawVocab = model["vocab"] as? [String: Any] else {
            throw ProbeError.invalidInputIDs("invalid tokenizer JSON")
        }

        var parsedVocab: [String: Int32] = [:]
        var parsedTokenByID: [Int: String] = [:]
        var buckets: [Character: [String]] = [:]
        parsedVocab.reserveCapacity(rawVocab.count)
        parsedTokenByID.reserveCapacity(rawVocab.count)
        for (token, value) in rawVocab {
            let id: Int?
            if let intValue = value as? Int {
                id = intValue
            } else if let numberValue = value as? NSNumber {
                id = numberValue.intValue
            } else {
                id = nil
            }
            guard let id, let int32ID = Int32(exactly: id) else {
                continue
            }
            parsedVocab[token] = int32ID
            parsedTokenByID[id] = token
            if let first = token.first,
               !(token.hasPrefix("<") && token.hasSuffix(">")) {
                buckets[first, default: []].append(token)
            }
        }

        vocab = parsedVocab
        tokenByID = parsedTokenByID
        padTokenID = parsedVocab["<pad>"] ?? 0
        vocabPiecesByFirstCharacter = buckets.mapValues { pieces in
            pieces.sorted { lhs, rhs in
                if lhs.count == rhs.count {
                    return lhs < rhs
                }
                return lhs.count > rhs.count
            }
        }
    }

    /// Chat window with a run of placeholder positions reserved for image
    /// hidden states: [pad][BOS user-header][image x N][prompt][turn tail].
    /// Returns the ids plus the window index where the image block starts.
    func encodeChatWindowWithImage(
        prompt: String,
        sequenceLength: ProbeSequenceLength,
        imageTokenCount: Int
    ) throws -> (ids: [Int32], imageStart: Int) {
        let header: [Int32] = [2, 105, 2364, 107]
        let tail: [Int32] = [106, 107, 105, 4368, 107, 100, 45518, 107, 101]
        // Gemma 4 wraps image soft tokens with begin/end-of-image markers and
        // fills the block with the image placeholder id (the merge replaces
        // those positions' embeddings with vision features). The real <eoi>
        // (258882) has a 5.7x-RMS embedding row that overflows the fp16
        // decoder (all logits pin at the +30 softcap), so close the block
        // with a newline instead.
        let beginImage: Int32 = 255999   // <boi>
        let imagePlaceholder: Int32 = 258880  // <image>
        let endImage: Int32 = 107        // \n (avoid <eoi> 258882: fp16 overflow)
        var promptIDs = try encodeText(prompt)
        let budget = sequenceLength.rawValue - header.count - imageTokenCount - 2 - tail.count
        guard budget >= 0 else {
            throw ProbeError.invalidInputIDs(
                "window \(sequenceLength.rawValue) too small for \(imageTokenCount) image tokens plus chat template"
            )
        }
        if promptIDs.count > budget {
            promptIDs = Array(promptIDs.prefix(budget))
        }
        var ids = header
        ids.append(beginImage)
        ids.append(contentsOf: [Int32](repeating: imagePlaceholder, count: imageTokenCount))
        ids.append(endImage)
        ids.append(contentsOf: promptIDs)
        ids.append(contentsOf: tail)
        let leftPad = sequenceLength.rawValue - ids.count
        ids = [Int32](repeating: padTokenID, count: leftPad) + ids
        return (ids, leftPad + header.count + 1)
    }

    /// Chat window with a run of placeholder positions reserved for audio
    /// hidden states, mirroring the image window. The real <eoa> (258883) has
    /// a 3.6x-RMS embedding row with the same fp16-decoder overflow risk as
    /// <eoi>, so the block closes with a newline instead.
    func encodeChatWindowWithAudio(
        prompt: String,
        sequenceLength: ProbeSequenceLength,
        audioTokenCount: Int
    ) throws -> (ids: [Int32], audioStart: Int) {
        let header: [Int32] = [2, 105, 2364, 107]
        let tail: [Int32] = [106, 107, 105, 4368, 107, 100, 45518, 107, 101]
        let beginAudio: Int32 = 256000        // <boa>
        let audioPlaceholder: Int32 = 258881  // <audio>
        let endAudio: Int32 = 107             // \n (avoid <eoa> 258883: fp16 overflow)
        var promptIDs = try encodeText(prompt)
        let budget = sequenceLength.rawValue - header.count - audioTokenCount - 2 - tail.count
        guard budget >= 0 else {
            throw ProbeError.invalidInputIDs(
                "window \(sequenceLength.rawValue) too small for \(audioTokenCount) audio tokens plus chat template"
            )
        }
        if promptIDs.count > budget {
            promptIDs = Array(promptIDs.prefix(budget))
        }
        var ids = header
        ids.append(beginAudio)
        ids.append(contentsOf: [Int32](repeating: audioPlaceholder, count: audioTokenCount))
        ids.append(endAudio)
        ids.append(contentsOf: promptIDs)
        ids.append(contentsOf: tail)
        let leftPad = sequenceLength.rawValue - ids.count
        ids = [Int32](repeating: padTokenID, count: leftPad) + ids
        return (ids, leftPad + header.count + 1)
    }

    func encodeChatWindow(prompt: String, sequenceLength: ProbeSequenceLength) throws -> [Int32] {
        let ids = try encodeChat(prompt: prompt)
        if ids.count < sequenceLength.rawValue {
            return Array(repeating: padTokenID, count: sequenceLength.rawValue - ids.count) + ids
        }
        return Array(ids.suffix(sequenceLength.rawValue))
    }

    func decode(tokenIDs: [Int]) -> String {
        let pieces = tokenIDs.compactMap { tokenByID[$0] }
        var bytes: [UInt8] = []
        var output = ""

        func flushBytes() {
            guard !bytes.isEmpty else { return }
            output += String(decoding: bytes, as: UTF8.self)
            bytes.removeAll()
        }

        for piece in pieces {
            if let byte = byteFallbackValue(piece) {
                bytes.append(byte)
                continue
            }
            flushBytes()
            guard !piece.hasPrefix("<") || !piece.hasSuffix(">") else {
                continue
            }
            output += piece
        }
        flushBytes()
        return output.replacingOccurrences(of: "▁", with: " ")
    }

    private func encodeChat(prompt: String) throws -> [Int32] {
        var ids: [Int32] = [2, 105, 2364, 107]
        ids.append(contentsOf: try encodeText(prompt))
        ids.append(contentsOf: [106, 107, 105, 4368, 107, 100, 45518, 107, 101])
        return ids
    }

    private func encodeText(_ text: String) throws -> [Int32] {
        let normalized = text.replacingOccurrences(of: " ", with: "▁")
        let pieces = greedyPieces(for: normalized)
        return try pieces.map { piece in
            guard let id = vocab[piece] else {
                throw ProbeError.invalidInputIDs("tokenizer piece is missing from vocab: \(piece)")
            }
            return id
        }
    }

    private func greedyPieces(for text: String) -> [String] {
        var pieces: [String] = []
        var index = text.startIndex
        while index < text.endIndex {
            let first = text[index]
            var matchedPiece: String?
            if let candidates = vocabPiecesByFirstCharacter[first] {
                let remaining = text[index...]
                for candidate in candidates where remaining.hasPrefix(candidate) {
                    matchedPiece = candidate
                    break
                }
            }

            if let matchedPiece {
                pieces.append(matchedPiece)
                index = text.index(index, offsetBy: matchedPiece.count)
            } else {
                let scalarText = String(first)
                for byte in scalarText.utf8 {
                    pieces.append(String(format: "<0x%02X>", byte))
                }
                index = text.index(after: index)
            }
        }
        return pieces
    }

    private func byteFallbackValue(_ piece: String) -> UInt8? {
        guard piece.hasPrefix("<0x"), piece.hasSuffix(">") else {
            return nil
        }
        let start = piece.index(piece.startIndex, offsetBy: 3)
        let end = piece.index(before: piece.endIndex)
        return UInt8(piece[start..<end], radix: 16)
    }
}

final class GemmaTokenizerStore {
    static let shared = GemmaTokenizerStore()
    private(set) lazy var tokenizer: GemmaBPETokenizer? = {
        guard let url = Bundle.main.url(forResource: "gemma4-tokenizer", withExtension: "json", subdirectory: "Models") else {
            return nil
        }
        return try? GemmaBPETokenizer(url: url)
    }()
}

@MainActor
final class ChatViewModel: ObservableObject {
    static let generatedTokenPresets = [ProbeRunner.autoGeneratedTokenCount, 64, 128, 256]

    @Published var endpointComputeSelection = ProbeComputeSelection.selectedEndpointFromProcess(
        default: ProbeSequenceLength.usesPal4Stack ? .all : .cpuOnly
    )
    @Published var decoderComputeSelection = ProbeComputeSelection.selectedDecoderFromProcess(default: .all)
    @Published var layerSelection = ProbeLayerSelection.selectedFromProcess(default: .first48)
    @Published var cacheClearPolicy = ProbeCacheClearPolicy.selectedFromProcess(default: .runEndOnly)
    @Published var sequenceLength: ProbeSequenceLength {
        didSet {
            guard oldValue != sequenceLength else { return }
            inputIDsText = ProbeRunner.defaultInputIDsText(sequenceLength: sequenceLength)
        }
    }
    @Published var inputIDsText: String
    @Published var generatedTokenCount = ChatViewModel.selectedGeneratedTokenPresetFromProcess()
    @Published var retainedDecoderModelCount = ProbeRunner.selectedRetainedDecoderModelCountFromProcess()
    @Published var messageText = ""
    @Published var isGenerating = false
    @Published var messages: [ChatMessage] = []
    @Published var steps: [ProbeStep] = []
    @Published var summary = "Idle"
    @Published var generatedTokenText = ""
    @Published var currentMemoryText = ProbeMemory.currentText()
    @Published var photoItem: PhotosPickerItem?
    @Published var attachedImage: UIImage?
    @Published var imageNormSigned = false
    /// Raw waveform frames [1, 32, 640] staged by the HTTP API (audio_b64).
    var attachedAudioFeatures: MLMultiArray?
    @Published private(set) var isAPIEnabled = false
    @Published private(set) var apiAddress = ""
    @Published private(set) var apiToken = ""
    @Published private(set) var apiDiagnosticsEnabled = false
    @Published var lastStatText = ""
    @Published var streamTick = 0
    /// Human-readable phase shown before the first token streams (prefill has
    /// no per-token callback, so this fills the otherwise blank "thinking" gap).
    /// Cleared on the first decoded token, when the reply starts flowing.
    @Published var generationPhase = ""
    private var didAutoRun = false
    private var controlServer: HTTPControlServer?
    private var streamingMessageID: UUID?
    private var streamingTokens: [Int] = []

    /// Appends an empty assistant bubble that a streamed reply grows into.
    private func beginStreamingAssistant() {
        streamingTokens = []
        let message = ChatMessage(role: .assistant, text: "", tokens: [], detail: nil, isError: false)
        streamingMessageID = message.id
        messages.append(message)
    }

    /// Grows the in-progress assistant bubble by one token so text appears
    /// live. Invoked via `Task { @MainActor }` from the background generation
    /// closure (see `onToken` in send), so it is already main-actor isolated.
    func streamToken(_ tokenID: Int) {
        guard let id = streamingMessageID,
              let index = messages.firstIndex(where: { $0.id == id }) else { return }
        // First decoded token = prefill done, reply is now streaming.
        if streamingTokens.isEmpty { generationPhase = "" }
        streamingTokens.append(tokenID)
        messages[index].text = TokenDisplay.joinedLabels(for: streamingTokens)
        messages[index].tokens = streamingTokens
        streamTick &+= 1
    }

    private func finalizeStreaming(text: String, tokens: [Int], detail: String?, isError: Bool) {
        if let id = streamingMessageID, let index = messages.firstIndex(where: { $0.id == id }) {
            messages[index].text = text
            messages[index].tokens = tokens
            messages[index].detail = detail
            messages[index].isError = isError
        } else {
            messages.append(ChatMessage(role: .assistant, text: text, tokens: tokens, detail: detail, isError: isError))
        }
        streamingMessageID = nil
        streamTick &+= 1
    }

    func configureControlServerFromEnvironment() {
        let environment = ProcessInfo.processInfo.environment
        guard Self.boolValue(environment["COREML_PROBE_ENABLE_API"]) else { return }
        let configuredToken = environment["COREML_PROBE_API_TOKEN"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let token = configuredToken.flatMap { $0.count >= 16 ? $0 : nil } ?? Self.makeSessionToken()
        let diagnostics = Self.diagnosticsAvailable
            && Self.boolValue(environment["COREML_PROBE_API_DIAGNOSTICS"])
        startControlServer(token: token, diagnosticsEnabled: diagnostics)
    }

    func setAPIEnabled(_ enabled: Bool) {
        if enabled {
            startControlServer(token: Self.makeSessionToken(), diagnosticsEnabled: false)
        } else {
            stopControlServer()
        }
    }

    func setAPIDiagnosticsEnabled(_ enabled: Bool) {
        let effectiveValue = Self.diagnosticsAvailable && enabled && isAPIEnabled
        apiDiagnosticsEnabled = effectiveValue
        controlServer?.setDiagnosticsEnabled(effectiveValue)
    }

    func handleDidEnterBackground() {
        stopControlServer()
        ProbeRunner.releaseResidentChatModels()
    }

    private func startControlServer(token: String, diagnosticsEnabled: Bool) {
        guard controlServer == nil else { return }
        let server = HTTPControlServer(
            chatViewModel: self,
            authToken: token,
            diagnosticsEnabled: diagnosticsEnabled
        )
        server.start()
        guard server.lastError == nil else {
            summary = "API start failed: \(server.lastError ?? "unknown error")"
            return
        }
        controlServer = server
        isAPIEnabled = true
        apiAddress = server.displayAddress
        apiToken = token
        apiDiagnosticsEnabled = Self.diagnosticsAvailable && diagnosticsEnabled
    }

    private func stopControlServer() {
        let previousToken = apiToken
        controlServer?.stop()
        controlServer = nil
        isAPIEnabled = false
        apiAddress = ""
        apiToken = ""
        apiDiagnosticsEnabled = false
        if !previousToken.isEmpty, UIPasteboard.general.string == previousToken {
            UIPasteboard.general.string = ""
        }
    }

    private static var diagnosticsAvailable: Bool {
#if DEBUG
        true
#else
        false
#endif
    }

    private static func makeSessionToken() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "")
            + UUID().uuidString.replacingOccurrences(of: "-", with: "")
    }

    private static func boolValue(_ rawValue: String?) -> Bool {
        guard let rawValue else { return false }
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    var recentSteps: [ProbeStep] {
        Array(steps.suffix(16))
    }

    init() {
        let sequenceLength = ProbeRunner.selectedSequenceLengthFromProcess()
        self.sequenceLength = sequenceLength
        self.inputIDsText = ProbeRunner.selectedInputIDsTextFromProcess(sequenceLength: sequenceLength)
    }

    private static func selectedGeneratedTokenPresetFromProcess() -> Int {
        let selected = ProbeRunner.selectedGeneratedTokenCountFromProcess(default: ProbeRunner.autoGeneratedTokenCount)
        guard generatedTokenPresets.contains(selected) else {
            return ProbeRunner.autoGeneratedTokenCount
        }
        return selected
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

    func runIfRequested(_ config: AutomationConfig) {
        guard config.runKind == .chat, !didAutoRun else { return }
        didAutoRun = true
        messageText = config.chatPrompt
        print("[CoreMLProbe] automation requested kind=\(config.runKind?.rawValue ?? "-") auto_exit=\(config.autoExit)")
        send { success, summary in
            AutomationExit.completeIfRequested(config, success: success, summary: summary)
        }
    }

    func send(completion: ((Bool, String) -> Void)? = nil) {
        let text = messageText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isGenerating else {
            completion?(false, "generation already running")
            return
        }
        guard !text.isEmpty else {
            completion?(false, "empty message")
            return
        }
        if let image = attachedImage {
            sendWithImage(text: text, image: image, completion: completion)
            return
        }
        if let features = attachedAudioFeatures {
            sendWithAudio(text: text, features: features, completion: completion)
            return
        }

        // With the KV assets bundled, route plain text chat through the Seq320
        // window too: one 320-token prefill then seq-1 decode steps, which
        // reads longer replies than the fixed sliding window and shares the
        // image path's fast decode.
        let effectiveSequenceLength: ProbeSequenceLength =
            ProbeRunner.kvChatAvailable ? .seq320 : self.sequenceLength
        let resolvedWindow: TokenWindowResolution
        do {
            resolvedWindow = try Self.resolveTokenWindow(
                messageText: text,
                fallbackText: inputIDsText,
                sequenceLength: effectiveSequenceLength
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
            completion?(false, error.localizedDescription)
            return
        }

        let inputIDsText = resolvedWindow.text
        let generatedTokenCount = generatedTokenCount
        let retainedDecoderModelCount = retainedDecoderModelCount
        let computePlan = ProbeComputePlan(endpoint: endpointComputeSelection, decoder: decoderComputeSelection)
        let layerSelection = layerSelection
        let cacheClearPolicy = cacheClearPolicy
        let sequenceLength = effectiveSequenceLength

        if resolvedWindow.shouldUpdateInputField {
            self.inputIDsText = inputIDsText
        }
        messageText = ""
        isGenerating = true
        summary = "Generating"
        generationPhase = "考え中…"
        generatedTokenText = ""
        currentMemoryText = ProbeMemory.currentText()
        messages.append(ChatMessage(
            role: .user,
            text: text,
            tokens: [],
            detail: resolvedWindow.detail,
            isError: false
        ))
        ChatSessionLog.append(role: "user", text: text, tokens: [], seconds: nil, detail: nil)
        beginStreamingAssistant()

        Task {
            let onToken: (Int) -> Void = { [weak self] id in
                Task { @MainActor in self?.streamToken(id) }
            }
            let result = await Task.detached(priority: .userInitiated) {
                ProbeRunner.run(
                    computePlan: computePlan,
                    mode: .generateTokenLoop,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
                    inputIDsText: inputIDsText,
                    generatedTokenCount: generatedTokenCount,
                    retainedDecoderModelCount: retainedDecoderModelCount,
                    persistent: true,
                    onToken: onToken
                )
            }.value

            let success: Bool
            let completionSummary: String
            switch result {
            case .success(let report):
                let tokens = Self.generatedTokenIDs(from: report)
                let displayText = TokenDisplay.joinedLabels(for: tokens)
                let finalText = displayText.isEmpty ? report.summary : displayText
                steps = report.steps
                summary = report.summary
                generatedTokenText = tokens.map { "#\($0)" }.joined(separator: ", ")
                finalizeStreaming(
                    text: finalText,
                    tokens: tokens,
                    detail: Self.generatedTokenDetail(from: report),
                    isError: false
                )
                ChatSessionLog.append(
                    role: "assistant",
                    text: finalText,
                    tokens: tokens,
                    seconds: Self.generationSeconds(from: report),
                    detail: Self.generatedTokenDetail(from: report)
                )
                lastStatText = Self.statText(tokens: tokens, report: report)
                if let nextWindow = Self.slidTokenWindow(
                    from: inputIDsText,
                    appending: tokens,
                    sequenceLength: sequenceLength
                ) {
                    self.inputIDsText = nextWindow
                }
                success = true
                completionSummary = report.summary
            case .failure(let error):
                steps = error.steps
                summary = error.message
                generatedTokenText = ""
                finalizeStreaming(text: "Generation failed", tokens: [], detail: error.message, isError: true)
                success = false
                completionSummary = error.message
            }

            currentMemoryText = ProbeMemory.currentText()
            isGenerating = false
        generationPhase = ""
            completion?(success, completionSummary)
        }
    }

    /// Image-attached turn: runs the real Gemma4 image embedder over 32
    /// pixel patches and overlays the resulting hidden states into the
    /// decoder window. The decoder stack itself is unchanged (same resident
    /// pal4/ANE models), so per-token speed matches text-only chat.
    private func sendWithImage(text: String, image: UIImage, completion: ((Bool, String) -> Void)?) {
        guard let tokenizer = GemmaTokenizerStore.shared.tokenizer else {
            completion?(false, "tokenizer unavailable")
            return
        }
        // Prefer the full-fidelity path (Seq320 window, 256 image tokens =
        // the real processor budget) whenever its assets are bundled; fall
        // back to the Seq64/32-patch micro smoke otherwise.
        let useSeq320 = ProbeRunner.imageChatSeq320Available
        let imageSequenceLength: ProbeSequenceLength = useSeq320 ? .seq320 : sequenceLength
        let imagePatchCount = useSeq320 ? 256 : ProbeRunner.imagePatchCount

        let window: (ids: [Int32], imageStart: Int)
        let pixelValues: MLMultiArray
        do {
            window = try tokenizer.encodeChatWindowWithImage(
                prompt: text,
                sequenceLength: imageSequenceLength,
                imageTokenCount: imagePatchCount
            )
            pixelValues = try Self.makeImagePixelValues(from: image, signed: imageNormSigned, patchCount: imagePatchCount)
        } catch {
            summary = error.localizedDescription
            completion?(false, error.localizedDescription)
            return
        }

        let inputIDsText = window.ids.map(String.init).joined(separator: ",")
        let imageStart = window.imageStart
        let generatedTokenCount = generatedTokenCount
        let retainedDecoderModelCount = retainedDecoderModelCount
        let computePlan = ProbeComputePlan(endpoint: endpointComputeSelection, decoder: decoderComputeSelection)
        let layerSelection = layerSelection
        let cacheClearPolicy = cacheClearPolicy
        let sequenceLength = imageSequenceLength

        messageText = ""
        attachedImage = nil
        photoItem = nil
        isGenerating = true
        summary = "Generating (image)"
        generationPhase = "画像を見ています…"
        generatedTokenText = ""
        currentMemoryText = ProbeMemory.currentText()
        messages.append(ChatMessage(
            role: .user,
            text: text,
            tokens: [],
            detail: "image: \(imagePatchCount) patches, seq \(imageSequenceLength.rawValue), window start \(imageStart)",
            isError: false,
            image: image
        ))
        ChatSessionLog.append(role: "user", text: text + " [image]", tokens: [], seconds: nil, detail: nil)
        beginStreamingAssistant()

        Task {
            let onToken: (Int) -> Void = { [weak self] id in
                Task { @MainActor in self?.streamToken(id) }
            }
            let result = await Task.detached(priority: .userInitiated) { () -> Result<ProbeReport, ProbeFailure> in
                do {
                    let imageHidden = try ProbeRunner.encodeImage(pixelValues: pixelValues)
                    return ProbeRunner.run(
                        computePlan: computePlan,
                        mode: .generateTokenLoop,
                        layerSelection: layerSelection,
                        cacheClearPolicy: cacheClearPolicy,
                        sequenceLength: sequenceLength,
                        inputIDsText: inputIDsText,
                        generatedTokenCount: generatedTokenCount,
                        retainedDecoderModelCount: retainedDecoderModelCount,
                        imageHidden: imageHidden,
                        imageStartPosition: imageStart,
                        imageBlockBidirectional: true,
                        persistent: true,
                        onToken: onToken
                    )
                } catch {
                    return .failure(ProbeFailure(steps: [], message: String(describing: error)))
                }
            }.value

            let success: Bool
            let completionSummary: String
            switch result {
            case .success(let report):
                let tokens = Self.generatedTokenIDs(from: report)
                let displayText = TokenDisplay.joinedLabels(for: tokens)
                let finalText = displayText.isEmpty ? report.summary : displayText
                steps = report.steps
                summary = report.summary
                generatedTokenText = tokens.map { "#\($0)" }.joined(separator: ", ")
                finalizeStreaming(
                    text: finalText,
                    tokens: tokens,
                    detail: Self.generatedTokenDetail(from: report),
                    isError: false
                )
                ChatSessionLog.append(
                    role: "assistant",
                    text: finalText,
                    tokens: tokens,
                    seconds: Self.generationSeconds(from: report),
                    detail: Self.generatedTokenDetail(from: report)
                )
                lastStatText = Self.statText(tokens: tokens, report: report)
                success = true
                completionSummary = report.summary
            case .failure(let error):
                steps = error.steps
                summary = error.message
                finalizeStreaming(text: "Generation failed", tokens: [], detail: error.message, isError: true)
                success = false
                completionSummary = error.message
            }

            currentMemoryText = ProbeMemory.currentText()
            isGenerating = false
        generationPhase = ""
            completion?(success, completionSummary)
        }
    }

    /// Audio counterpart of sendWithImage: caller supplies raw waveform
    /// frames [1, 32, 640] (16 kHz PCM in [-1, 1]); the audio embedder output
    /// overlays the <audio> placeholder block via the shared hidden-overlay
    /// path in the token loop.
    private func sendWithAudio(text: String, features: MLMultiArray, completion: ((Bool, String) -> Void)?) {
        guard let tokenizer = GemmaTokenizerStore.shared.tokenizer else {
            completion?(false, "tokenizer unavailable")
            return
        }
        let window: (ids: [Int32], audioStart: Int)
        do {
            window = try tokenizer.encodeChatWindowWithAudio(
                prompt: text,
                sequenceLength: sequenceLength,
                audioTokenCount: ProbeRunner.audioTokenCount
            )
        } catch {
            summary = error.localizedDescription
            completion?(false, error.localizedDescription)
            return
        }

        let inputIDsText = window.ids.map(String.init).joined(separator: ",")
        let audioStart = window.audioStart
        let generatedTokenCount = generatedTokenCount
        let retainedDecoderModelCount = retainedDecoderModelCount
        let computePlan = ProbeComputePlan(endpoint: endpointComputeSelection, decoder: decoderComputeSelection)
        let layerSelection = layerSelection
        let cacheClearPolicy = cacheClearPolicy
        let sequenceLength = self.sequenceLength

        messageText = ""
        attachedAudioFeatures = nil
        isGenerating = true
        summary = "Generating (audio)"
        generationPhase = "音声を聞いています…"
        generatedTokenText = ""
        currentMemoryText = ProbeMemory.currentText()
        messages.append(ChatMessage(
            role: .user,
            text: text,
            tokens: [],
            detail: "audio: \(ProbeRunner.audioTokenCount) tokens (40ms each), window start \(audioStart)",
            isError: false
        ))
        ChatSessionLog.append(role: "user", text: text + " [audio]", tokens: [], seconds: nil, detail: nil)
        beginStreamingAssistant()

        Task {
            let onToken: (Int) -> Void = { [weak self] id in
                Task { @MainActor in self?.streamToken(id) }
            }
            let result = await Task.detached(priority: .userInitiated) { () -> Result<ProbeReport, ProbeFailure> in
                do {
                    let audioHidden = try ProbeRunner.encodeAudio(inputFeatures: features)
                    return ProbeRunner.run(
                        computePlan: computePlan,
                        mode: .generateTokenLoop,
                        layerSelection: layerSelection,
                        cacheClearPolicy: cacheClearPolicy,
                        sequenceLength: sequenceLength,
                        inputIDsText: inputIDsText,
                        generatedTokenCount: generatedTokenCount,
                        retainedDecoderModelCount: retainedDecoderModelCount,
                        imageHidden: audioHidden,
                        imageStartPosition: audioStart,
                        persistent: true,
                        onToken: onToken
                    )
                } catch {
                    return .failure(ProbeFailure(steps: [], message: String(describing: error)))
                }
            }.value

            let success: Bool
            let completionSummary: String
            switch result {
            case .success(let report):
                let tokens = Self.generatedTokenIDs(from: report)
                let displayText = TokenDisplay.joinedLabels(for: tokens)
                let finalText = displayText.isEmpty ? report.summary : displayText
                steps = report.steps
                summary = report.summary
                generatedTokenText = tokens.map { "#\($0)" }.joined(separator: ", ")
                finalizeStreaming(
                    text: finalText,
                    tokens: tokens,
                    detail: Self.generatedTokenDetail(from: report),
                    isError: false
                )
                ChatSessionLog.append(
                    role: "assistant",
                    text: finalText,
                    tokens: tokens,
                    seconds: Self.generationSeconds(from: report),
                    detail: Self.generatedTokenDetail(from: report)
                )
                lastStatText = Self.statText(tokens: tokens, report: report)
                success = true
                completionSummary = report.summary
            case .failure(let error):
                steps = error.steps
                summary = error.message
                finalizeStreaming(text: "Generation failed", tokens: [], detail: error.message, isError: true)
                success = false
                completionSummary = error.message
            }

            currentMemoryText = ProbeMemory.currentText()
            isGenerating = false
        generationPhase = ""
            completion?(success, completionSummary)
        }
    }

    /// Human-readable "N tok · Ts · X tok/s" for the chat footer.
    private static func statText(tokens: [Int], report: ProbeReport) -> String {
        let totals = report.steps
            .filter { $0.name.hasPrefix("Token ") && $0.name.hasSuffix(" total") }
            .compactMap(\.seconds)
        guard !totals.isEmpty, !tokens.isEmpty else { return "" }
        // Warm rate = tokens after the first (the first can carry a resident
        // model (re)load on a cold turn).
        let warm = totals.count >= 2 ? Array(totals.dropFirst()) : totals
        let warmSeconds = warm.reduce(0, +)
        guard warmSeconds > 0 else { return "" }
        let tps = Double(warm.count) / warmSeconds
        return String(format: "%d tok · %.2f tok/s", tokens.count, tps)
    }

    /// Sum of per-token totals, for tok/s reporting in the session log.
    private static func generationSeconds(from report: ProbeReport) -> Double? {
        let seconds = report.steps
            .filter { $0.name.hasPrefix("Token ") && $0.name.hasSuffix(" total") }
            .compactMap(\.seconds)
        return seconds.isEmpty ? nil : seconds.reduce(0, +)
    }

    /// Downscales the image to a square grid of 48x48 patches (aspect-fill)
    /// and packs them row-major into the embedder contract [1, N, 6912] fp32.
    /// patchCount 256 = the real processor budget (16x16 grid, 768x768 px;
    /// merged-patch layout is plain row-major 48x48x3, verified against
    /// patches_merge). patchCount 32 = the legacy micro smoke (6x6 grid,
    /// first 32 patches). The real processor only rescales 1/255 to [0, 1],
    /// so signed=false matches training; signed=true stays as an A/B knob.
    private static func makeImagePixelValues(from image: UIImage, signed: Bool, patchCount: Int = ProbeRunner.imagePatchCount) throws -> MLMultiArray {
        let patchSide = 48
        let gridColumns = patchCount == 256 ? 16 : 6
        let side = gridColumns * patchSide
        let patchDim = ProbeRunner.imagePatchDim

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: side, height: side), format: format)
        let squareImage = renderer.image { _ in
            let imageSize = image.size
            let scale = max(CGFloat(side) / imageSize.width, CGFloat(side) / imageSize.height)
            let drawSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
            let origin = CGPoint(
                x: (CGFloat(side) - drawSize.width) / 2,
                y: (CGFloat(side) - drawSize.height) / 2
            )
            image.draw(in: CGRect(origin: origin, size: drawSize))
        }

        guard let cgImage = squareImage.cgImage else {
            throw ProbeError.unexpectedShape("could not rasterize attached image")
        }
        var rgba = [UInt8](repeating: 0, count: side * side * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba,
            width: side,
            height: side,
            bitsPerComponent: 8,
            bytesPerRow: side * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ProbeError.unexpectedShape("could not create bitmap context")
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: side, height: side))

        let array = try MLMultiArray(
            shape: [1, NSNumber(value: patchCount), NSNumber(value: patchDim)],
            dataType: .float32
        )
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        for patch in 0..<patchCount {
            let patchX = patch % gridColumns
            let patchY = patch / gridColumns
            let base = patch * patchDim
            for y in 0..<patchSide {
                for x in 0..<patchSide {
                    let pixelX = patchX * patchSide + x
                    let pixelY = patchY * patchSide + y
                    let sourceIndex = (pixelY * side + pixelX) * 4
                    let destinationIndex = base + (y * patchSide + x) * 3
                    if signed {
                        pointer[destinationIndex] = Float(rgba[sourceIndex]) / 127.5 - 1.0
                        pointer[destinationIndex + 1] = Float(rgba[sourceIndex + 1]) / 127.5 - 1.0
                        pointer[destinationIndex + 2] = Float(rgba[sourceIndex + 2]) / 127.5 - 1.0
                    } else {
                        pointer[destinationIndex] = Float(rgba[sourceIndex]) / 255.0
                        pointer[destinationIndex + 1] = Float(rgba[sourceIndex + 1]) / 255.0
                        pointer[destinationIndex + 2] = Float(rgba[sourceIndex + 2]) / 255.0
                    }
                }
            }
        }
        return array
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

        if let tokenizer = GemmaTokenizerStore.shared.tokenizer {
            let ids = try tokenizer.encodeChatWindow(prompt: messageText, sequenceLength: sequenceLength)
            let text = formatTokenWindow(ids)
            return TokenWindowResolution(
                text: text,
                detail: tokenWindowDetail(text: text, source: "from bundled tokenizer", ids: ids, sequenceLength: sequenceLength),
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
        let leftPadCount = ids.prefix { $0 == 0 }.count
        if leftPadCount > 0 {
            parts.append("left-pad=\(leftPadCount)")
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
    @Published var retainedDecoderModelCount = ProbeRunner.selectedRetainedDecoderModelCountFromProcess()
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

    func runIfRequested(_ config: AutomationConfig) {
        guard let runKind = config.runKind, !runKind.usesChat, !didAutoRun else { return }
        didAutoRun = true
        print("[CoreMLProbe] automation requested kind=\(runKind.rawValue) auto_exit=\(config.autoExit)")
        let completion: (Bool, String) -> Void = { success, summary in
            AutomationExit.completeIfRequested(config, success: success, summary: summary)
        }
        switch runKind {
        case .probe:
            run(completion: completion)
        case .speedSweep:
            runSpeedSweep(completion: completion)
        case .stabilitySweep:
            runStabilitySweep(completion: completion)
        case .retentionSweep:
            runRetentionSweep(completion: completion)
        case .multimodalSmoke:
            runMode = .multimodalSmoke
            run(completion: completion)
        case .imageSmoke:
            runMode = .imageSmoke
            run(completion: completion)
        case .audioSmoke:
            runMode = .audioSmoke
            run(completion: completion)
        case .memoryRamp:
            runMode = .memoryRamp
            run(completion: completion)
        case .chat:
            break
        }
    }

    func run(completion: ((Bool, String) -> Void)? = nil) {
        guard !isRunning else {
            completion?(false, "probe already running")
            return
        }
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
        let retainedDecoderModelCount = retainedDecoderModelCount
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                ProbeRunner.run(
                    computePlan: computePlan,
                    mode: runMode,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
                    sequenceLength: sequenceLength,
                    inputIDsText: inputIDsText,
                    generatedTokenCount: generatedTokenCount,
                    retainedDecoderModelCount: retainedDecoderModelCount
                )
            }.value

            let success: Bool
            let completionSummary: String
            switch result {
            case .success(let report):
                steps = report.steps
                summary = report.summary
                success = true
                completionSummary = report.summary
            case .failure(let error):
                steps = error.steps
                summary = error.message
                success = false
                completionSummary = error.message
            }
            currentMemoryText = ProbeMemory.currentText()
            isRunning = false
            completion?(success, completionSummary)
        }
    }

    func runSpeedSweep(completion: ((Bool, String) -> Void)? = nil) {
        runSweep(title: "speed sweep", cases: ProbeSweep.speedCases, completion: completion)
    }

    func runStabilitySweep(completion: ((Bool, String) -> Void)? = nil) {
        runSweep(title: "stability sweep", cases: ProbeSweep.stabilityCases, completion: completion)
    }

    func runRetentionSweep(completion: ((Bool, String) -> Void)? = nil) {
        runSweep(title: "retention sweep", cases: ProbeSweep.retentionCases, completion: completion)
    }

    private func runSweep(
        title: String,
        cases sweepCases: [ProbeSweepCase],
        completion: ((Bool, String) -> Void)? = nil
    ) {
        guard !isRunning else {
            completion?(false, "probe already running")
            return
        }
        isRunning = true
        steps.removeAll()
        summary = "Running \(title)"
        currentMemoryText = ProbeMemory.currentText()

        let layerSelection: ProbeLayerSelection = .first48
        let cacheClearPolicy: ProbeCacheClearPolicy = .runEndOnly
        let sequenceLength = self.sequenceLength
        let inputIDsText = inputIDsText

        Task {
            let sweepResult = await Task.detached(priority: .userInitiated) {
                var combinedSteps: [ProbeStep] = []
                var summaryLines: [String] = []
                var allSucceeded = true
                print("[CoreMLProbe] sweep started kind=\(title) cases=\(sweepCases.count) mode=generate-token-loop seq=\(sequenceLength.rawValue) layers=\(layerSelection.rawValue) cache=\(cacheClearPolicy.rawValue)")

                for (index, sweepCase) in sweepCases.enumerated() {
                    let caseNumber = index + 1
                    let caseDetail = "case=\(caseNumber)/\(sweepCases.count) endpoint=\(sweepCase.endpoint.title) decoder=\(sweepCase.decoder.title) tokens=\(sweepCase.tokens) retain=\(sweepCase.retainedDecoders)"
                    print("[CoreMLProbe] sweep case started \(caseDetail)")
                    combinedSteps.append(ProbeStep(
                        name: "Sweep case \(caseNumber)",
                        seconds: nil,
                        memoryMB: ProbeMemory.currentMB(),
                        detail: caseDetail
                    ))

                    let result = ProbeRunner.run(
                        computePlan: ProbeComputePlan(endpoint: sweepCase.endpoint, decoder: sweepCase.decoder),
                        mode: .generateTokenLoop,
                        layerSelection: layerSelection,
                        cacheClearPolicy: cacheClearPolicy,
                        sequenceLength: sequenceLength,
                        inputIDsText: inputIDsText,
                        generatedTokenCount: sweepCase.tokens,
                        retainedDecoderModelCount: sweepCase.retainedDecoders
                    )
                    summaryLines.append(ProbeSweep.summaryLine(for: sweepCase, result: result))

                    switch result {
                    case .success(let report):
                        combinedSteps.append(contentsOf: report.steps)
                    case .failure(let failure):
                        allSucceeded = false
                        combinedSteps.append(contentsOf: failure.steps)
                    }
                    print("[CoreMLProbe] sweep case finished \(caseDetail) result=\(summaryLines.last ?? "-")")
                }

                combinedSteps.append(ProbeStep(
                    name: "Sweep summary",
                    seconds: nil,
                    memoryMB: ProbeMemory.currentMB(),
                    detail: summaryLines.joined(separator: " | ")
                ))
                print("[CoreMLProbe] sweep finished kind=\(title) \(summaryLines.joined(separator: " | "))")
                return (steps: combinedSteps, summary: summaryLines.joined(separator: " | "), success: allSucceeded)
            }.value

            steps = sweepResult.steps
            summary = sweepResult.summary
            currentMemoryText = ProbeMemory.currentText()
            isRunning = false
            completion?(sweepResult.success, sweepResult.summary)
        }
    }
}
