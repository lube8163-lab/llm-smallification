import SwiftUI
import UIKit

struct DiffusionProbeContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var viewModel = DiffusionProbeViewModel()
    @State private var showingSettings = false
    @State private var showingDiagnostics = false

    var body: some View {
        NavigationStack {
            ChatSurface(viewModel: viewModel)
                .navigationTitle("LLaDA Chat")
                .toolbar {
                    ToolbarItemGroup(placement: .topBarTrailing) {
                        Button {
                            showingDiagnostics = true
                        } label: {
                            Image(systemName: "waveform.path.ecg")
                        }
                        .accessibilityLabel("Diagnostics")

                        Button {
                            showingSettings = true
                        } label: {
                            Image(systemName: "slider.horizontal.3")
                        }
                        .accessibilityLabel("Settings")
                    }
                }
        }
        .sheet(isPresented: $showingSettings) {
            NavigationStack {
                ProbeSettingsView(viewModel: viewModel)
                    .navigationTitle("Settings")
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("Done") {
                                showingSettings = false
                            }
                        }
                    }
            }
        }
        .sheet(isPresented: $showingDiagnostics) {
            NavigationStack {
                DiagnosticsView(viewModel: viewModel)
                    .navigationTitle("Diagnostics")
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("Done") {
                                showingDiagnostics = false
                            }
                        }
                    }
            }
        }
        .task {
            viewModel.autorunIfRequested()
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                UIApplication.shared.isIdleTimerDisabled = true
            } else {
                viewModel.persistNow()
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }
}

private struct ChatSurface: View {
    @ObservedObject var viewModel: DiffusionProbeViewModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    StatusStrip(viewModel: viewModel)
                        .id("status")

                    ForEach(viewModel.chatMessages) { message in
                        ChatBubble(message: message)
                            .id(message.id)
                    }

                    if viewModel.chatMessages.isEmpty {
                        EmptyChatState()
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 96)
            }
            .scrollDismissesKeyboard(.interactively)
            .safeAreaInset(edge: .bottom) {
                VStack(spacing: 0) {
                    ChatInputBar(viewModel: viewModel)

                    if viewModel.demoKeyboardVisible {
                        DemoKeyboardView(highlightedKey: viewModel.demoKeyboardHighlightedKey)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
                .background(.regularMaterial)
            }
            .onChange(of: viewModel.chatMessages) { _, messages in
                guard let last = messages.last else { return }
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }
}

private struct StatusStrip: View {
    @ObservedObject var viewModel: DiffusionProbeViewModel

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(viewModel.isRunning ? Color.orange : Color.green)
                .frame(width: 8, height: 8)

            Text(viewModel.status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            Spacer(minLength: 8)

            Text(viewModel.selectedQuant)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

private struct EmptyChatState: View {
    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 34, weight: .medium))
                .foregroundStyle(.secondary)

            Text("Start a local chat")
                .font(.headline)

            Text("IQ4_XS runs fastest after the first load.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 72)
    }
}

private struct ChatBubble: View {
    let message: DiffusionChatMessage

    var body: some View {
        HStack {
            if message.role == .assistant {
                bubble
                Spacer(minLength: 48)
            } else {
                Spacer(minLength: 48)
                bubble
            }
        }
    }

    private var bubble: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(message.role == .user ? "You" : "LLaDA")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)

            Text(message.content.isEmpty ? "..." : message.content)
                .font(.body)
                .textSelection(.enabled)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(backgroundColor)
        .foregroundStyle(message.role == .user ? .white : .primary)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var backgroundColor: Color {
        switch message.role {
        case .user:
            return .blue
        case .assistant:
            return Color(.secondarySystemGroupedBackground)
        }
    }
}

private struct DemoKeyboardView: View {
    let highlightedKey: String?

    private let rows: [[String]] = [
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["Z", "X", "C", "V", "B", "N", "M", ".", "?"],
        ["SPACE", "RETURN"]
    ]

    var body: some View {
        VStack(spacing: 8) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                HStack(spacing: 6) {
                    ForEach(row, id: \.self) { key in
                        DemoKeyboardKey(label: key,
                                        isHighlighted: highlightedKey == key,
                                        isWide: key == "SPACE" || key == "RETURN")
                    }
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 10)
        .background(Color(.systemGroupedBackground))
        .animation(.easeOut(duration: 0.08), value: highlightedKey)
    }
}

private struct DemoKeyboardKey: View {
    let label: String
    let isHighlighted: Bool
    let isWide: Bool

    var body: some View {
        Text(displayLabel)
            .font(.caption.weight(.semibold))
            .foregroundStyle(isHighlighted ? .white : .primary)
            .frame(minWidth: isWide ? 80 : 34,
                   maxWidth: isWide ? .infinity : 34,
                   minHeight: 34,
                   maxHeight: 34)
            .background(isHighlighted ? Color.blue : Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .shadow(color: Color.black.opacity(isHighlighted ? 0.18 : 0.08),
                    radius: isHighlighted ? 4 : 1,
                    y: isHighlighted ? 2 : 1)
            .scaleEffect(isHighlighted ? 1.05 : 1)
    }

    private var displayLabel: String {
        switch label {
        case "SPACE":
            return "space"
        case "RETURN":
            return "return"
        default:
            return label
        }
    }
}

private struct ChatInputBar: View {
    @ObservedObject var viewModel: DiffusionProbeViewModel
    @FocusState private var inputFocused: Bool

    private var canSend: Bool {
        !viewModel.chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !viewModel.isRunning
            && viewModel.selectedCandidate?.isPresent == true
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Message", text: $viewModel.chatInput, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.plain)
                .focused($inputFocused)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

            Button {
                viewModel.sendChat()
            } label: {
                if viewModel.isRunning {
                    ProgressView()
                        .frame(width: 22, height: 22)
                } else {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 22, height: 22)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canSend)
            .accessibilityLabel("Send")
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .onChange(of: viewModel.demoInputFocused) { _, shouldFocus in
            inputFocused = shouldFocus
        }
        .onAppear {
            inputFocused = viewModel.demoInputFocused
        }
    }
}

private struct ProbeSettingsView: View {
    @ObservedObject var viewModel: DiffusionProbeViewModel

    var body: some View {
        Form {
            Section("Model") {
                Picker("Quant", selection: $viewModel.selectedQuant) {
                    ForEach(viewModel.candidates) { candidate in
                        Text(candidate.quant).tag(candidate.quant)
                    }
                }
                LabeledContent("Path") {
                    Text(viewModel.selectedCandidate?.path ?? "Not found")
                        .font(.caption)
                        .lineLimit(4)
                        .multilineTextAlignment(.trailing)
                }
                Button {
                    viewModel.reloadCandidates()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }

            Section("Chat") {
                Stepper("Seq \(viewModel.seqLen)", value: $viewModel.seqLen, in: 64...256, step: 32)
                Stepper("Steps \(viewModel.steps)", value: $viewModel.steps, in: 4...64, step: 4)
                Stepper("Block \(viewModel.blockLength)", value: $viewModel.blockLength, in: 16...128, step: 16)
                Stepper("Temp \(viewModel.temperature, specifier: "%.1f")", value: $viewModel.temperature, in: 0...2, step: 0.1)
                Stepper("Seed \(viewModel.seed)", value: $viewModel.seed, in: 0...999_999, step: 1)
                Toggle("Adaptive Quality", isOn: $viewModel.adaptiveQualityBoost)
                Button(role: .destructive) {
                    viewModel.clearChat()
                } label: {
                    Label("Clear Chat", systemImage: "trash")
                }
                .disabled(viewModel.isRunning)
            }

            Section("Probe") {
                TextField("Prompt", text: $viewModel.prompt, axis: .vertical)
                    .lineLimit(2...5)
                Stepper("Probe Seq \(viewModel.seqLen)", value: $viewModel.seqLen, in: 32...256, step: 32)
                Button {
                    viewModel.run()
                } label: {
                    Label("Run Probe", systemImage: "play.fill")
                }
                .disabled(viewModel.isRunning)
            }
        }
    }
}

private struct DiagnosticsView: View {
    @ObservedObject var viewModel: DiffusionProbeViewModel

    var body: some View {
        List {
            Section("Status") {
                Text(viewModel.status)
                    .font(.headline)
                if !viewModel.output.isEmpty {
                    Text(viewModel.output)
                        .textSelection(.enabled)
                }
            }

            Section("Log") {
                Text(viewModel.logText.isEmpty ? viewModel.modelSearchSummary() : viewModel.logText)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
            }
        }
    }
}
