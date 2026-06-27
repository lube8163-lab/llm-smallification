import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ProbeViewModel()
    private let autoRun = ProcessInfo.processInfo.arguments.contains("--autorun")
        || ProcessInfo.processInfo.environment["COREML_PROBE_AUTORUN"] == "1"

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Picker("Mode", selection: $viewModel.runMode) {
                        ForEach(ProbeRunMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    Picker("Compute", selection: $viewModel.computeSelection) {
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
            }
            .navigationTitle("Core ML Probe")
        }
        .task {
            viewModel.runIfRequested(autoRun)
        }
    }
}

@MainActor
final class ProbeViewModel: ObservableObject {
    @Published var runMode = ProbeRunMode.selectedFromProcess(default: .loadEmbedding)
    @Published var computeSelection = ProbeComputeSelection.selectedFromProcess(default: .cpuOnly)
    @Published var layerSelection = ProbeLayerSelection.selectedFromProcess(default: .first8)
    @Published var cacheClearPolicy = ProbeCacheClearPolicy.selectedFromProcess(default: .afterEveryModel)
    @Published var inputIDsText = ProbeRunner.selectedInputIDsTextFromProcess()
    @Published var generatedTokenCount = ProbeRunner.selectedGeneratedTokenCountFromProcess()
    @Published var isRunning = false
    @Published var steps: [ProbeStep] = []
    @Published var summary = "Idle"
    @Published var currentMemoryText = ProbeMemory.currentText()
    private var didAutoRun = false

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

        let computeSelection = computeSelection
        let runMode = runMode
        let layerSelection = layerSelection
        let cacheClearPolicy = cacheClearPolicy
        let inputIDsText = inputIDsText
        let generatedTokenCount = generatedTokenCount
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                ProbeRunner.run(
                    computeSelection: computeSelection,
                    mode: runMode,
                    layerSelection: layerSelection,
                    cacheClearPolicy: cacheClearPolicy,
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
