import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ProbeViewModel()
    private let autoRun = ProcessInfo.processInfo.arguments.contains("--autorun")
        || ProcessInfo.processInfo.environment["COREML_PROBE_AUTORUN"] == "1"

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Picker("Compute", selection: $viewModel.computeSelection) {
                        ForEach(ProbeComputeSelection.allCases) { selection in
                            Text(selection.title).tag(selection)
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
    @Published var computeSelection: ProbeComputeSelection = .all
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
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                ProbeRunner.run(computeSelection: computeSelection)
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
