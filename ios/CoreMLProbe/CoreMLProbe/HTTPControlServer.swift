import Foundation
import Network
import UIKit

/// Minimal LAN HTTP server so the chat pipeline can be driven and observed
/// from outside the device (curl from a Mac, a coding agent, CI). Runs inside
/// the app on `COREML_PROBE_API_PORT` (default 8765).
///
/// Endpoints:
///   GET  /status            → JSON config/readiness snapshot
///   GET  /log?lines=N       → tail of Documents/probe-steps.log (plain text)
///   GET  /chatlog           → Documents/chat-log.jsonl (plain text)
///   POST /generate          → {"prompt": "...", "tokens": 8} runs one chat
///                             turn and returns the assistant reply + timings
final class HTTPControlServer {
    private let port: UInt16
    private var listener: NWListener?
    private weak var chatViewModel: ChatViewModel?
    private let queue = DispatchQueue(label: "http-control-server")

    private(set) var lastError: String?

    init(chatViewModel: ChatViewModel) {
        self.chatViewModel = chatViewModel
        let environment = ProcessInfo.processInfo.environment
        self.port = environment["COREML_PROBE_API_PORT"].flatMap { UInt16($0) } ?? 8765
    }

    var displayAddress: String {
        "\(Self.wifiIPv4Address() ?? "<wifi-ip>"):\(port)"
    }

    func start() {
        do {
            let listener = try NWListener(using: .tcp, on: NWEndpoint.Port(rawValue: port)!)
            listener.newConnectionHandler = { [weak self] connection in
                self?.handle(connection: connection)
            }
            listener.start(queue: queue)
            self.listener = listener
            print("[CoreMLProbe] HTTP control server listening on port \(port)")
            triggerLocalNetworkPermission()
        } catch {
            lastError = String(describing: error)
            print("[CoreMLProbe] HTTP control server failed to start: \(error)")
        }
    }

    /// iOS gates LAN traffic behind the Local Network privacy permission, and
    /// the system prompt only fires on OUTGOING local traffic — a bare
    /// listener accepts TCP connections but incoming payloads never reach the
    /// app until permission is granted. Dial our own LAN address once at
    /// startup so the prompt appears; the connection itself is throwaway.
    private func triggerLocalNetworkPermission() {
        guard let address = Self.wifiIPv4Address() else { return }
        let connection = NWConnection(
            host: NWEndpoint.Host(address),
            port: NWEndpoint.Port(rawValue: port)!,
            using: .tcp
        )
        connection.start(queue: queue)
        queue.asyncAfter(deadline: .now() + 3) {
            connection.cancel()
        }
    }

    // MARK: - Connection handling

    private func handle(connection: NWConnection) {
        connection.start(queue: queue)
        receiveRequest(connection: connection, buffer: Data())
    }

    private func receiveRequest(connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if error != nil {
                connection.cancel()
                return
            }
            if let request = HTTPRequest(raw: buffer) {
                self.route(request: request, connection: connection)
            } else if isComplete {
                connection.cancel()
            } else if buffer.count > (4 << 20) {
                self.send(connection: connection, status: "413 Payload Too Large", body: Data(), contentType: "text/plain")
            } else {
                self.receiveRequest(connection: connection, buffer: buffer)
            }
        }
    }

    private func route(request: HTTPRequest, connection: NWConnection) {
        switch (request.method, request.path) {
        case ("GET", "/status"):
            sendJSON(connection: connection, object: statusObject())
        case ("GET", "/log"):
            let lines = Int(request.query["lines"] ?? "") ?? 120
            let text = ProbeRunner.stepFileTail(maxLines: lines)
            send(connection: connection, status: "200 OK", body: Data(text.utf8), contentType: "text/plain; charset=utf-8")
        case ("GET", "/chatlog"):
            let text = ChatSessionLog.readAll()
            send(connection: connection, status: "200 OK", body: Data(text.utf8), contentType: "text/plain; charset=utf-8")
        case ("POST", "/generate"):
            handleGenerate(request: request, connection: connection)
        default:
            send(connection: connection, status: "404 Not Found", body: Data("not found\n".utf8), contentType: "text/plain")
        }
    }

    private func statusObject() -> [String: Any] {
        var object: [String: Any] = [
            "decoder_variant": ProbeSequenceLength.decoderVariant,
            "endpoint_variant": ProbeSequenceLength.endpointVariant,
            "recommended_retain": ProbeRunner.recommendedRetainedDecoderModelCount,
            "keep_e5_cache": ProbeRunner.keepsE5Cache,
            "seq_len": ProbeRunner.selectedSequenceLengthFromProcess().rawValue,
            "image_scale": ProbeRunner.imageHiddenScale,
            "memory_mb": ProbeMemory.currentMB(),
            "available_mb": ProbeMemory.availableMB()
        ]
        DispatchQueue.main.sync {
            if let vm = self.chatViewModel {
                object["is_generating"] = vm.isGenerating
                object["message_count"] = vm.messages.count
            }
        }
        return object
    }

    private func handleGenerate(request: HTTPRequest, connection: NWConnection) {
        guard let body = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
              let prompt = body["prompt"] as? String, !prompt.isEmpty else {
            send(connection: connection, status: "400 Bad Request", body: Data("expected JSON {\"prompt\": ...}\n".utf8), contentType: "text/plain")
            return
        }
        let tokens = body["tokens"] as? Int
        // Optional image attachment: base64 JPEG/PNG plus normalization mode
        // ("unit" -> [0,1], "signed" -> [-1,1]) so the image path can be
        // exercised and A/B-tested entirely from outside the device.
        var image: UIImage?
        if let imageB64 = body["image_b64"] as? String {
            guard let data = Data(base64Encoded: imageB64), let decoded = UIImage(data: data) else {
                send(connection: connection, status: "400 Bad Request", body: Data("image_b64 is not decodable image data\n".utf8), contentType: "text/plain")
                return
            }
            image = decoded
        }
        let normSigned = (body["image_norm"] as? String) == "signed"
        // Optional image_hidden scale override (A/B the scale hypothesis without
        // reconverting the embedder). Persists on ProbeRunner for this request.
        if let imageScale = body["image_scale"] as? Double {
            ProbeRunner.imageHiddenScale = Float(imageScale)
        }
        let started = Date()

        DispatchQueue.main.async { [weak self] in
            guard let self, let vm = self.chatViewModel else { return }
            guard !vm.isGenerating else {
                self.send(connection: connection, status: "409 Conflict", body: Data("generation already running\n".utf8), contentType: "text/plain")
                return
            }
            // tokens=0 (or omitted-as-0) means auto: run until <eos>.
            if let tokens { vm.generatedTokenCount = min(max(tokens, 0), ProbeRunner.maxGeneratedTokenCount) }
            if let image {
                vm.attachedImage = image
                vm.imageNormSigned = normSigned
            }
            vm.messageText = prompt
            vm.send { success, summary in
                DispatchQueue.main.async {
                    let reply = vm.messages.last(where: { $0.role == .assistant })
                    let object: [String: Any] = [
                        "ok": success,
                        "summary": summary,
                        "reply_text": reply?.text ?? "",
                        "reply_tokens": reply?.tokens ?? [],
                        "seconds": Date().timeIntervalSince(started)
                    ]
                    self.queue.async {
                        self.sendJSON(connection: connection, object: object)
                    }
                }
            }
        }
    }

    // MARK: - Response helpers

    private func sendJSON(connection: NWConnection, object: [String: Any]) {
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted])) ?? Data("{}".utf8)
        send(connection: connection, status: "200 OK", body: data, contentType: "application/json")
    }

    private func send(connection: NWConnection, status: String, body: Data, contentType: String) {
        var header = "HTTP/1.1 \(status)\r\n"
        header += "Content-Type: \(contentType)\r\n"
        header += "Content-Length: \(body.count)\r\n"
        header += "Connection: close\r\n\r\n"
        var payload = Data(header.utf8)
        payload.append(body)
        connection.send(content: payload, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    private static func wifiIPv4Address() -> String? {
        var address: String?
        var interfaces: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&interfaces) == 0, let first = interfaces else { return nil }
        defer { freeifaddrs(interfaces) }
        var pointer: UnsafeMutablePointer<ifaddrs>? = first
        while let current = pointer {
            let interface = current.pointee
            if interface.ifa_addr.pointee.sa_family == UInt8(AF_INET),
               String(cString: interface.ifa_name) == "en0" {
                var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                getnameinfo(interface.ifa_addr, socklen_t(interface.ifa_addr.pointee.sa_len),
                            &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST)
                address = String(cString: host)
            }
            pointer = interface.ifa_next
        }
        return address
    }
}

private struct HTTPRequest {
    let method: String
    let path: String
    let query: [String: String]
    let body: Data

    init?(raw: Data) {
        guard let headerEnd = raw.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        guard let head = String(data: raw[..<headerEnd.lowerBound], encoding: .utf8) else { return nil }
        let lines = head.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else { return nil }
        method = String(parts[0]).uppercased()

        let target = String(parts[1])
        if let questionMark = target.firstIndex(of: "?") {
            path = String(target[..<questionMark])
            var query: [String: String] = [:]
            for pair in target[target.index(after: questionMark)...].split(separator: "&") {
                let keyValue = pair.split(separator: "=", maxSplits: 1)
                if keyValue.count == 2 {
                    query[String(keyValue[0])] = String(keyValue[1]).removingPercentEncoding ?? String(keyValue[1])
                }
            }
            self.query = query
        } else {
            path = target
            query = [:]
        }

        var contentLength = 0
        for line in lines.dropFirst() {
            let keyValue = line.split(separator: ":", maxSplits: 1)
            if keyValue.count == 2, keyValue[0].trimmingCharacters(in: .whitespaces).lowercased() == "content-length" {
                contentLength = Int(keyValue[1].trimmingCharacters(in: .whitespaces)) ?? 0
            }
        }

        let bodyStart = headerEnd.upperBound
        let available = raw.count - raw.distance(from: raw.startIndex, to: bodyStart)
        guard available >= contentLength else { return nil }
        body = raw.subdata(in: bodyStart..<raw.index(bodyStart, offsetBy: contentLength))
    }
}

/// Append-only JSONL chat transcript in Documents, one line per message, with
/// timings. Fetchable via GET /chatlog or `devicectl device copy from`.
enum ChatSessionLog {
    private static var url: URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("chat-log.jsonl")
    }

    static func append(role: String, text: String, tokens: [Int], seconds: Double?, detail: String?) {
        guard let url else { return }
        var object: [String: Any] = [
            "ts": ISO8601DateFormatter().string(from: Date()),
            "role": role,
            "text": text,
            "tokens": tokens
        ]
        if let seconds { object["seconds"] = seconds }
        if let detail { object["detail"] = detail }
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              var line = String(data: data, encoding: .utf8) else { return }
        line += "\n"
        if FileManager.default.fileExists(atPath: url.path) {
            if let handle = try? FileHandle(forWritingTo: url) {
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: Data(line.utf8))
                try? handle.close()
            }
        } else {
            try? Data(line.utf8).write(to: url)
        }
    }

    static func readAll() -> String {
        guard let url, let text = try? String(contentsOf: url, encoding: .utf8) else { return "" }
        return text
    }
}
