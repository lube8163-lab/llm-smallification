#import "DiffusionBridge.h"

#include "ggml.h"
#include "llama.h"
#include "../../../external/llama.cpp/examples/diffusion/diffusion.h"

#include <mach/mach.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

@interface DiffusionBridgeResult ()

@property(nonatomic, readwrite) BOOL success;
@property(nonatomic, copy, readwrite) NSString *summary;
@property(nonatomic, copy, readwrite) NSString *output;
@property(nonatomic, copy, readwrite) NSString *log;
@property(nonatomic, readwrite) double loadSeconds;
@property(nonatomic, readwrite) double generationSeconds;
@property(nonatomic, readwrite) double peakFootprintMB;

@end

@implementation DiffusionBridgeResult
@end

static double nowSeconds(void) {
    return CFAbsoluteTimeGetCurrent();
}

static double currentFootprintMB(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t result = task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count);
    if (result != KERN_SUCCESS) {
        return -1.0;
    }
    return static_cast<double>(info.phys_footprint) / 1024.0 / 1024.0;
}

static void appendLog(NSMutableString *log, NSString *format, ...) NS_FORMAT_FUNCTION(2, 3);
static void appendLog(NSMutableString *log, NSString *format, ...) {
    va_list args;
    va_start(args, format);
    NSString *line = [[NSString alloc] initWithFormat:format arguments:args];
    va_end(args);
    [log appendString:line];
    if (![line hasSuffix:@"\n"]) {
        [log appendString:@"\n"];
    }
    fprintf(stdout, "%s", [line UTF8String]);
    if (![line hasSuffix:@"\n"]) {
        fprintf(stdout, "\n");
    }
}

static DiffusionBridgeResult *makeResult(BOOL success,
                                         NSString *summary,
                                         NSString *output,
                                         NSMutableString *log,
                                         double loadSeconds,
                                         double generationSeconds,
                                         double peakFootprintMB) {
    DiffusionBridgeResult *result = [DiffusionBridgeResult new];
    result.success = success;
    result.summary = summary ?: @"";
    result.output = output ?: @"";
    result.log = [log copy] ?: @"";
    result.loadSeconds = loadSeconds;
    result.generationSeconds = generationSeconds;
    result.peakFootprintMB = peakFootprintMB;
    return result;
}

static NSString *stringFromBytes(const std::string &bytes) {
    NSString *text = [[NSString alloc] initWithBytes:bytes.data()
                                             length:bytes.size()
                                           encoding:NSUTF8StringEncoding];
    return text ?: [[NSString alloc] initWithBytes:bytes.data()
                                           length:bytes.size()
                                         encoding:NSISOLatin1StringEncoding] ?: @"";
}

static bool tokenizePrompt(const llama_vocab *vocab,
                           const std::string &text,
                           std::vector<llama_token> &tokens,
                           std::string &error) {
    int32_t needed = llama_tokenize(vocab,
                                    text.c_str(),
                                    static_cast<int32_t>(text.size()),
                                    nullptr,
                                    0,
                                    true,
                                    true);
    if (needed == INT32_MIN) {
        error = "tokenization overflow";
        return false;
    }
    if (needed < 0) {
        needed = -needed;
    }
    if (needed <= 0) {
        error = "tokenization returned no tokens";
        return false;
    }

    tokens.resize(needed);
    int32_t actual = llama_tokenize(vocab,
                                    text.c_str(),
                                    static_cast<int32_t>(text.size()),
                                    tokens.data(),
                                    static_cast<int32_t>(tokens.size()),
                                    true,
                                    true);
    if (actual < 0) {
        error = "token buffer was too small";
        return false;
    }
    tokens.resize(actual);
    return true;
}

static std::string detokenize(const llama_vocab *vocab, const std::vector<llama_token> &tokens) {
    if (tokens.empty()) {
        return "";
    }

    std::vector<char> buffer(8192);
    int32_t written = llama_detokenize(vocab,
                                       tokens.data(),
                                       static_cast<int32_t>(tokens.size()),
                                       buffer.data(),
                                       static_cast<int32_t>(buffer.size()),
                                       true,
                                       false);
    if (written < 0) {
        buffer.resize(static_cast<size_t>(-written) + 1);
        written = llama_detokenize(vocab,
                                   tokens.data(),
                                   static_cast<int32_t>(tokens.size()),
                                   buffer.data(),
                                   static_cast<int32_t>(buffer.size()),
                                   true,
                                   false);
    }
    if (written <= 0) {
        return "";
    }
    return std::string(buffer.data(), static_cast<size_t>(written));
}

static std::string formatLladaMoEPrompt(const std::string &userPrompt) {
    return "<role>SYSTEM</role>Reply shortly.\ndetailed thinking off<|role_end|><role>HUMAN</role>"
        + userPrompt
        + "<|role_end|><role>ASSISTANT</role>";
}

struct StepCallbackState {
    int32_t nInput;
    double startSeconds;
    __unsafe_unretained NSMutableString *log;
};

static bool diffusionProbeStepCallback(int32_t step,
                                       int32_t totalSteps,
                                       const llama_token *tokens,
                                       int32_t nTokens,
                                       void *userData) {
    (void) tokens;
    (void) nTokens;
    StepCallbackState *state = static_cast<StepCallbackState *>(userData);
    double elapsed = nowSeconds() - state->startSeconds;
    appendLog(state->log, @"[DiffusionProbe] step %d/%d elapsed=%.3fs", step + 1, totalSteps, elapsed);
    return true;
}

struct PersistentDiffusionSession {
    bool backendStarted = false;
    std::string modelPath;
    int32_t seqLen = 0;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;

    ~PersistentDiffusionSession() {
        unloadAll();
        if (backendStarted) {
            llama_backend_free();
            backendStarted = false;
        }
    }

    void ensureBackend() {
        if (!backendStarted) {
            ggml_time_init();
            llama_backend_init();
            backendStarted = true;
        }
    }

    void unloadContext() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        seqLen = 0;
    }

    void unloadAll() {
        unloadContext();
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        modelPath.clear();
    }

    bool prepare(NSString *requestedModelPath,
                 int32_t requestedSeqLen,
                 NSMutableString *log,
                 double &loadSeconds,
                 double &peakFootprint,
                 NSString **error) {
        ensureBackend();

        std::string nextPath([requestedModelPath fileSystemRepresentation]);
        if (model && modelPath != nextPath) {
            appendLog(log, @"[DiffusionProbe] unloading previous model");
            unloadAll();
        }

        bool loadedModel = false;
        bool createdContext = false;

        if (!model) {
            double start = nowSeconds();
            llama_model_params modelParams = llama_model_default_params();
            modelParams.n_gpu_layers = -1;
            modelParams.use_mmap = true;
            modelParams.use_mlock = false;

            model = llama_model_load_from_file(nextPath.c_str(), modelParams);
            loadSeconds += nowSeconds() - start;
            peakFootprint = std::max(peakFootprint, currentFootprintMB());
            loadedModel = true;

            if (!model) {
                if (error) {
                    *error = @"Failed to load model";
                }
                return false;
            }
            if (!llama_model_is_diffusion(model)) {
                unloadAll();
                if (error) {
                    *error = @"Model is not marked as diffusion";
                }
                return false;
            }
            modelPath = nextPath;
        }

        if (!ctx || seqLen != requestedSeqLen) {
            unloadContext();

            llama_context_params ctxParams = llama_context_default_params();
            ctxParams.n_ctx = static_cast<uint32_t>(requestedSeqLen);
            ctxParams.n_batch = static_cast<uint32_t>(requestedSeqLen);
            ctxParams.n_ubatch = static_cast<uint32_t>(requestedSeqLen);
            ctxParams.no_perf = false;

            double start = nowSeconds();
            ctx = llama_init_from_model(model, ctxParams);
            loadSeconds += nowSeconds() - start;
            peakFootprint = std::max(peakFootprint, currentFootprintMB());
            createdContext = true;

            if (!ctx) {
                if (error) {
                    *error = @"Failed to create llama context";
                }
                return false;
            }
            seqLen = requestedSeqLen;
            llama_set_n_threads(ctx, 4, 4);
        }

        appendLog(log,
                  @"[DiffusionProbe] session model=%@ context=%@",
                  loadedModel ? @"loaded" : @"reused",
                  createdContext ? @"created" : @"reused");
        return true;
    }
};

static PersistentDiffusionSession &sharedSession() {
    static PersistentDiffusionSession session;
    return session;
}

static std::mutex &sharedSessionMutex() {
    static std::mutex mutex;
    return mutex;
}

@implementation DiffusionBridge

+ (DiffusionBridgeResult *)runWithModelPath:(NSString *)modelPath
                                     prompt:(NSString *)prompt
                                     seqLen:(int32_t)seqLen
                                      steps:(int32_t)steps
                                blockLength:(int32_t)blockLength
                                temperature:(float)temperature
                                       seed:(int32_t)seed
                            formattedPrompt:(BOOL)formattedPrompt {
    NSMutableString *log = [NSMutableString string];
    double peakFootprint = currentFootprintMB();
    double loadSeconds = 0.0;
    double generationSeconds = 0.0;

    if (modelPath.length == 0) {
        return makeResult(NO, @"Missing model path", @"", log, loadSeconds, generationSeconds, peakFootprint);
    }
    if (seqLen <= 0 || steps <= 0 || blockLength <= 0) {
        return makeResult(NO, @"Invalid diffusion parameters", @"", log, loadSeconds, generationSeconds, peakFootprint);
    }
    if (temperature < 0.0f || temperature > 5.0f) {
        return makeResult(NO, @"Invalid temperature", @"", log, loadSeconds, generationSeconds, peakFootprint);
    }
    if (seqLen % blockLength != 0 || steps % (seqLen / blockLength) != 0) {
        return makeResult(NO, @"seqLen/blockLength and steps must align for block scheduling", @"", log, loadSeconds, generationSeconds, peakFootprint);
    }

    std::lock_guard<std::mutex> lock(sharedSessionMutex());

    appendLog(log, @"[DiffusionProbe] model=%@", modelPath);
    appendLog(log, @"[DiffusionProbe] seqLen=%d steps=%d blockLength=%d temperature=%.3f seed=%d", seqLen, steps, blockLength, temperature, seed);

    NSString *prepareError = nil;
    PersistentDiffusionSession &session = sharedSession();
    if (!session.prepare(modelPath, seqLen, log, loadSeconds, peakFootprint, &prepareError)) {
        return makeResult(NO, prepareError ?: @"Failed to prepare diffusion session", @"", log, loadSeconds, generationSeconds, peakFootprint);
    }

    llama_model *model = session.model;
    llama_context *ctx = session.ctx;

    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> inputTokens;
    std::string tokenError;
    std::string promptString = prompt ? std::string([prompt UTF8String]) : std::string();
    std::string promptBytes = formattedPrompt ? promptString : formatLladaMoEPrompt(promptString);
    if (!tokenizePrompt(vocab, promptBytes, inputTokens, tokenError)) {
        return makeResult(NO, stringFromBytes(tokenError), @"", log, loadSeconds, generationSeconds, peakFootprint);
    }
    if (static_cast<int32_t>(inputTokens.size()) >= seqLen) {
        return makeResult(NO, @"Prompt is too long for seqLen", @"", log, loadSeconds, generationSeconds, peakFootprint);
    }
    if (seqLen - static_cast<int32_t>(inputTokens.size()) < 8) {
        return makeResult(NO, @"Prompt leaves too little room for output", @"", log, loadSeconds, generationSeconds, peakFootprint);
    }

    llama_token maskToken = llama_vocab_mask(vocab);
    if (maskToken == LLAMA_TOKEN_NULL) {
        return makeResult(NO, @"Model has no mask token", @"", log, loadSeconds, generationSeconds, peakFootprint);
    }

    diffusion_params params;
    params.steps = steps;
    params.temperature = temperature;
    params.mask_token_id = maskToken;
    params.seed = seed;
    params.algorithm = DIFFUSION_ALGORITHM_CONFIDENCE_BASED;
    params.schedule = DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED;
    params.block_length = blockLength;
    params.max_length = seqLen;
    params.top_p = 1.0f;
    params.top_k = 0;

    char shiftLogits[8] = {};
    if (llama_model_meta_val_str(model, "diffusion.shift_logits", shiftLogits, sizeof(shiftLogits)) >= 0) {
        params.shift_logits = strcmp(shiftLogits, "true") == 0;
    } else {
        params.shift_logits = true;
    }

    StepCallbackState callbackState = {
        static_cast<int32_t>(inputTokens.size()),
        nowSeconds(),
        log,
    };
    params.step_callback = diffusionProbeStepCallback;
    params.step_callback_user_data = &callbackState;

    appendLog(log, @"[DiffusionProbe] inputTokens=%zu maskToken=%d shiftLogits=%@", inputTokens.size(), maskToken, params.shift_logits ? @"true" : @"false");

    llama_memory_clear(llama_get_memory(ctx), true);

    std::vector<llama_token> outputTokens(static_cast<size_t>(seqLen));
    int32_t nGenerated = 0;
    double generationStart = nowSeconds();
    diffusion_generate(ctx,
                       inputTokens.data(),
                       outputTokens.data(),
                       static_cast<int32_t>(inputTokens.size()),
                       params,
                       nGenerated);
    llama_synchronize(ctx);
    generationSeconds = nowSeconds() - generationStart;
    peakFootprint = std::max(peakFootprint, currentFootprintMB());

    NSString *summary = @"Diffusion generation failed";
    NSString *output = @"";
    BOOL success = NO;
    if (nGenerated > 0) {
        std::vector<llama_token> suffix;
        for (int32_t i = static_cast<int32_t>(inputTokens.size()); i < std::min(nGenerated, seqLen); i++) {
            llama_token token = outputTokens[static_cast<size_t>(i)];
            if (token == maskToken || llama_vocab_is_eog(vocab, token) || llama_vocab_is_control(vocab, token)) {
                continue;
            }
            suffix.push_back(token);
        }
        output = stringFromBytes(detokenize(vocab, suffix));
        summary = [NSString stringWithFormat:@"OK load=%.3fs generation=%.3fs footprint=%.1fMB", loadSeconds, generationSeconds, peakFootprint];
        success = YES;
        appendLog(log, @"[DiffusionProbe] %@", summary);
        appendLog(log, @"[DiffusionProbe] output=%@", output);
    }

    return makeResult(success, summary, output, log, loadSeconds, generationSeconds, peakFootprint);
}

@end

#include "../../../external/llama.cpp/examples/diffusion/diffusion.cpp"
