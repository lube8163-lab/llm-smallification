#import <Foundation/Foundation.h>
#include <stdint.h>

NS_ASSUME_NONNULL_BEGIN

@interface DiffusionBridgeResult : NSObject

@property(nonatomic, readonly) BOOL success;
@property(nonatomic, copy, readonly) NSString *summary;
@property(nonatomic, copy, readonly) NSString *output;
@property(nonatomic, copy, readonly) NSString *log;
@property(nonatomic, readonly) double loadSeconds;
@property(nonatomic, readonly) double generationSeconds;
@property(nonatomic, readonly) double peakFootprintMB;

@end

@interface DiffusionBridge : NSObject

+ (DiffusionBridgeResult *)runWithModelPath:(NSString *)modelPath
                                     prompt:(NSString *)prompt
                                     seqLen:(int32_t)seqLen
                                      steps:(int32_t)steps
                                blockLength:(int32_t)blockLength
                                temperature:(float)temperature
                                       seed:(int32_t)seed
                            formattedPrompt:(BOOL)formattedPrompt
    NS_SWIFT_NAME(run(modelPath:prompt:seqLen:steps:blockLength:temperature:seed:formattedPrompt:));

@end

NS_ASSUME_NONNULL_END
