#import <Foundation/Foundation.h>

@interface Foo : NSObject
- (int)add:(int)a to:(int)b;
@end

@implementation Foo
- (int)add:(int)a to:(int)b {
    return a + b;
}
@end
