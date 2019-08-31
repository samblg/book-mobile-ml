#ifndef TENSORFLOW_LITE_TOCO_TOCO_PORT_H_
#define TENSORFLOW_LITE_TOCO_TOCO_PORT_H_

#include <string>
#include "google/protobuf/text_format.h"
#include "tensorflow/lite/toco/format_port.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#if defined(PLATFORM_GOOGLE)
#include "absl/strings/cord.h"
#endif  // PLATFORM_GOOGLE

#ifdef PLATFORM_GOOGLE
#define TFLITE_PROTO_NS proto2
#else
#define TFLITE_PROTO_NS google::protobuf
#endif

#ifdef __ANDROID__
#include <sstream>
namespace std {

template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}

#ifdef __ARM_ARCH_7A__
double round(double x);
#endif
}
#endif

namespace toco {
namespace port {

void InitGoogleWasDoneElsewhere();
void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags);
void CheckInitGoogleIsDone(const char* message);

namespace file {
class Options {};
inline Options Defaults() {
  Options o;
  return o;
}
tensorflow::Status GetContents(const string& filename, string* contents,
                               const Options& options);
tensorflow::Status SetContents(const string& filename, const string& contents,
                               const Options& options);
string JoinPath(const string& base, const string& filename);
tensorflow::Status Writable(const string& filename);
tensorflow::Status Readable(const string& filename, const Options& options);
tensorflow::Status Exists(const string& filename, const Options& options);
}  // namespace file

#if defined(PLATFORM_GOOGLE)
void CopyToBuffer(const ::Cord& src, char* dest);
#endif  // PLATFORM_GOOGLE
void CopyToBuffer(const string& src, char* dest);

inline uint32 ReverseBits32(uint32 n) {
  n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
  n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
  n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
  return (((n & 0xFF) << 24) | ((n & 0xFF00) << 8) | ((n & 0xFF0000) >> 8) |
          ((n & 0xFF000000) >> 24));
}
}  // namespace port

inline bool ParseFromStringOverload(const std::string& in,
                                    TFLITE_PROTO_NS::Message* proto) {
  return TFLITE_PROTO_NS::TextFormat::ParseFromString(in, proto);
}

template <typename Proto>
bool ParseFromStringEitherTextOrBinary(const std::string& input_file_contents,
                                       Proto* proto) {
  if (proto->ParseFromString(input_file_contents)) {
    return true;
  }

  if (ParseFromStringOverload(input_file_contents, proto)) {
    return true;
  }

  return false;
}

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TOCO_PORT_H_
