// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: common.proto

#include "common.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace sparse_net_library {
class Synapse_intervalDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<Synapse_interval> _instance;
} _Synapse_interval_default_instance_;
}  // namespace sparse_net_library
static void InitDefaultsscc_info_Synapse_interval_common_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::sparse_net_library::_Synapse_interval_default_instance_;
    new (ptr) ::sparse_net_library::Synapse_interval();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::sparse_net_library::Synapse_interval::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_Synapse_interval_common_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_Synapse_interval_common_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_common_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_common_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_common_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_common_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::sparse_net_library::Synapse_interval, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::sparse_net_library::Synapse_interval, starts_),
  PROTOBUF_FIELD_OFFSET(::sparse_net_library::Synapse_interval, interval_size_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::sparse_net_library::Synapse_interval)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::sparse_net_library::_Synapse_interval_default_instance_),
};

const char descriptor_table_protodef_common_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\014common.proto\022\022sparse_net_library\"9\n\020Sy"
  "napse_interval\022\016\n\006starts\030\n \001(\021\022\025\n\rinterv"
  "al_size\030\013 \001(\r*\375\001\n\022transfer_functions\022\035\n\031"
  "TRANSFER_FUNCTION_UNKNOWN\020\000\022\036\n\032TRANSFER_"
  "FUNCTION_IDENTITY\020\001\022\035\n\031TRANSFER_FUNCTION"
  "_SIGMOID\020\002\022\032\n\026TRANSFER_FUNCTION_TANH\020\003\022\031"
  "\n\025TRANSFER_FUNCTION_ELU\020\004\022\032\n\026TRANSFER_FU"
  "NCTION_SELU\020\005\022\032\n\026TRANSFER_FUNCTION_RELU\020"
  "\006\022\032\n\025TRANSFER_FUNCTION_END\020\200\004b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_common_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_common_2eproto_sccs[1] = {
  &scc_info_Synapse_interval_common_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_common_2eproto_once;
static bool descriptor_table_common_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_common_2eproto = {
  &descriptor_table_common_2eproto_initialized, descriptor_table_protodef_common_2eproto, "common.proto", 357,
  &descriptor_table_common_2eproto_once, descriptor_table_common_2eproto_sccs, descriptor_table_common_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_common_2eproto::offsets,
  file_level_metadata_common_2eproto, 1, file_level_enum_descriptors_common_2eproto, file_level_service_descriptors_common_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_common_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_common_2eproto), true);
namespace sparse_net_library {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* transfer_functions_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_common_2eproto);
  return file_level_enum_descriptors_common_2eproto[0];
}
bool transfer_functions_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 512:
      return true;
    default:
      return false;
  }
}


// ===================================================================

void Synapse_interval::InitAsDefaultInstance() {
}
class Synapse_interval::_Internal {
 public:
};

Synapse_interval::Synapse_interval()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:sparse_net_library.Synapse_interval)
}
Synapse_interval::Synapse_interval(const Synapse_interval& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&starts_, &from.starts_,
    static_cast<size_t>(reinterpret_cast<char*>(&interval_size_) -
    reinterpret_cast<char*>(&starts_)) + sizeof(interval_size_));
  // @@protoc_insertion_point(copy_constructor:sparse_net_library.Synapse_interval)
}

void Synapse_interval::SharedCtor() {
  ::memset(&starts_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&interval_size_) -
      reinterpret_cast<char*>(&starts_)) + sizeof(interval_size_));
}

Synapse_interval::~Synapse_interval() {
  // @@protoc_insertion_point(destructor:sparse_net_library.Synapse_interval)
  SharedDtor();
}

void Synapse_interval::SharedDtor() {
}

void Synapse_interval::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const Synapse_interval& Synapse_interval::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_Synapse_interval_common_2eproto.base);
  return *internal_default_instance();
}


void Synapse_interval::Clear() {
// @@protoc_insertion_point(message_clear_start:sparse_net_library.Synapse_interval)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ::memset(&starts_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&interval_size_) -
      reinterpret_cast<char*>(&starts_)) + sizeof(interval_size_));
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* Synapse_interval::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // sint32 starts = 10;
      case 10:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 80)) {
          starts_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarintZigZag32(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // uint32 interval_size = 11;
      case 11:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 88)) {
          interval_size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool Synapse_interval::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:sparse_net_library.Synapse_interval)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // sint32 starts = 10;
      case 10: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (80 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::int32, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_SINT32>(
                 input, &starts_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint32 interval_size = 11;
      case 11: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (88 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::uint32, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_UINT32>(
                 input, &interval_size_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:sparse_net_library.Synapse_interval)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:sparse_net_library.Synapse_interval)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

::PROTOBUF_NAMESPACE_ID::uint8* Synapse_interval::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:sparse_net_library.Synapse_interval)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // sint32 starts = 10;
  if (this->starts() != 0) {
    stream->EnsureSpace(&target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteSInt32ToArray(10, this->starts(), target);
  }

  // uint32 interval_size = 11;
  if (this->interval_size() != 0) {
    stream->EnsureSpace(&target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(11, this->interval_size(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:sparse_net_library.Synapse_interval)
  return target;
}

size_t Synapse_interval::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:sparse_net_library.Synapse_interval)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // sint32 starts = 10;
  if (this->starts() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SInt32Size(
        this->starts());
  }

  // uint32 interval_size = 11;
  if (this->interval_size() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32Size(
        this->interval_size());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Synapse_interval::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:sparse_net_library.Synapse_interval)
  GOOGLE_DCHECK_NE(&from, this);
  const Synapse_interval* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Synapse_interval>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:sparse_net_library.Synapse_interval)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:sparse_net_library.Synapse_interval)
    MergeFrom(*source);
  }
}

void Synapse_interval::MergeFrom(const Synapse_interval& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:sparse_net_library.Synapse_interval)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.starts() != 0) {
    set_starts(from.starts());
  }
  if (from.interval_size() != 0) {
    set_interval_size(from.interval_size());
  }
}

void Synapse_interval::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:sparse_net_library.Synapse_interval)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Synapse_interval::CopyFrom(const Synapse_interval& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:sparse_net_library.Synapse_interval)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Synapse_interval::IsInitialized() const {
  return true;
}

void Synapse_interval::InternalSwap(Synapse_interval* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(starts_, other->starts_);
  swap(interval_size_, other->interval_size_);
}

::PROTOBUF_NAMESPACE_ID::Metadata Synapse_interval::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace sparse_net_library
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::sparse_net_library::Synapse_interval* Arena::CreateMaybeMessage< ::sparse_net_library::Synapse_interval >(Arena* arena) {
  return Arena::CreateInternal< ::sparse_net_library::Synapse_interval >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
