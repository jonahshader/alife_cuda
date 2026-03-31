#pragma once

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Buffer type aliases
template <typename T>
using HostBuffer = thrust::host_vector<T>;

template <typename T>
using DeviceBuffer = thrust::device_vector<T>;

// Raw pointer extraction — works for both host and device vectors
template <typename T>
T *raw_ptr(thrust::host_vector<T> &v) {
  return v.data();
}

template <typename T>
T *raw_ptr(thrust::device_vector<T> &v) {
  return v.data().get();
}

// --- Field-level macro helpers ---

// Scalar struct fields
#define DEF_SCALAR(type, name) type name{};
#define DEF_SCALAR_WITH_INIT(type, name, init) type name{init};

// Templated SoA struct fields
#define DEF_BUFFER_FIELD(type, name, ...) Buffer<type> name{};

// Ptrs struct fields
#define DEF_PTR_FIELD(type, name, ...) type *name{nullptr};
#define SET_PTR_FIELD(type, name, ...) name = raw_ptr(soa.name);

// Free function helpers
#define COPY_FIELD_OP(type, name, ...) dst.name = src.name;
#define SWAP_FIELD_OP(type, name, ...) a.name.swap(b.name);
#define RESIZE_FIELD_NO_INIT(type, name, ...) soa.name.resize(n);
#define RESIZE_FIELD_WITH_INIT(type, name, init) soa.name.resize(n, init);
#define PUSH_BACK_FIELD_OP(type, name, ...) soa.name.push_back(single.name);

// --- Main macro: generates scalar, templated SoA, Ptrs, and free functions ---

#define DEFINE_STRUCTS(Name, FIELDS)                                                               \
  /* Scalar struct (single element) */                                                             \
  struct Name {                                                                                    \
    FIELDS(DEF_SCALAR, DEF_SCALAR_WITH_INIT)                                                       \
  };                                                                                               \
                                                                                                   \
  /* Templated SoA struct — Buffer can be HostBuffer or DeviceBuffer */                            \
  template <template <typename> class Buffer>                                                      \
  struct Name##SoA {                                                                               \
    FIELDS(DEF_BUFFER_FIELD, DEF_BUFFER_FIELD)                                                     \
  };                                                                                               \
                                                                                                   \
  /* Ptrs struct with templated get_ptrs for kernel access */                                      \
  struct Name##Ptrs {                                                                              \
    FIELDS(DEF_PTR_FIELD, DEF_PTR_FIELD)                                                           \
                                                                                                   \
    template <template <typename> class Buffer>                                                    \
    void get_ptrs(Name##SoA<Buffer> &soa) {                                                        \
      FIELDS(SET_PTR_FIELD, SET_PTR_FIELD)                                                         \
    }                                                                                              \
  };                                                                                               \
                                                                                                   \
  /* Copy between any two buffer backends */                                                       \
  template <template <typename> class Dst, template <typename> class Src>                          \
  void copy(Name##SoA<Dst> &dst, const Name##SoA<Src> &src) {                                      \
    FIELDS(COPY_FIELD_OP, COPY_FIELD_OP)                                                           \
  }                                                                                                \
                                                                                                   \
  /* Swap within same backend */                                                                   \
  template <template <typename> class Buffer>                                                      \
  void swap_all(Name##SoA<Buffer> &a, Name##SoA<Buffer> &b) {                                      \
    FIELDS(SWAP_FIELD_OP, SWAP_FIELD_OP)                                                           \
  }                                                                                                \
                                                                                                   \
  /* Resize all fields */                                                                          \
  template <template <typename> class Buffer>                                                      \
  void resize_all(Name##SoA<Buffer> &soa, size_t n) {                                              \
    FIELDS(RESIZE_FIELD_NO_INIT, RESIZE_FIELD_WITH_INIT)                                           \
  }                                                                                                \
                                                                                                   \
  /* Push back a single scalar element */                                                          \
  template <template <typename> class Buffer>                                                      \
  void push_back(Name##SoA<Buffer> &soa, const Name &single) {                                     \
    FIELDS(PUSH_BACK_FIELD_OP, PUSH_BACK_FIELD_OP)                                                 \
  }                                                                                                \
                                                                                                   \
  /* Push back a vector of scalar elements */                                                      \
  template <template <typename> class Buffer>                                                      \
  void push_back(Name##SoA<Buffer> &soa, const std::vector<Name> &vec) {                           \
    for (const auto &single : vec) {                                                               \
      push_back(soa, single);                                                                      \
    }                                                                                              \
  }
