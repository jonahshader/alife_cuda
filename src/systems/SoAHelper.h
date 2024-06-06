#pragma once

#define DEFINE_STRUCT(StructName, MacroName) \
    struct StructName { MacroName(DEF_SCALAR, DEF_SCALAR_WITH_INIT) };

#define DEFINE_STRUCT_PTR(StructName, MacroName) \
    struct StructName##Ptrs { \
    MacroName(DEF_SCALAR_PTR, DEF_SCALAR_PTR) \
\
    void get_ptrs(StructName##SoADevice& s) { \
        MacroName(SET_PTR, SET_PTR) \
    } \
    void get_ptrs(StructName##SoA& s) { \
        MacroName(SET_PTR_HOST, SET_PTR_HOST) \
    } \
};

#define DEFINE_SOA_STRUCT(StructName, MacroName)            \
    struct StructName##SoA {                                \
        MacroName(DEF_VECTOR, DEF_VECTOR)                   \
                                                            \
        void push_back(const StructName& single) {          \
             MacroName(PUSH_BACK_SINGLE, PUSH_BACK_SINGLE)  \
        }                                                   \
                                                            \
        void push_back(const std::vector<StructName>& vec) {\
            for (const auto& s : vec) {                     \
                push_back(s);                               \
            }                                               \
        }                                                   \
                                                            \
        void swap_all(StructName##SoA &s) {                 \
             MacroName(SWAP, SWAP)                          \
        }                                                   \
                                                            \
        void resize_all(size_t new_size) {                  \
            MacroName(RESIZE_WITHOUT_INIT, RESIZE_WITH_INIT)\
        }                                                   \
    };

#define DEFINE_DEVICE_SOA_STRUCT(StructName, MacroName) \
    struct StructName##SoADevice {                      \
        MacroName(DEF_DEVICE_VECTOR, DEF_DEVICE_VECTOR) \
                                                        \
    void copy_from_host(const StructName##SoA& host) {  \
        MacroName(COPY_FROM_HOST, COPY_FROM_HOST)       \
    }                                                   \
                                                        \
    void copy_to_host(StructName##SoA& host) const {    \
        MacroName(COPY_TO_HOST, COPY_TO_HOST)           \
    }                                                   \
                                                        \
};

#define DEFINE_STRUCTS(StructName, MacroName) \
    DEFINE_STRUCT(StructName, MacroName)      \
    DEFINE_SOA_STRUCT(StructName, MacroName)  \
    DEFINE_DEVICE_SOA_STRUCT(StructName, MacroName) \
    DEFINE_STRUCT_PTR(StructName, MacroName)

#define DEF_SCALAR(type, name) type name{};
#define DEF_SCALAR_WITH_INIT(type, name, init) type name{init};
#define DEF_SCALAR_PTR(type, name, ...) type* name{nullptr};
#define DEF_VECTOR(type, name, ...) thrust::host_vector<type> name{};
#define DEF_DEVICE_VECTOR(type, name, ...) thrust::device_vector<type> name{};
#define PUSH_BACK_SINGLE(type, name, ...) name.push_back(single.name);
#define SWAP(type, name, ...) name.swap(s.name);
// TODO: avoid reallocating vectors?
#define COPY_FROM_HOST(type, name, ...) name = host.name;
#define COPY_TO_HOST(type, name, ...) host.name = name;
#define SET_PTR(type, name, ...) name = s.name.data().get();
#define SET_PTR_HOST(type, name, ...) name = s.name.data();
#define RESIZE_WITHOUT_INIT(type, name, ...) name.resize(new_size);
#define RESIZE_WITH_INIT(type, name, init) name.resize(new_size, init);