#pragma once

#include <glm/glm.hpp>

#include <sstream>
#include <string>

class Shader {
public:
  unsigned int ID; // opengl program id

  // constructor reads and builds the shader
  Shader(const char *vertex_path, const char *fragment_path);
  // use/activate the shader
  void use() const;
  // utility uniform functions
  void set_bool(const std::string &name, bool value) const;
  void set_int(const std::string &name, int value) const;
  void set_uint(const std::string &name, unsigned int value) const;
  void set_float(const std::string &name, float value) const;
  void set_matrix4(const std::string &name, glm::mat4 value) const;
  void set_vec3i(const std::string &name, int x, int y, int z) const;
  void set_vec3(const std::string &name, float x, float y, float z) const;
};
