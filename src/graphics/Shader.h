#pragma once

#include <glm/glm.hpp>

#include <string>
#include <sstream>

class Shader {
public:
  unsigned int ID; // opengl program id

  // constructor reads and builds the shader
  Shader(const char *vertexPath, const char *fragmentPath);
  // use/activate the shader
  void use() const;
  // utility uniform functions
  void setBool(const std::string &name, bool value) const;
  void setInt(const std::string &name, int value) const;
  void setUInt(const std::string &name, unsigned int value) const;
  void setFloat(const std::string &name, float value) const;
  void setMatrix4(const std::string &name, glm::mat4 value) const;
  void setVec3i(const std::string &name, int x, int y, int z) const;
  void setVec3(const std::string &name, float x, float y, float z) const;
};
