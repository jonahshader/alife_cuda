#include "shader.h"

#include <glad/glad.h>

#include <fstream>
#include <iostream>

Shader::Shader(const char *vertex_path, const char *fragment_path) {
  // retrieve the vertex/fragment source code from filePath
  std::string vertex_code;
  std::string fragment_code;
  std::ifstream v_shader_file;
  std::ifstream f_shader_file;
  // ensure ifstream objects can throw exceptions
  v_shader_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  f_shader_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // open files
    v_shader_file.open(vertex_path);
    f_shader_file.open(fragment_path);
    std::stringstream v_shader_stream, f_shader_stream;
    // read file's buffer contents into streams
    v_shader_stream << v_shader_file.rdbuf();
    f_shader_stream << f_shader_file.rdbuf();
    // close file handlers
    v_shader_file.close();
    f_shader_file.close();
    // convert stream into string
    vertex_code = v_shader_stream.str();
    fragment_code = f_shader_stream.str();
  } catch (std::ifstream::failure &e) {
    std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ" << std::endl;
  }
  const char *v_shader_code = vertex_code.c_str();
  const char *f_shader_code = fragment_code.c_str();

  // compile shaders
  unsigned int vertex, fragment;
  int success;
  char info_log[512];

  // vertex shader
  vertex = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex, 1, &v_shader_code, NULL);
  glCompileShader(vertex);
  // print errors if any
  glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertex, 512, NULL, info_log);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << info_log << std::endl;
  }

  // fragment shader
  fragment = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment, 1, &f_shader_code, NULL);
  glCompileShader(fragment);
  // print errors if any
  glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragment, 512, NULL, info_log);
    std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << info_log << std::endl;
  }

  // shader program
  ID = glCreateProgram();
  glAttachShader(ID, vertex);
  glAttachShader(ID, fragment);
  glLinkProgram(ID);
  // print linking errors if any
  glGetProgramiv(ID, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(ID, 512, NULL, info_log);
    std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << info_log << std::endl;
  }

  // delete the shaders as they're linked into our program now and are no longer nessesary
  glDeleteShader(vertex);
  glDeleteShader(fragment);
}

void Shader::use() const {
  glUseProgram(ID);
}

void Shader::set_bool(const std::string &name, bool value) const {
  glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}

void Shader::set_int(const std::string &name, int value) const {
  glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::set_uint(const std::string &name, unsigned int value) const {
  glUniform1ui(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::set_float(const std::string &name, float value) const {
  auto location = glGetUniformLocation(ID, name.c_str());
  if (location < 0) {
    std::cout << "ERROR: Uniform location now found! Location: " << location << " Name: " << name
              << std::endl;
  }
  glUniform1f(location, value);
}

void Shader::set_matrix4(const std::string &name, glm::mat4 value) const {
  glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &value[0][0]);
}

void Shader::set_vec3i(const std::string &name, int x, int y, int z) const {
  glUniform3i(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void Shader::set_vec3(const std::string &name, float x, float y, float z) const {
  glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
}
