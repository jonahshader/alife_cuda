#pragma once

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>

class ParameterManager
{
private:
  std::string filename;
  std::unordered_map<std::string, std::string> params;

  void load_file()
  {
    std::ifstream file(filename);
    if (!file)
    {
      std::cout << "File not found. Creating a new file: " << filename << std::endl;
      std::ofstream new_file(filename);
      new_file.close();
      return;
    }
    std::string line;
    while (std::getline(file, line))
    {
      size_t pos = line.find('=');
      if (pos != std::string::npos)
      {
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        params[key] = value;
      }
    }
  }

  static std::string trim(const std::string &str)
  {
    size_t first = str.find_first_not_of(" \t");
    if (std::string::npos == first)
    {
      return str;
    }
    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, (last - first + 1));
  }

public:
  ParameterManager(const std::string &filename) : filename(filename)
  {
    load_file();
  }

  template <typename T>
  void get(const std::string &param_name, T &value)
  {
    if (params.find(param_name) == params.end())
    {
      set(param_name, value);
      return;
    }

    std::istringstream iss(params[param_name]);
    if (!(iss >> value))
    {
      throw std::runtime_error("Failed to convert parameter to requested type");
    }
  }

  template <typename T>
  void get(const std::string &param_name, std::vector<T> &value)
  {
    if (params.find(param_name) == params.end())
    {
      set(param_name, value);
      return;
    }

    value.clear();
    std::istringstream iss(params[param_name]);
    std::string item;
    while (std::getline(iss, item, ','))
    {
      std::istringstream item_stream(trim(item));
      T temp;
      if (item_stream >> temp)
      {
        value.push_back(temp);
      }
      else
      {
        throw std::runtime_error("Failed to convert vector item to requested type");
      }
    }
  }

  template <typename T>
  void set(const std::string &param_name, const T &value)
  {
    std::ostringstream oss;
    oss << value;
    params[param_name] = oss.str();
  }

  template <typename T>
  void set(const std::string &param_name, const std::vector<T> &value)
  {
    std::ostringstream oss;
    for (size_t i = 0; i < value.size(); ++i)
    {
      if (i > 0)
        oss << ", ";
      oss << value[i];
    }
    params[param_name] = oss.str();
  }

  void save()
  {
    std::ofstream file(filename);
    if (!file)
    {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }
    for (const auto &pair : params)
    {
      file << pair.first << "=" << pair.second << std::endl;
    }
  }
};