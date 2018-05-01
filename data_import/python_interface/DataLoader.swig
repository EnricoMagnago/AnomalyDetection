%include std_string.i

%module DataLoader
%{
    #define SWIG_FILE_WITH_INIT
    #include "../DataLoader.hpp"
%}

std::string time_to_string(const Data& data, const size_t index);

class DataLoader
{
public:

  DataLoader(const std::string& root_dir);
  void load_all(Data& data, const int log = 200);
  void load_subset(Data& data, const unsigned long max = 100000, const int log = 200);
};