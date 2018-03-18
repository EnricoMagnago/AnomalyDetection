%include std_string.i

%module DataLoader
%{
    #define SWIG_FILE_WITH_INIT
    #include "../DataLoader.hpp"
%}

class DataLoader
{
public:

  DataLoader(const std::string& root_dir);
  void load_data(Data& data, const int log = -1);
};