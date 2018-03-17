#ifndef DATA_LOADER_HPP__
#define DATA_LOADER_HPP__

#include "DataTypes.hpp"
#include <string>
#include <set>
#include <ctime>

class DataLoader
{
public:

  /* prepare to load files under root_dir,
     files are extracted recursively starting from root_dir
  */
  DataLoader(const std::string& root_dir);

  /* fill data by reading the files,
     in data the value -2 is used to identify
     missing or wrong measures.
  */
  void load_data(Data& data, const int log = -1);

private:
  static constexpr unsigned int expected_measures = 15235628;

  struct Comparator {
    bool operator()(const std::pair<time_t, std::string>& lhs,
                    const std::pair<time_t, std::string>& rhs) const
    {
      return lhs.first < rhs.first;
    }
  };
  /* store set of files ordered by date */
  std::set<std::pair<time_t, std::string>, DataLoader::Comparator> files_;

  void load_file(Data& data, const std::string& file_name);
};

#endif /* DATA_LOADER_HPP__ */
