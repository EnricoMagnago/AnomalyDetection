#ifndef DATA_LOADER_HPP__
#define DATA_LOADER_HPP__

#include "DataTypes.hpp"
#include <string>
#include <set>
#include <ctime>
#include <iostream>


std::string time_to_string(const Data& data, const size_t index);

std::string anomaly_to_string(const Anomaly& anomaly);

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

     log: tells after how many records to log the current advancement. -1: no log.
  */
  void load_all(Data& data, const int log = 200);

  /* fill data by reading the files,
     in data the value -2 is used to identify
     missing or wrong measures.

     log: tells after how many records to log the current advancement. -1: no log.
     max: number of samples to load.
  */
  void load_subset(Data& data, const unsigned long max = 100000, const int log = 200);

private:
  static constexpr unsigned int expected_measures = 634804;

  struct Comparator {
    bool operator()(const std::pair<time_t, std::string>& lhs,
                    const std::pair<time_t, std::string>& rhs) const
    {
      return lhs.first < rhs.first;
    }
  };
  /* store set of files ordered by date */
  std::set<std::pair<time_t, std::string>, DataLoader::Comparator> files_;

  struct AnomalyComparator {
    bool operator()(const Anomaly& lhs,
                    const Anomaly& rhs) const
    {
      if (lhs.begin == rhs.begin) {
        if (lhs.end == rhs.end) {
          if (lhs.tanks.size() == rhs.tanks.size()) {
            if (lhs.description.compare(rhs.description) == 0) {
              std::cerr << "found 2 equal Anomalies" << std::endl;
              exit(1);
            }
            return lhs.description.compare(rhs.description) < 0;
          }
          return lhs.tanks.size() < rhs.tanks.size();
        }
        return lhs.end < rhs.end;
      }
      return lhs.begin < rhs.begin;
    }
  };
  std::set<Anomaly, AnomalyComparator> anomalies;


  void load_file(Data& data, const std::string& file_name) const;

  /* parse line set date, return its index in the string */
  std::size_t parse_line_date(const std::string& line, std::time_t& date) const;

  void parse_line_data(const std::string& line, const std::size_t& date_index,
                       TankMeasures&  measures,
                       PowerMeasures& powers) const;

  void parse_anomalies(std::ifstream& input);

  void add_anomalies(Data& data, const time_t time, const size_t index) const;
};

#endif /* DATA_LOADER_HPP__ */
