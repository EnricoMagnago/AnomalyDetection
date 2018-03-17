#include "DataLoader.hpp"

#include <iostream>
#include <cstring>
#include <iomanip>
#include <istream>
#include <fstream>
#include <experimental/filesystem>

#include <cassert>

namespace fs = std::experimental::filesystem;


DataLoader::DataLoader(const std::string& root_dir) : files_()
{
  /* recursive iteration: iterate over the leaves of the file system */
  for (auto& file : fs::recursive_directory_iterator(root_dir)) {
    /* get string representing the current file */
    const std::string file_name(file.path());
    const size_t separator_index = file_name.find_last_of('/') + 1;
    const size_t length = file_name.size() - 4 - separator_index;
    /* get file name without extension and path:
       if file_name is `/path/to/file.csv`
       date_str contains `file`
    */
    std::istringstream date_str(file_name.substr(separator_index, length));
    /* file name represents a date, parse the date,
       insert pair (date, name) into the set,
       set is ordered by date */
    struct std::tm tm;
    std::memset(&tm, 0, sizeof(struct std::tm));
    date_str >> std::get_time(&tm, "%Y-%m-%d-h%H");
    const std::time_t time = mktime(&tm);

    this->files_.insert(std::make_pair(time, file_name));
  }

}

void DataLoader::load_data(Data& data, const int log)
{
  data.index_to_time.reserve(expected_measures);
  data.measures.reserve(expected_measures);
  const size_t size = files_.size();
  size_t index = 0;
  for (const std::pair<time_t, std::string>& it : files_) {
    if (log != -1 && index % log == 0)
      std::cout << "loading: " << index << "/" << size << std::endl;
    load_file(data, it.second);
    ++index;
  }
}

void DataLoader::load_file(Data& data, const std::string& file_name)
{
  std::ifstream input_file(file_name);
  if (!input_file.good()) {
    std::cerr << "could not open file: " << file_name << std::endl;
    exit(1);
  }
  /* Structures used to accumulate measurements,
     they collect the data for each timestamp,
     at every new time-stamp they are re-initialized.
   */
  TankMeasures  measures;
  PowerMeasures powers;

  std::string line;
  while(std::getline(input_file, line)) {
    /* parse date */
    const std::size_t date_index = line.find_first_of(';');
    std::istringstream date_str(line.substr(0, date_index));
    struct std::tm tm;
    std::memset(&tm, 0, sizeof(struct std::tm));
    date_str >> std::get_time(&tm, "%d/%m/%Y %H:%M:%S");
    const std::time_t date = mktime(&tm);

    /* check if the current line refers to a new date or not */
    if (data.index_to_time.empty() || data.index_to_time.back() != date) {
      /* dates in the input file should be sorted */
      assert(data.index_to_time.empty() || data.index_to_time.back() < date);

      /* if new date, add it to the list */
      data.index_to_time.push_back(date);
      /* push collected data into the list */
      data.measures.push_back(std::make_pair(std::move(measures), std::move(powers)));

      /* re initialize local structures, fill with -2: used to signal missing values */
      measures = TankMeasures();
      for (uint8_t i = 0; i < 3; ++i)
        measures[i].fill(-2);

      powers = PowerMeasures();
      powers.fill(-2);
    }

    /* parse measure type and value */
    const std::size_t measure_type_index = line.find_first_of(';', date_index + 1);
    const std::string measure_type_code =
      line.substr(date_index + 1, measure_type_index - date_index - 1);
    assert(measure_type_code.length() == measures.size());
    const uint8_t tank_id = measure_type_code[0] - '0';
    const uint8_t sensor_id = measure_type_code[2] - '0';
    const double measured_value = stod(line.substr(measure_type_index+1));

    assert(tank_id < 4);
    assert(sensor_id < 6);

    if (tank_id <= measures.size()) {
      assert(measures[tank_id][sensor_id] = -2);
      /* actual tank sensor measure, can not be negative, if it's negative set to -2 */
      measures[tank_id][sensor_id] = (measured_value < 0)? -2 : measured_value;
    }
    else {
      assert(powers[sensor_id] = -2);
      /* this measure refers to power, can not be negative, if it's negative set to -2 */
      powers[sensor_id] = (measured_value < 0)? -2 : measured_value;
    }
  }


  input_file.close();
}