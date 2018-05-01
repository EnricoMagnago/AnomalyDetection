#include "DataLoader.hpp"

#include <iostream>
#include <cstring>
#include <iomanip>
#include <istream>
#include <fstream>
#include <experimental/filesystem>

#include <cassert>

namespace fs = std::experimental::filesystem;

std::string time_to_string(const Data& data, const size_t index)
{
  const time_t time = data.index_to_time[index];
  std::stringstream s;
  s << std::put_time(std::gmtime(&time), "%c %Z");
  return s.str();
}


DataLoader::DataLoader(const std::string& root_dir) : files_()
{
  /* recursive iteration: iterate over the leaves of the file system */
  for (auto& file : fs::recursive_directory_iterator(root_dir)) {
    /* get string representing the current file */
    const std::string file_name(file.path());
    if (file_name.size() > 4 &&
        file_name.substr(file_name.size() - 4, file_name.size()) == ".csv") {
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

}

void DataLoader::load_all(Data& data, const int log)
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

void DataLoader::load_subset(Data& data, const unsigned long max, const int log)
{
  /* usually a file contains around 60 samples, we might load 1 file in excess */
  data.index_to_time.reserve(max + 60);
  data.measures.reserve(max + 60);
  size_t index = 0;

  for (const std::pair<time_t, std::string>& it : files_) {
    load_file(data, it.second);
    ++index;
    /* forcefully quit loop if we reach limit */
    if (data.index_to_time.size() >= max)
      break;
  }

  /* cut samples to the exact number,
     we might have loaded a few more than requested */
  data.index_to_time.resize(max);
  data.measures.resize(max);
}


void DataLoader::load_file(Data& data, const std::string& file_name) const
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
  /* init with default values */
  for (uint8_t i = 0; i < 3; ++i)
    measures[i].fill(-2);
  powers.fill(-2);

  std::string line;
  std::time_t old_date;
  std::time_t date;

  if (std::getline(input_file, line)) {

    /* first iteration outside the loop to initialize old_date */
    std::size_t date_index = this->parse_line_date(line, old_date) + 1;
    this->parse_line_data(line, date_index, measures, powers);


    while(std::getline(input_file, line)) {
      date_index = this->parse_line_date(line, date) + 1;

      /* check if the current line refers to a new date or not */
      if (old_date != date) {
        /* dates in the input file should be sorted */
        assert(old_date < date);
        /* if new date, add it to the list */
        data.index_to_time.push_back(old_date);
        /* push collected data into the list */
        data.measures.push_back(std::make_pair(std::move(measures),
                                               std::move(powers)));

        /* re initialize local structures,
           fill with -2: used to signal missing values */
        measures = TankMeasures();
        for (uint8_t i = 0; i < 3; ++i)
          measures[i].fill(-2);

        powers = PowerMeasures();
        powers.fill(-2);

        /* update old date */
        old_date = date;
      }

      this->parse_line_data(line, date_index, measures, powers);
    }
  }
  input_file.close();

  /* add last measurements */
  data.index_to_time.push_back(date);
  data.measures.push_back(std::make_pair(std::move(measures),
                                         std::move(powers)));
}

std::size_t DataLoader::parse_line_date(const std::string& line,
                                        std::time_t& date) const
{
  /* parse date */
  const std::size_t date_index = line.find_first_of(';');
  std::istringstream date_str(line.substr(0, date_index));
  struct std::tm tm;
  std::memset(&tm, 0, sizeof(struct std::tm));
  date_str >> std::get_time(&tm, "%d/%m/%Y %H:%M:%S");
  date = mktime(&tm);

  return date_index;
}

void DataLoader::parse_line_data(const std::string& line, const std::size_t& date_index,
                                 TankMeasures&  measures,
                                 PowerMeasures& powers) const
{
   /* parse measure type and value */
    const std::size_t measure_type_index = line.find_first_of(';', date_index);
    const std::string measure_type_code =
      line.substr(date_index, measure_type_index - date_index);
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
