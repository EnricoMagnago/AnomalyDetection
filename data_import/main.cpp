#include <iostream>
#include "DataLoader.hpp"

int main(){
  DataLoader loader("../dataset/");
  Data data;
  std::cout << "loading files" << std::endl;
  loader.load_all(data, 400);

  std::cout << data.measures.size() << " ; " << data.index_to_time.size() << std::endl;
  std::cout << data.measures[0].first[1][0];
  std::cout << data.measures.back().first[1][0];
  for (const auto& it : data.anomaly_indexes) {
    const size_t index = it.first;
    const std::vector<Anomaly>& anomalies = it.second;
    std::cout << "\n" << time_to_string(data, index) << ":\n";
    for (const Anomaly& anomaly : anomalies) {
      std::cout << "\t" << anomaly_to_string(anomaly) << "\n";
    }
  }
}
