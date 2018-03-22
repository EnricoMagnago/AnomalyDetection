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
}
