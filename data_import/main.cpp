#include <iostream>
#include "DataLoader.hpp"

int main(){
  DataLoader loader("../dataset/");
  Data data;
  std::cout << "loading files" << std::endl;
  loader.load_subset(data, 40);

  std::cout << data.measures[0].first[1][0];
}
