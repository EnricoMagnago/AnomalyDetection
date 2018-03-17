#include <iostream>
#include "DataLoader.hpp"

int main(){
  DataLoader loader("../dataset/");
  Data data;
  std::cout << "loading files" << std::endl;
  loader.load_data(data, 80);
}
