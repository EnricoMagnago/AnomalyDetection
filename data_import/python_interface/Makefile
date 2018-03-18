swig:
	swig -python -c++ -v DataTypes.swig
	swig -python -c++ -v DataLoader.swig
	g++ -c -fPIC -Wall -DNDEBUG -std=c++17 DataTypes_wrap.cxx DataLoader_wrap.cxx ../DataLoader.cpp -I/usr/include/python3.6 -lpython3.6 -lstdc++fs
	g++ -shared -fPIC -ldl -Wall -DNDEBUG -std=c++17 DataTypes_wrap.o -o _DataTypes.so -lstdc++fs
	g++ -shared -fPIC -ldl -Wall -DNDEBUG -std=c++17 DataTypes_wrap.o DataLoader_wrap.o DataLoader.o -o _DataLoader.so -lstdc++fs

clean:
	rm -rf __pycache__ *.o *.so *.pyc
	rm -rf DataTypes.py DataTypes_wrap.* DataLoader.py DataLoader_wrap.* DataLoader.pyc
