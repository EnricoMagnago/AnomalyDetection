%module DataTypes

%include "std_vector.i"
%include "std_array.i"
%include "std_pair.i"
%include "pointer.i"
%include <std_string.i>

typedef long time_t;
typedef long unsigned int size_t;
%{
    #define SWIG_FILE_WITH_INIT
    #include "../DataTypes.hpp"
%}


%template() std::array<double, 6>;
%template() std::array<std::array<double, 6>, 3>;
%template() std::pair<std::array<std::array<double, 6>, 3>, std::array<double, 6> >;
%template(Measures) std::vector<std::pair<std::array<std::array<double, 6>, 3>, std::array<double, 6> > >;
%template(Times) std::vector<time_t>;

%include "../DataTypes.hpp"

%pointer_class(Anomaly, Anomaly_ptr);
%template(IntVector) std::vector<int>;
%template(AnomaliesList) std::vector<Anomaly>;
%template(AnomaliesIndexes) std::pair<size_t, std::vector<Anomaly> >;
%template(AnomaliesMap) std::vector<std::pair<size_t, std::vector<Anomaly> > >;
