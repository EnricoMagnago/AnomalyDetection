%module DataTypes

%include "std_vector.i"
%include "std_array.i"
%include "std_pair.i"

typedef long time_t;
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
