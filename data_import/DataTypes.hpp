#ifndef DATA_TYPES_HPP__
#define DATA_TYPES_HPP__

#include <vector>
#include <array>

/* use to access TankMeasures */
enum TankType {TANK1 = 0, TANK2, TANK3};

/* use to access SensorMeasures */
enum SensorType {OXIGEN = 0, NITROGEN, SST, AMMONIA, VALVE, FLOW};

typedef std::array<double, 6> SensorMeasures;
typedef std::array<SensorMeasures, 3> TankMeasures;
typedef std::array<double, 6> PowerMeasures;


/* structure used to represent data,
   the indexes correspond: data at measures[i] has been collected
       in date index_to_time[i];
*/
struct Data
{
  std::vector<time_t> index_to_time;
  std::vector<std::pair<TankMeasures, PowerMeasures>> measures;
};


#endif /* DATA_TYPES_HPP__ */
