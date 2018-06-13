#ifndef DATA_TYPES_HPP__
#define DATA_TYPES_HPP__

#include <vector>
#include <array>

/* use to access TankMeasures */
enum TankType {TANK1 = 0, TANK2, TANK3};

/* use to access SensorMeasures */
enum SensorType {OXYGEN = 0, NITROGEN, SST, AMMONIA, VALVE, FLOW};

typedef std::array<double, 6> SensorMeasures;
typedef std::array<SensorMeasures, 3> TankMeasures;
typedef std::array<double, 6> PowerMeasures;


/* structure used to represent an anomaly */
struct Anomaly
{
  time_t begin;
  time_t end;
  std::string description;
  std::vector<TankType> tanks;
};

/* structure used to represent data,
   the indexes correspond: data at measures[i] has been collected
       in date index_to_time[i];
*/
struct Data
{
  std::vector<time_t> index_to_time;
  std::vector<std::pair<TankMeasures, PowerMeasures> > measures;
  /* each element of the vector is a pair (index, anomalies) where:
     index is the measure index and
     anomalies is the list of anomalies to which the measure at position `index` belogs.
   */
  std::vector<std::pair<size_t, std::vector<const Anomaly*> > > anomaly_indexes;
};


#endif /* DATA_TYPES_HPP__ */
