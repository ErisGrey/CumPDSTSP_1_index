#ifndef ENERGYMODEL_H
#define ENERGYMODEL_H
#define PI 3.14159
#define MAX_ALPHA 45.0
#include <vector>
using namespace std;


class EnergyModel
{
public:
    const double frame_weight;
    const double battery_weight;
    const int num_propellers; // default 4 for quadcopter
    const double r; // (m)
    const double single_capacity; // Ah
    const double voltage;
    const double max_discharge_current; // 40
    vector<double> flightTime;


    const double g = 9.81; // (m2/s)
    const double h_flight; // (m)
    const double v_takeoff; // (m/s) vertical velocity during take off
    const double v_landing; // (m/s) vertical velocity during landing
    const double rho; // (kg/m3) density of air
    const double peukert_constant;

    EnergyModel(double f_weight, double b_weight, int num_propellers, double r, double single_capacity, double voltage,
        double max_discharge_current, double hflight, double vtakeoff, double vlanding, double rho, double peukert);

    double get_P_takeoff(const double& parcel_weight);
    double get_P_landing(const double& parcel_weight);
    double get_P_flight(const double& w_parcel, const double& v_h);
    double get_P_hover(const double& w_parcel);
    std::vector<double> get_optimal_power(const double& w_parcel, const double& distance);

    double get_endurance(const double& w_parcel, const double& distance); // based on Peukert effect

    double get_alpha(const double& w_parcel, const double& vh);
    double get_k1();
    double get_k2();

    double get_c1();
    double get_c2();
    double get_c3();
    double get_c4();
    double get_c5();
    double get_c6();

    void show_discharge_curve();
    double getW_single_drone() const { return frame_weight + battery_weight; }
    double getA() const { return num_propellers * PI * r * r; }
    bool checkCapacity(const double& c) const {
        return c >= 0.1 * single_capacity;
    }
};

#endif // ENERGYMODEL_H