#include <math.h>
#include <vector>
#include "energymodel.h"

using namespace std;

double EnergyModel::get_P_takeoff(const double& parcel_weight)
{
    const double T = g * (getW_single_drone() + parcel_weight);
    return get_k1() *
        T *
        (v_takeoff / 2 +
            sqrt(pow(v_takeoff / 2, 2) + T / pow(get_k2(), 2))) // induced power for lifting
        +
        get_c2() * pow(T, 1.5); // profile power for takeoff
}

double EnergyModel::get_P_landing(const double& parcel_weight)
{
    const double T = g * (getW_single_drone() + parcel_weight);
    return get_k1() *
        T *
        (v_landing / 2 +
            sqrt(pow(v_landing / 2, 2) + T / pow(get_k2(), 2))) // induced power for lifting
        +
        get_c2() * pow(T, 1.5); // profile power for takeoff
}

double EnergyModel::get_P_flight(const double& w_parcel, const double& vh)
{
    const double T = g * (getW_single_drone() + w_parcel);
    double alpha = get_alpha(w_parcel, vh);
    double cos_a = cos(alpha * PI / 180);

    return (get_c1() + get_c2()) * pow(pow(T - get_c5() * pow(vh * cos_a, 2), 2)
        +
        pow(get_c4() * vh * vh, 2)
        , 0.75)
        +
        get_c4() * pow(vh, 3);
}

double EnergyModel::get_P_hover(const double& w_parcel)
{
    return (get_c1() + get_c2()) * pow((getW_single_drone() + w_parcel) * g, 1.5);
}

std::vector<double> EnergyModel::get_optimal_power(const double& w_parcel, const double& distance) // distance in m
{
    double vin_best = -1, vout_best = -1;
    double c_used_best = -1;
    double speed_0 = 4; // m/s
    double speed_1 = 28; // m/s
    double i_nominal = single_capacity; // at the beginning, I set this to 1

    double t_ver_takeoff = h_flight / v_takeoff; // second
    double t_ver_landing = h_flight / v_landing; // second
    double p_takeoff_in = get_P_takeoff(w_parcel);
    double p_landing_in = get_P_landing(w_parcel);
    double p_takeoff_out = get_P_takeoff(0); // return with no load
    double p_landing_out = get_P_landing(0); // return with no load

    double i_takeoff_in = p_takeoff_in / voltage;
    double i_landing_in = p_landing_in / voltage;
    double i_takeoff_out = p_takeoff_out / voltage;
    double i_landing_out = p_landing_out / voltage;

    double i_takeoff_in_eff = i_takeoff_in * pow(i_takeoff_in / i_nominal, peukert_constant - 1);
    double i_landing_in_eff = i_landing_in * pow(i_landing_in / i_nominal, peukert_constant - 1);
    double i_takeoff_out_eff = i_takeoff_out * pow(i_takeoff_out / i_nominal, peukert_constant - 1);
    double i_landing_out_eff = i_landing_out * pow(i_landing_out / i_nominal, peukert_constant - 1);

    double c_left = single_capacity
        - (i_takeoff_in_eff + i_takeoff_out_eff) * t_ver_takeoff / 3600 // second to h
        - (i_landing_in_eff + i_landing_out_eff) * t_ver_landing / 3600; // second to h
    if (!checkCapacity(c_left) || i_takeoff_in > max_discharge_current) // return null immediately, can't even lift hahaha
        return vector<double>{-1, -1, -1, -1};

    for (double vin = speed_0; vin <= speed_1; vin += 0.2) {
        double t_in = distance / vin; // second
        double p_in = get_P_flight(w_parcel, vin);
        double i_in = p_in / voltage;
        double i_in_eff = i_in * pow(i_in / i_nominal, peukert_constant - 1);
        double c_afterIn = c_left - i_in_eff * t_in / 3600;

        if (!checkCapacity(c_afterIn) || i_in > max_discharge_current)
            continue;
        for (double vout = speed_1; vout >= speed_0; vout -= 0.2) {
            double t_out = distance / vout; // second
            double p_out = get_P_flight(0, vout); // no load
            double i_out = p_out / voltage;
            double i_out_eff = i_out * pow(i_out / i_nominal, peukert_constant - 1);
            double c_afterOut = c_afterIn - i_out_eff * t_out / 3600;

            if (!checkCapacity(c_afterOut) || i_out > max_discharge_current)
                continue; // instead of continue
            // update vin pin vout pout
            bool faster = ((1 / vin_best + 1 / vout_best) >
                (1 / vin + 1 / vout));
            bool friendly = (1 / vin_best + 1 / vout_best == 1 / vin + 1 / vout)
                && (single_capacity - c_afterOut < c_used_best);
            if (faster || vin_best == -1 || friendly) {
                vin_best = vin;
                vout_best = vout;
                c_used_best = single_capacity - c_afterOut;
            }
        }
    }
    return vector<double>{vin_best, vout_best, c_used_best};
}

//double EnergyModel::get_endurance( const double &w_parcel, const double &distance)
//{

//}

double EnergyModel::get_alpha(const double& w_parcel, const double& vh)
{
    for (double alpha = 0; alpha < 90; alpha += 0.5) {
        double rad = alpha * PI / 180;
        double lhs = tan(rad) * ((getW_single_drone() + w_parcel) * g - get_c5() * pow(vh * cos(rad), 2));
        double rhs = get_c4() * vh * vh;
        double delta = abs(lhs - rhs);
        if (delta <= 5)
            return alpha;
    }

    //assert(false); // never get here
    for (double alpha = 0; alpha < 90; alpha += 0.1) {
        double rad = alpha * PI / 180;
        double lhs = tan(rad) * ((getW_single_drone() + w_parcel) * g - get_c5() * pow(vh * cos(rad), 2));
        double rhs = get_c4() * vh * vh;
        double delta = abs(lhs - rhs);
        if (delta <= 0.2)
            return alpha;
    }

    return -1;
}

double EnergyModel::get_k1()
{
    return 0.8554;
}

double EnergyModel::get_k2()
{
    return sqrt(2 * rho * getA());
}



double EnergyModel::get_c1() // k1/k2
{
    return get_k1() / get_k2();
}

double EnergyModel::get_c2() // k3^1.5 N c cd rho R4/8    ->>> 0.3177
/* N  the total number of blades in a single propeller
 * cd the drag coefficient of the blade
 * c the blade chord width
*/
{
    return 0.3177;
}

double EnergyModel::get_c3() // contribute very little -> assume c3 = 0 to simplify (Liu, 2017)
{
    return 0;
}

double EnergyModel::get_c4() // <<<<<<<<<<<<<<<<<<<<HERE
/*
 * Cd rho Aquad / 2    (kg/m)
 * Cd - drag coefficient of the vehicle body
 * Aquad is the cross-sectional area of the vehicle when
against wind
 * NOTICE: Cd and Aquad are complex function of the vehicle geometry and flight direction, making c4 is hard to identify.
 */
{
    return 0.0296;  //     just need consider single drone with shared mass
}

double EnergyModel::get_c5()
/*
 * N c cl rho R/4
 * cl lift coefficient
 */
{
    return 0.0279 / 0.15 * r; // assume only true for R = 0.15m
}

double EnergyModel::get_c6() // assume c6 = 0 to simplify (Liu, 2017)
{
    return 0;
}



EnergyModel::EnergyModel(double fweight, double bweight, int numPropellers, double R, double single_capacity, double voltage,
    double maxDischargeCurrent, double hflight, double vtakeoff, double vlanding, double rho, double peukert) : frame_weight(fweight), battery_weight(bweight), num_propellers(numPropellers), r(R), single_capacity(single_capacity), voltage(voltage), max_discharge_current(maxDischargeCurrent), h_flight(hflight), v_takeoff(vtakeoff), v_landing(vlanding), rho(rho), peukert_constant(peukert)

{

}