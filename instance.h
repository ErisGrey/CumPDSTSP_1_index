#pragma once
#ifndef INSTANCE_H
#define INSTANCE_H

#include <vector>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include "energymodel.h"

using namespace std;


class Instance {
public:
    string instanceName;
    int num_nodes;
    int num_drones;
    int truck_speed;
    double v_truck;
    double coupletime; // around 1m
    vector <vector<int>> distance;
    vector <int> drone_speed;
    vector <int> truck_time;
    vector<float> x;
    vector<float> y;
    unordered_map<int, vector<int>> c_drone_considered;
    vector<double> polar;
    unordered_set<int> freeCustomer;
    unordered_set<int> truckonlyCustomer;


    EnergyModel* energyModel;

    int max_combination;
    double max_weight_allowed;


    vector<double> weight;
    vector<vector<double>> dist_drone, dist_truck;
    vector<vector<double>> time_truck;

    vector<vector<int>> adjList;

    Instance(string inputFile, string paramFile);
    ~Instance();
    void export_stat();

    int getNum_nodes() const;
    int getNum_drones() const;
    double getWeight(int i) const;
    double tdrone(const int& customer) const;
    double ttruck(int i, int j);
    double dtruck(int i, int j);
    double serviceTime_drone(const int& customer) const;
    bool isTruckonly(int customer);



private:
    //    double coupletime;
    double frame_weight;
    int num_propellers;

    double rotor_blade_radius;
    double peukert_constant;
    double h_flight; // = 100; // (m) < 400 feet
    double v_takeoff;
    double v_landing;
    double air_density_rho;

    double battery_weight;
    double battery_capacity;
    double max_discharge_current;
    double voltage;

    void read_input(const string& inputFile);
    void read_otherParams(const string& paramsFile);
    void initialize();
    template <class Container>
    void split(const std::string& str, Container& cont, char delim);
};

#endif
