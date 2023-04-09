#include "instance.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <iostream>
#include <math.h>
Instance::Instance(string instanceFile, string paramsFile)
{
	read_input(instanceFile);
	read_otherParams(paramsFile);
	initialize();
	// ENERGY MODEL
	energyModel = new EnergyModel(frame_weight, battery_weight, num_propellers, rotor_blade_radius, battery_capacity, voltage, max_discharge_current, h_flight, v_takeoff, v_landing, air_density_rho, peukert_constant);

	energyModel->flightTime.resize(num_nodes, -1);
	energyModel->flightTime[0] = 0;
	cout.precision(10);
	for (int customer = 1; customer < getNum_nodes(); ++customer) {
		double dist = dist_drone[0][customer] * 1000; // km to m

		if (getWeight(customer) + energyModel->getW_single_drone() > max_weight_allowed)
			continue;
		vector<double> optimal_setting = energyModel->get_optimal_power(getWeight(customer), dist);
		if (optimal_setting[0] != -1) {
			energyModel->flightTime[customer] =
				dist / optimal_setting[0] + // inbound trip in second
				dist / optimal_setting[1] + // outbound trip in second
				2 * energyModel->h_flight / energyModel->v_takeoff + // 2 times takeoff in second
				2 * energyModel->h_flight / energyModel->v_landing  // 2 time landing in second
				;
			energyModel->flightTime[customer] = round(energyModel->flightTime[customer] / 60); // /60 to minute
		}

	}




	for (int i = 1; i < num_nodes; ++i) {
		if (tdrone(i) < 0)
		{
			truckonlyCustomer.insert(i);
			cout << i << endl;
		}
		else
		{
			freeCustomer.insert(i);
			cout << i << " +" << endl;
		}

	}

	return;
}

Instance::~Instance()
{
	delete energyModel;
}

void Instance::read_input(const string& inputFile)
{
	instanceName = inputFile.substr(53,7);
	ifstream myFile(inputFile);
	if (!myFile.is_open())
	{
		// End program immediately
		cout << "Unable to open instance file \n";
		exit(0);
	}

	string line;
	vector< string > numbers;

	getline(myFile, line);

	split(line, numbers, ',');

	num_drones = stoi(numbers[1]);


	getline(myFile, line);
	split(line, numbers, ',');
	v_truck = stod(numbers[1]);

	getline(myFile, line);
	split(line, numbers, ',');
	battery_weight = stod(numbers[1]);

	getline(myFile, line);
	split(line, numbers, ',');
	battery_capacity = stod(numbers[1]);

	getline(myFile, line);
	split(line, numbers, ',');
	max_discharge_current = stod(numbers[1]);

	getline(myFile, line);
	split(line, numbers, ',');
	voltage = stod(numbers[1]);

	//    getline(myFile, line);
	//    split(line, numbers, ',');
	//    endurance_drone = stoi(numbers[1]);

	getline(myFile, line);

	num_nodes = 0;
	while (getline(myFile, line))
	{
		split(line, numbers, '\t');
		x.push_back(stod(numbers[1]));
		y.push_back(stod(numbers[2]));
		float w = stod(numbers[3]);
		weight.push_back(w);
		//        if (w > cap_drone){
		//            overweight.push_back(1);
		//        } else {
		//            overweight.push_back(0);
		//        }
		num_nodes++;
	}
	weight[0] = 0;

	for (int i = 0; i < num_nodes; ++i) {
		weight[i] = std::floor(weight[i] * 1000) / 1000; /*Utils::round(weight[i], 3);*/
		//weight[i] = Utils::round(weight[i], 2);
	}
	myFile.close();
}

void Instance::read_otherParams(const string& paramsFile)
{
	ifstream myFile(paramsFile);
	if (!myFile.is_open()) {
		cout << "Unable to open params file \n";
		exit(0);
	}

	string line;
	string token;
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		coupletime = stod(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		max_combination = stoi(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		max_weight_allowed = stod(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		frame_weight = stod(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		num_propellers = stoi(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		rotor_blade_radius = stod(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		peukert_constant = stod(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		h_flight = stod(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		v_takeoff = stod(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		v_landing = stod(token);
	}
	getline(myFile, line);
	{
		stringstream ss(line);
		getline(ss, token, ',');
		getline(ss, token, ',');
		air_density_rho = stod(token);
	}

}

void Instance::initialize()
{
	// Compute distance and time matrix
	time_truck.resize(num_nodes);
	dist_drone.resize(num_nodes);
	dist_truck.resize(num_nodes);
	for (int i = 0; i < num_nodes; i++)
	{
		time_truck[i].resize(num_nodes);
		dist_drone[i].resize(num_nodes);
		dist_truck[i].resize(num_nodes);
	}

	for (int i = 0; i < num_nodes; i++) {
		for (int j = 0; j < num_nodes; j++) {
			float euc_d = pow(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2), 0.5);
			float man_d = abs(x[i] - x[j]) + abs(y[i] - y[j]);

			dist_drone[i][j] = euc_d;
			dist_truck[i][j] = man_d;

			//            time_drone[i][j] = round(euc_d/v_drone);
			time_truck[i][j] = man_d / v_truck * 60.0;

			time_truck[i][j] = ceil(time_truck[i][j]);
		}
	}

	/*for (int i = 0; i < num_nodes; i++) {
		for (int j = 0; j < num_nodes; j++) {

			cout << "time_truck[" << i << "][" << j << "] = " << time_truck[i][j] << endl;
		}
	}*/

	// polar
	polar.push_back(0);
	for (int i = 1; i < num_nodes; ++i) {
		double dx = x[i] - x[0];
		double dy = y[i] - y[0];
		double a = atan(dy / dx) * 180.0 / 3.14159265;
		if (dx >= 0 && dy >= 0) {

		}
		else if (dx < 0 && dy >= 0) {
			a += 180;
		}
		else if (dx < 0 && dy < 0) {
			a += 180;
		}
		else if (dx >= 0 && dy < 0) {
			a += 360;
		}

		polar.push_back(a);
		//        cout << i <<" "<< a << "\n";
	}

	int n = num_nodes;
	adjList.resize(num_nodes);
	vector<int> customers;
	for (int i = 0; i < n; i++) {
		if (i != 0)
			customers.push_back(i);
		adjList[i].reserve(n);
	}


}

double Instance::tdrone(const int& customer) const
{
	return energyModel->flightTime[customer];
}

double Instance::serviceTime_drone(const int& customer) const
{

	return tdrone(customer);

}

double Instance::ttruck(int i, int j)
{
	return time_truck[i][j];
}

double Instance::dtruck(int i, int j)
{
	return dist_truck[i][j];
}

int Instance::getNum_nodes() const
{
	return num_nodes;
}

int Instance::getNum_drones() const
{
	return num_drones;
}

double Instance::getWeight(int i) const
{
	return weight[i];
}

bool Instance::isTruckonly(int customer)
{
	if (getWeight(customer) > max_weight_allowed)
		return true;
	else
		return false;
}

void Instance::export_stat()
{
	int inreach = 0;
	int overw = 0;
	int overw_inreach = 0;
	int el = 0;

	string s = "";
	s += instanceName + ",";
	s += to_string(num_nodes - 1) + ",";
	s += to_string(inreach) + ",";
	s += to_string(overw) + ",";
	s += to_string(overw_inreach) + ",";
	s += to_string(el);

	std::ofstream outCSV;
	outCSV.open("instance_stat_summary.csv", std::ios_base::app);
	outCSV << s << "\n" << std::flush;
	outCSV.close();
}

template<class Container>
void Instance::split(const std::string& str, Container& cont, char delim)
{
	cont.clear();
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, delim)) {
		if (token != "")
			cont.push_back(token);
	}
}