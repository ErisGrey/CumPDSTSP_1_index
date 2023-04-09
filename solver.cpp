#include "solver.h"
#include "instance.h"
#include <tuple>
ILOSTLBEGIN
#define pos(u,v) min(u,v)][max(u,v)

Solver::Solver(Instance* instance, string input, string output, double time_limit, double mem_limit)
	:instance(instance), inputFile(input), outputFile(output), time_limit(time_limit), mem_limit(mem_limit) {
	cerr << "-- Solving \n";
	startTime = chrono::high_resolution_clock::now();
	outputFile = instance->instanceName;
	gap = 1e5;
	status = "-";

	/*  SET -------------------------------- */
	for (int i = 0; i < instance->num_nodes; ++i)
		N.push_back(i);
	for (int i = 1; i < instance->num_nodes; ++i)
		C.push_back(i);

	// --
	numNode = instance->num_nodes;
	numUAV = instance->num_drones;
	UB = 1e5;
	UB_tsp = 1e5;

	truckOnly = instance->truckonlyCustomer;
	freeCustomers = instance->freeCustomer;
	freeCustomers0 = instance->freeCustomer;
	freeCustomers0.insert(0);
}

Solver::~Solver() {
	//    cerr << "Runtime = " << (double)(clock() - startTime) / CLOCKS_PER_SEC << "s\n";
}

void Solver::Solve() {
	try {
		createModel();

		masterCplex.exportModel("lpex.lp");
		masterCplex.setParam(IloCplex::Param::Parallel, 1);
		masterCplex.setParam(IloCplex::Param::Threads, 16);
		masterCplex.setParam(IloCplex::TiLim, time_limit);
		masterCplex.setParam(IloCplex::TreLim, mem_limit);
		masterCplex.setParam(IloCplex::Param::MIP::Strategy::RINSHeur, 10);
		masterCplex.setParam(IloCplex::EpGap, 0.01);
		masterCplex.solve();
		if (masterCplex.getStatus() == IloAlgorithm::Infeasible) {
			cout << UB << endl;
			cout << "Infeasible" << endl;
		}
		else if (masterCplex.getStatus() == IloAlgorithm::Unbounded) {
			cout << "Unbounded" << endl;
		}
		else if (masterCplex.getStatus() == IloAlgorithm::Unknown) {
			cout << "Unknown" << endl;
		}
		else {
			cout << "DONE ..." << endl;
			cout << masterCplex.getObjValue() << endl;
			dispay_solution();
		}


	}
	catch (IloException& e) {
		cerr << "Conver exception caught: " << e << endl;
	}
	catch (...) {
		cerr << "Unknown exception caught" << endl;
	}


	auto endTime = chrono::high_resolution_clock::now();
	runtime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	runtime = runtime / 1000;


	write_output();
	cout << "Environment cleared \n";
	//        workerCplex.end();
	masterCplex.end();

	//        workerEnv.end();
	masterEnv.end();
}

void Solver::createModel() {
	masterModel = IloModel(masterEnv);
	masterCplex = IloCplex(masterEnv);

	x = NumVar2D(masterEnv, numNode); // x_ij
	//y = NumVar2D(masterEnv, numNode); // y_ij
	z = NumVar2D(masterEnv, numNode); // z_ij
	//u = IloNumVarArray(masterEnv, numNode);
	v = IloNumVarArray(masterEnv, numNode); // v_i
	a = IloNumVarArray(masterEnv, numNode);
	z1 = IloNumVarArray(masterEnv, numNode);
	stringstream name;
	// x_ij
	for (int i = 0; i < numNode; i++)
	{
		x[i] = IloNumVarArray(masterEnv, numNode);
		for (int j = 0; j < numNode; j++) {
			if (j == i) continue;
			name << "x." << i << "." << j;
			x[i][j] = IloNumVar(masterEnv, 0, 1, ILOINT, name.str().c_str());
			name.str("");
		}
	}
	// v_i
	for (int i : N)
	{
		name << "v." << i;
		v[i] = IloNumVar(masterEnv, 0, IloInfinity, ILOFLOAT, name.str().c_str());
		name.str("");
	}

	//u_i
	/*for (int i : C)
	{
		name << "u." << i;
		u[i] = IloNumVar(masterEnv, 1, numNode-1, ILOINT, name.str().c_str());
		name.str("");
	}*/

	//a_i
	for (int i : freeCustomers)
	{
		name << "a." << i;
		a[i] = IloNumVar(masterEnv, 0, IloInfinity, ILOINT, name.str().c_str());
		name.str("");
	}

	//z1_i
	for (int i : C)
	{
		name << "z1." << i;
		z1[i] = IloNumVar(masterEnv, 0, 1, ILOINT, name.str().c_str());
		name.str("");

	}

	// OBJ FUNCTION
	IloExpr exprSolution(masterEnv);
	for (int i : C)
	{
		exprSolution += v[i];

	}

	for (int i : freeCustomers)
	{
		exprSolution += a[i] * instance->energyModel->flightTime[i];
	}

	masterModel.add(IloMinimize(masterEnv, exprSolution));

	// CONSTRAINT -------------------------------------

	for (int j : truckOnly) {
		masterModel.add(z1[j] == 1);
	}
	// only one truck out of depot
	{
		IloExpr sum(masterEnv);
		for (int j : C)
			sum += x[0][j];
		masterModel.add(sum == 1);
		sum.end();
	}
	// 1. either truck or drone
	for (int j : C) {
		IloExpr sumz(masterEnv);
		for (int i : N)
		{
			if (i == j) continue;
			sumz += x[i][j];
		}

		masterModel.add(sumz == z1[j]);
		sumz.end();
	}
	//2. only truck
	for (int j : truckOnly) {
		IloExpr sumt1(masterEnv);
		for (int i : N) {
			if (i == j) continue;
			sumt1 += x[i][j];
		}
		masterModel.add(sumt1 == 1);
		sumt1.end();
	}
	//3. Vao = ra
	for (int i : N)
	{
		IloExpr sum(masterEnv);
		for (int j : N) {
			if (i == j) continue;
			sum += x[i][j] - x[j][i];
		}
		masterModel.add(sum == 0);
		sum.end();
	}

	
	int BIGM = 999;
	masterModel.add(v[0] == 0);
	/*for (int i : C)
	{
		masterModel.add(v[i] - instance->time_truck[0][i] >= -BIGM * (1 - x[0][i]));
	}*/

	//subtour
	/*for (int i : C)
	{
		for (int j : C)
		{
			if (i == j) continue;
			masterModel.add(u[i] - u[j] + numNode * x[i][j] <= numNode - 1);
		}
	}*/

	// linking v and x
	{
		for (int i : N) {
			for (int j : C) {
				if (i == j) continue;
				IloExpr sum(masterEnv);
				sum += v[j] - v[i] - instance->time_truck[i][j] * x[i][j];
				masterModel.add(sum >= -(1 - x[i][j]) * BIGM);
			}

		}
	}

	/*for (int i : C) {
		masterModel.add(v[i] <= BIGM * z1[i]);
		masterModel.add(v[i] >= z1[i]);
	}*/




	for (int j : freeCustomers)
	{

		//masterModel.add(1 - z1[j] <= a[j]);
		masterModel.add(a[j] <= (1 + j * 1.0 / numUAV) * (1 - z1[j]));
	}

	for (int j : freeCustomers)
	{
		IloExpr condij(masterEnv);
		for (int i = 1; i <= j; i++)
		{
			condij += z1[i];
		}
		masterModel.add((j - condij) * 1.0 / numUAV - (j * 1.0 / numUAV) * z1[j] <= a[j]);

		condij.end();
	}

	masterCplex.extract(masterModel); // <<<< IMPORTANT
	cout << "Done create init MasterProblem\n";

}

void Solver::dispay_solution()
{


	unordered_map<int, int> edges;
	////    unordered_map<int,double> start;

	//    cout << "-----------x_ij-----------" << endl;
	//    for (int i : Scenes) {
	//        for(int j : Scenes){
	//            if (i == j) continue;
	//            if (cplex.getValue(y[i][j]) > 0.1){
	//                edges[i] = j;
	////                start[j] = cplex.getValue(t[i][j]);
	////                if (i == numSin)
	////                    cout << "x_" << "N" << "_" << j  << " ";
	////                else if (j == numSin)
	////                    cout << "x_" << i << "_" << "N"  << " ";
	////                else
	////                    cout << "x_" << i << "_" << j  << " ";
	//            }
	//        }
	////        cout << endl;
	//    }

	//    vector<vector<int>> tours;
	//    while (!edges.empty()) {
	//        int start = edges.begin()->first;
	//        int current = -1;
	//        vector<int> tour = {start};
	//        current = start;
	//        do {
	//            int next = edges[current];
	//            tour.push_back(next);
	//            edges.erase(current);
	//            current = next;
	//        } while (start != current);
	//        tours.push_back(tour);
	//    }

	//    for (auto tour : tours){
	//        cout << "tour: ";
	//        Utils::print_vec(tour);
	//    }








	cout << "X value:\n  ";
	for (int i = 0; i < numNode; ++i) {
		for (int j = 0; j < numNode; ++j) {
			if (i == j) continue;
			if (masterCplex.getValue(x[i][j]) > 0.0001) {
				cout << i << " " << j << "," << endl;
			}
		}

	}
	cout << endl;
	cout.precision(10);

	cout << "CumValue:" << endl;
	for (int i : C)
	{
		cout << "v[" << i << "]= " << masterCplex.getValue(v[i]) << endl;
	}
	cout << endl;

		cout << "Drone value:  " << endl;
		for (int i : C)
		{
			if (masterCplex.getValue(z1[i]) == 0)
			{
				cout << i << " " << masterCplex.getValue(a[i]) << ",";
				cout << "drone_time[" << i << "] = " << instance->energyModel->flightTime[i] << endl;
			}
		}


	cout << endl;
	cout << masterCplex.getObjValue() << endl;

	/*cout << "Y value:  " << endl;
	for (int i : freeCustomers0) {
		for (int j : freeCustomers0) {
			if (i == j) continue;
			if (masterCplex.getValue(y[i][j]) == 1 && j != 0) {
				cout << i << " " << j << " " << masterCplex.getValue(y[i][j]) << ",";
				cout << "drone_time[" << j << "] = " << instance->energyModel->flightTime[j] << endl;
			}
		}
	}*/
	cout << endl;
	cout << masterCplex.getObjValue() << endl;

}

void Solver::write_output()
{

	std::ofstream ocsv;
	ocsv.open("CPLEX_output_summary.csv", std::ios_base::app);
	ocsv << instance->instanceName << ","
		<<  masterCplex.getObjValue() << ","
		<< masterCplex.getBestObjValue() << ","
		<< fixed << std::setprecision(2)
		<< masterCplex.getMIPRelativeGap() * 100 << ","
		<< masterCplex.getStatus() << ","
		<< runtime << "\n"
		<< std::flush;
	ocsv.close();
}