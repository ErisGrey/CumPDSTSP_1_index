#ifndef BENDERSOLVER_H
#define BENDERSOLVER_H
#include "instance.h"
#include <ilcplex/ilocplex.h>
#include <unordered_map>
#include <unordered_set>

#include <chrono>
ILOSTLBEGIN

typedef IloArray<IloArray<IloNumVarArray>> NumVar3D;
typedef IloArray<IloArray<IloNumArray>> Num3D;
typedef IloArray<IloNumVarArray> NumVar2D;
typedef IloArray<IloNumArray> Num2D;

typedef unordered_map<int, int> MapSol2D;
typedef vector<unordered_map<int, int>> MapSol3D;
typedef vector<tuple<int, int>> Sol2D;
typedef vector<tuple<int, int, int>> Sol3D;


using namespace std;


class Solver {
public:
    Instance* instance;
    NumVar2D x;
    NumVar2D z;
    IloNumVarArray z1;
    IloNumVarArray a;
    IloNumVarArray u;
    IloNumVarArray k;
    IloNumVarArray v;
    IloNumVar alpha;


    IloObjective subObj;
    IloEnv masterEnv;
    IloCplex masterCplex;
    IloModel masterModel;
    double UB;
    double UB_tsp;
    int numNode;
    int numUAV;
    vector<int> N;
    vector<int> C;
    unordered_set<int> truckOnly;
    unordered_set<int> freeCustomers;
    unordered_set<int> freeCustomers0;



    Solver(Instance*, string input, string output, double time_limit, double mem_limit);
    void Solve();
    bool isTruckOnly(int i);
    double getTime_truck(int i, int j) {
        return instance->time_truck[i][j];
    }

    ~Solver();

    bool isFree(int i);
private:
    chrono::time_point<std::chrono::high_resolution_clock> startTime;
    double gap;
    double runtime;
    string status;
    string inputFile;
    string outputFile;
    double time_limit;
    double mem_limit;
    void createModel();
    void dispay_solution();
    void write_output();



};

#endif // BENDERSOLVER_H