#include <iostream>
#include "config.h"
#include "instance.h"
#include "solver.h"
int main(int argc, char* argv[])
{
    // FinalPDSTSP -i C:\Users\admin\Downloads\superDrone3\instances\small\15-r-e.csv -pi C:\Users\admin\Downloads\superDrone\other-params.txt
    // FinalPDSTSP -i C:\Users\admin\Downloads\superDrone3\instances\small\15-r-e1.csv -pi C:\Users\admin\Downloads\superDrone\other-params.txt
    Config config(argc, argv);

    Instance instance(config.input, config.param_input);

    Solver solver(&instance, instance.instanceName, instance.instanceName, config.time_limit, config.mem_limit);
    solver.Solve();

    return 0;
}