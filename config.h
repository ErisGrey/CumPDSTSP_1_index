#ifndef CONFIG_H
#define CONFIG_H
#include <iostream>
#include <string>
#include <chrono>
#include <random>
using namespace std;

class Config
{
public:

    string input;
    string param_input;
    int max_iter = 10000;
    double cooling_rate = 0.9995;
    double threshold = 0.15;
    double max_removal_rate = 0.3;
    int seed;

    string output = "output.txt";
    bool found_output = false;

    double time_limit = 300;
    bool found_time_limit = false;

    double mem_limit = 28000;
    bool found_mem_limit = false;

    Config(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            string key = argv[i];

            if (key == "-i") {
                string value = argv[++i];
                input = value;
                found_input = true;
            }

            else if (key == "-pi") {
                string value = argv[++i];
                param_input = value;
                found_param_input = true;
            }


            else {
                cerr << "Invalid argument !!!\n";
                exit(0);
            }
        }

        if (!found_input) {
            cerr << "ERROR: Input is required !!!\n";
            exit(0);
        }
        else if (!found_param_input) {
            cerr << "ERROR: param_input is required !!!\n";
            exit(0);
        }
    }
private:
    bool found_input = false;
    bool found_param_input = false;
    bool found_max_iter = false;
    bool found_cooling_rate = false;
    bool found_threshold = false;
    bool found_max_removal_rate = false;
    bool found_seed = false;
};

#endif // CONFIG_H