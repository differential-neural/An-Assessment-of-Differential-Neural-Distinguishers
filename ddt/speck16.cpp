#include<iostream>
#include<cstdlib>
#include<random>
#include<vector>
#include<cstring>
#include<tuple>
#include<unordered_map>
#include<map>
#include<omp.h>
#include<math.h>
#include<stdlib.h>
#include<vector>
#include<fstream>
#include<unordered_set>
#include<algorithm>
#include<string>
#include<sstream>

using namespace std;

/* Based on the code by Gohr */

#define WORD_SIZE 8
#define ALPHA 5
#define BETA 2
#define MASK_VAL 0xff
#define MAX_ROUNDS 50

uint16_t rol(uint16_t a, uint16_t b) {
    uint16_t n = ((a << b) & MASK_VAL) | (a >> (WORD_SIZE - b));
    return (n);
}

uint16_t ror(uint16_t a, uint16_t b) {
    uint16_t n = (a >> b) | (MASK_VAL & (a << (WORD_SIZE - b)));
    return (n);
}

//the following function calculates the probability of a xor-differential transition of one round of Speck16 according to Lipmaa-Moriai
double diff_prob(uint16_t in, uint16_t out) {
    //first, transform the output difference to what it looked like before the modular addition
    //transform also the input difference accordingly
    uint16_t in0 = in >> 8;
    uint16_t in1 = in & 0xff;
    uint16_t out0 = out >> 8;
    uint16_t out1 = out & 0xff;
    in0 = ror(in0, ALPHA);
    out1 = out1 ^ out0;
    out1 = ror(out1, BETA);
    if (out1 != in1) return (0);
    uint16_t x = in0 ^ in1 ^ out0;
    uint16_t y = (in0 ^ out0) | (in1 ^ out0);
    x = (x ^ (x << 1)) & MASK_VAL;
    y = (y << 1) & MASK_VAL;
    if ((x & y) != x) return (0);
    int weight = __builtin_popcount(y);
    double res = pow(2, -weight);
    return (res);
}

void calc_ddt_update(vector<double> &ddt, vector<double> &tmp) {
    uint32_t small = 1L << 16;
    vector<double> sums(1L << WORD_SIZE);
    for (uint16_t i = 1; i != 0; i++)
        sums[i >> WORD_SIZE] += ddt[i];
    #pragma omp parallel for
    for (uint32_t i = 1; i < small; i++) {
        uint16_t out = i;
        uint16_t in1 = ror((out >> WORD_SIZE) ^ (out & MASK_VAL), BETA);
        double p = 0;
        uint16_t ind = in1 << WORD_SIZE;
        if (sums[in1] != 0)
            for (uint16_t in2 = 0; in2 <= MASK_VAL; in2++) {
                uint16_t index = ind ^ in2;
                uint16_t inp = (in2 << WORD_SIZE) ^ in1;
                p += ddt[index] * diff_prob(inp, out);
            }
        uint16_t ind_out = (out >> WORD_SIZE) ^ ((out & MASK_VAL) << WORD_SIZE);
        tmp[ind_out] = p;
    }
    #pragma omp parallel for
    for (uint32_t out = 1; out < small; out++)
        ddt[out] = tmp[out];
}

void calc_ddt_for_in_diff(uint16_t in_diff, int num_rounds) {
    uint32_t num_diffs = 1L << 16;
    vector<double> ddt(num_diffs);
    vector<double> tmp(num_diffs);
    uint16_t ind = (in_diff >> WORD_SIZE) ^ ((in_diff & MASK_VAL) << WORD_SIZE);
    ddt[ind] = 1.0;
    for (int i = 0; i < num_rounds; i++) {
        calc_ddt_update(ddt, tmp);
        stringstream del;
        del << hex << in_diff;
        string delta = del.str();
        string rounds = to_string(i);
        string filename = "ddt_" + delta + "_" + rounds + "rounds.bin";
        ofstream fout(filename, ios::out | ios::binary);
        fout.write((char *) &ddt[0], ddt.size() * sizeof(double));
        fout.close();
    }
}

void calc_ddt(int num_rounds) {
    uint16_t max_iter = 0xffff;
    for (uint16_t diff = 0x0001; diff <= max_iter; diff++)
        calc_ddt_for_in_diff(diff, num_rounds);
}

int main() {
    calc_ddt(6);
    cout << "Done." << endl;
    return (0);
}

