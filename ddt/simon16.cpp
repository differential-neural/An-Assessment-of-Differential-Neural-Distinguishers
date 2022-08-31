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

#define WORD_SIZE 8
#define ALPHA 1
#define BETA 2
#define GAMMA 3
#define MASK_VAL 0xff
#define MAX_ROUNDS 50

//Rotations mod word size
static const int8_t TWO_ALPHA_MINUS_BETA = (2 * ALPHA - BETA) % WORD_SIZE;
static const int8_t ALPHA_MINUS_BETA = ALPHA - BETA % WORD_SIZE;

uint16_t rol(uint16_t a, uint16_t b) {
    uint16_t n = ((a << b) & MASK_VAL) | (a >> (WORD_SIZE - b));
    return (n);
}

uint16_t ror(uint16_t a, uint16_t b) {
    uint16_t n = (a >> b) | (MASK_VAL & (a << (WORD_SIZE - b)));
    return (n);
}

//The following function calculates the probability of Simon like functions according to
//Koelbl, Leander and Tiessen ("Observations on the SIMON block cipher family", Theorem 3)
double diff_prob(uint16_t in, uint16_t out) {
    uint16_t varibits = rol(in, ALPHA) | rol(in, BETA);
    uint16_t doublebits = rol(in, BETA) & ~(rol(in, ALPHA)) & rol(in, TWO_ALPHA_MINUS_BETA);
    uint16_t gamma = out ^ rol(in, GAMMA);

    int weight = __builtin_popcount(gamma);
    if ((in == MASK_VAL) && ((weight % 2) == 0))
        return (pow(2, -15));
    else if ((in != MASK_VAL) && ((gamma & ~varibits) == 0) &&
             (((gamma ^ rol(gamma, ALPHA_MINUS_BETA)) & doublebits) == 0)) {
        weight = __builtin_popcount(varibits ^ doublebits);
        return (pow(2, -weight));
    }

    return (0.0);
}

void calc_ddt_update(vector<double> &ddt, vector<double> &tmp) {
    uint32_t small = 1L << 16;
    vector<double> sums(1L << WORD_SIZE);

    for (uint16_t i = 1; i != 0; i++)
        //DP if we consider the left side only
        sums[i & MASK_VAL] += ddt[i];

    #pragma omp parallel for
    for (uint32_t i = 1; i < small; i++) {
        uint16_t out = i;
        //Calculate left side (of the input difference) based on output difference
        uint16_t in0 = out >> WORD_SIZE;
        uint16_t out0 = out & MASK_VAL;
        double p = 0;
        //Check if this input difference is even possible (based on left side)
        if (sums[in0] != 0)
            for (uint16_t in1 = 0; in1 <= MASK_VAL; in1++) {
                p += ddt[in0 ^ (in1 << WORD_SIZE)] * diff_prob(in0, out0 ^ in1);
            }
        tmp[out] = p;
    }

    ddt.swap(tmp);
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
    //We only need to consider input differences up to rotational equivalence
    uint16_t l_diffs[] = {0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 37, 39, 43, 45, 47, 51, 53, 55,
                          59, 61, 63, 85, 87, 91, 95, 111, 119, 127, 255};
    uint16_t max_iter = 0xff;

    //l_diff = 0, so r_diff != 0
    for (uint16_t r_diff = 0x01; r_diff <= max_iter; r_diff++)
        calc_ddt_for_in_diff(r_diff, num_rounds);

    //l_diff != 0
    for (int i = 1; i < 36; i++) {
        cout << "At " << ((float) i / 36) << "%" << endl;
        for (uint16_t r_diff = 0x00; r_diff <= max_iter; r_diff++)
            calc_ddt_for_in_diff((((uint16_t) l_diffs[i]) << 8 ^ r_diff), num_rounds);
    }
}

int main() {
    calc_ddt(7);
    cout << "Done." << endl;
    return (0);
}

