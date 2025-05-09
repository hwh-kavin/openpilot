#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                      Code generated with SymPy 1.13.2                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_382365343426424134) {
   out_382365343426424134[0] = delta_x[0] + nom_x[0];
   out_382365343426424134[1] = delta_x[1] + nom_x[1];
   out_382365343426424134[2] = delta_x[2] + nom_x[2];
   out_382365343426424134[3] = delta_x[3] + nom_x[3];
   out_382365343426424134[4] = delta_x[4] + nom_x[4];
   out_382365343426424134[5] = delta_x[5] + nom_x[5];
   out_382365343426424134[6] = delta_x[6] + nom_x[6];
   out_382365343426424134[7] = delta_x[7] + nom_x[7];
   out_382365343426424134[8] = delta_x[8] + nom_x[8];
   out_382365343426424134[9] = delta_x[9] + nom_x[9];
   out_382365343426424134[10] = delta_x[10] + nom_x[10];
   out_382365343426424134[11] = delta_x[11] + nom_x[11];
   out_382365343426424134[12] = delta_x[12] + nom_x[12];
   out_382365343426424134[13] = delta_x[13] + nom_x[13];
   out_382365343426424134[14] = delta_x[14] + nom_x[14];
   out_382365343426424134[15] = delta_x[15] + nom_x[15];
   out_382365343426424134[16] = delta_x[16] + nom_x[16];
   out_382365343426424134[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3442262595017034864) {
   out_3442262595017034864[0] = -nom_x[0] + true_x[0];
   out_3442262595017034864[1] = -nom_x[1] + true_x[1];
   out_3442262595017034864[2] = -nom_x[2] + true_x[2];
   out_3442262595017034864[3] = -nom_x[3] + true_x[3];
   out_3442262595017034864[4] = -nom_x[4] + true_x[4];
   out_3442262595017034864[5] = -nom_x[5] + true_x[5];
   out_3442262595017034864[6] = -nom_x[6] + true_x[6];
   out_3442262595017034864[7] = -nom_x[7] + true_x[7];
   out_3442262595017034864[8] = -nom_x[8] + true_x[8];
   out_3442262595017034864[9] = -nom_x[9] + true_x[9];
   out_3442262595017034864[10] = -nom_x[10] + true_x[10];
   out_3442262595017034864[11] = -nom_x[11] + true_x[11];
   out_3442262595017034864[12] = -nom_x[12] + true_x[12];
   out_3442262595017034864[13] = -nom_x[13] + true_x[13];
   out_3442262595017034864[14] = -nom_x[14] + true_x[14];
   out_3442262595017034864[15] = -nom_x[15] + true_x[15];
   out_3442262595017034864[16] = -nom_x[16] + true_x[16];
   out_3442262595017034864[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_6670007426242957292) {
   out_6670007426242957292[0] = 1.0;
   out_6670007426242957292[1] = 0.0;
   out_6670007426242957292[2] = 0.0;
   out_6670007426242957292[3] = 0.0;
   out_6670007426242957292[4] = 0.0;
   out_6670007426242957292[5] = 0.0;
   out_6670007426242957292[6] = 0.0;
   out_6670007426242957292[7] = 0.0;
   out_6670007426242957292[8] = 0.0;
   out_6670007426242957292[9] = 0.0;
   out_6670007426242957292[10] = 0.0;
   out_6670007426242957292[11] = 0.0;
   out_6670007426242957292[12] = 0.0;
   out_6670007426242957292[13] = 0.0;
   out_6670007426242957292[14] = 0.0;
   out_6670007426242957292[15] = 0.0;
   out_6670007426242957292[16] = 0.0;
   out_6670007426242957292[17] = 0.0;
   out_6670007426242957292[18] = 0.0;
   out_6670007426242957292[19] = 1.0;
   out_6670007426242957292[20] = 0.0;
   out_6670007426242957292[21] = 0.0;
   out_6670007426242957292[22] = 0.0;
   out_6670007426242957292[23] = 0.0;
   out_6670007426242957292[24] = 0.0;
   out_6670007426242957292[25] = 0.0;
   out_6670007426242957292[26] = 0.0;
   out_6670007426242957292[27] = 0.0;
   out_6670007426242957292[28] = 0.0;
   out_6670007426242957292[29] = 0.0;
   out_6670007426242957292[30] = 0.0;
   out_6670007426242957292[31] = 0.0;
   out_6670007426242957292[32] = 0.0;
   out_6670007426242957292[33] = 0.0;
   out_6670007426242957292[34] = 0.0;
   out_6670007426242957292[35] = 0.0;
   out_6670007426242957292[36] = 0.0;
   out_6670007426242957292[37] = 0.0;
   out_6670007426242957292[38] = 1.0;
   out_6670007426242957292[39] = 0.0;
   out_6670007426242957292[40] = 0.0;
   out_6670007426242957292[41] = 0.0;
   out_6670007426242957292[42] = 0.0;
   out_6670007426242957292[43] = 0.0;
   out_6670007426242957292[44] = 0.0;
   out_6670007426242957292[45] = 0.0;
   out_6670007426242957292[46] = 0.0;
   out_6670007426242957292[47] = 0.0;
   out_6670007426242957292[48] = 0.0;
   out_6670007426242957292[49] = 0.0;
   out_6670007426242957292[50] = 0.0;
   out_6670007426242957292[51] = 0.0;
   out_6670007426242957292[52] = 0.0;
   out_6670007426242957292[53] = 0.0;
   out_6670007426242957292[54] = 0.0;
   out_6670007426242957292[55] = 0.0;
   out_6670007426242957292[56] = 0.0;
   out_6670007426242957292[57] = 1.0;
   out_6670007426242957292[58] = 0.0;
   out_6670007426242957292[59] = 0.0;
   out_6670007426242957292[60] = 0.0;
   out_6670007426242957292[61] = 0.0;
   out_6670007426242957292[62] = 0.0;
   out_6670007426242957292[63] = 0.0;
   out_6670007426242957292[64] = 0.0;
   out_6670007426242957292[65] = 0.0;
   out_6670007426242957292[66] = 0.0;
   out_6670007426242957292[67] = 0.0;
   out_6670007426242957292[68] = 0.0;
   out_6670007426242957292[69] = 0.0;
   out_6670007426242957292[70] = 0.0;
   out_6670007426242957292[71] = 0.0;
   out_6670007426242957292[72] = 0.0;
   out_6670007426242957292[73] = 0.0;
   out_6670007426242957292[74] = 0.0;
   out_6670007426242957292[75] = 0.0;
   out_6670007426242957292[76] = 1.0;
   out_6670007426242957292[77] = 0.0;
   out_6670007426242957292[78] = 0.0;
   out_6670007426242957292[79] = 0.0;
   out_6670007426242957292[80] = 0.0;
   out_6670007426242957292[81] = 0.0;
   out_6670007426242957292[82] = 0.0;
   out_6670007426242957292[83] = 0.0;
   out_6670007426242957292[84] = 0.0;
   out_6670007426242957292[85] = 0.0;
   out_6670007426242957292[86] = 0.0;
   out_6670007426242957292[87] = 0.0;
   out_6670007426242957292[88] = 0.0;
   out_6670007426242957292[89] = 0.0;
   out_6670007426242957292[90] = 0.0;
   out_6670007426242957292[91] = 0.0;
   out_6670007426242957292[92] = 0.0;
   out_6670007426242957292[93] = 0.0;
   out_6670007426242957292[94] = 0.0;
   out_6670007426242957292[95] = 1.0;
   out_6670007426242957292[96] = 0.0;
   out_6670007426242957292[97] = 0.0;
   out_6670007426242957292[98] = 0.0;
   out_6670007426242957292[99] = 0.0;
   out_6670007426242957292[100] = 0.0;
   out_6670007426242957292[101] = 0.0;
   out_6670007426242957292[102] = 0.0;
   out_6670007426242957292[103] = 0.0;
   out_6670007426242957292[104] = 0.0;
   out_6670007426242957292[105] = 0.0;
   out_6670007426242957292[106] = 0.0;
   out_6670007426242957292[107] = 0.0;
   out_6670007426242957292[108] = 0.0;
   out_6670007426242957292[109] = 0.0;
   out_6670007426242957292[110] = 0.0;
   out_6670007426242957292[111] = 0.0;
   out_6670007426242957292[112] = 0.0;
   out_6670007426242957292[113] = 0.0;
   out_6670007426242957292[114] = 1.0;
   out_6670007426242957292[115] = 0.0;
   out_6670007426242957292[116] = 0.0;
   out_6670007426242957292[117] = 0.0;
   out_6670007426242957292[118] = 0.0;
   out_6670007426242957292[119] = 0.0;
   out_6670007426242957292[120] = 0.0;
   out_6670007426242957292[121] = 0.0;
   out_6670007426242957292[122] = 0.0;
   out_6670007426242957292[123] = 0.0;
   out_6670007426242957292[124] = 0.0;
   out_6670007426242957292[125] = 0.0;
   out_6670007426242957292[126] = 0.0;
   out_6670007426242957292[127] = 0.0;
   out_6670007426242957292[128] = 0.0;
   out_6670007426242957292[129] = 0.0;
   out_6670007426242957292[130] = 0.0;
   out_6670007426242957292[131] = 0.0;
   out_6670007426242957292[132] = 0.0;
   out_6670007426242957292[133] = 1.0;
   out_6670007426242957292[134] = 0.0;
   out_6670007426242957292[135] = 0.0;
   out_6670007426242957292[136] = 0.0;
   out_6670007426242957292[137] = 0.0;
   out_6670007426242957292[138] = 0.0;
   out_6670007426242957292[139] = 0.0;
   out_6670007426242957292[140] = 0.0;
   out_6670007426242957292[141] = 0.0;
   out_6670007426242957292[142] = 0.0;
   out_6670007426242957292[143] = 0.0;
   out_6670007426242957292[144] = 0.0;
   out_6670007426242957292[145] = 0.0;
   out_6670007426242957292[146] = 0.0;
   out_6670007426242957292[147] = 0.0;
   out_6670007426242957292[148] = 0.0;
   out_6670007426242957292[149] = 0.0;
   out_6670007426242957292[150] = 0.0;
   out_6670007426242957292[151] = 0.0;
   out_6670007426242957292[152] = 1.0;
   out_6670007426242957292[153] = 0.0;
   out_6670007426242957292[154] = 0.0;
   out_6670007426242957292[155] = 0.0;
   out_6670007426242957292[156] = 0.0;
   out_6670007426242957292[157] = 0.0;
   out_6670007426242957292[158] = 0.0;
   out_6670007426242957292[159] = 0.0;
   out_6670007426242957292[160] = 0.0;
   out_6670007426242957292[161] = 0.0;
   out_6670007426242957292[162] = 0.0;
   out_6670007426242957292[163] = 0.0;
   out_6670007426242957292[164] = 0.0;
   out_6670007426242957292[165] = 0.0;
   out_6670007426242957292[166] = 0.0;
   out_6670007426242957292[167] = 0.0;
   out_6670007426242957292[168] = 0.0;
   out_6670007426242957292[169] = 0.0;
   out_6670007426242957292[170] = 0.0;
   out_6670007426242957292[171] = 1.0;
   out_6670007426242957292[172] = 0.0;
   out_6670007426242957292[173] = 0.0;
   out_6670007426242957292[174] = 0.0;
   out_6670007426242957292[175] = 0.0;
   out_6670007426242957292[176] = 0.0;
   out_6670007426242957292[177] = 0.0;
   out_6670007426242957292[178] = 0.0;
   out_6670007426242957292[179] = 0.0;
   out_6670007426242957292[180] = 0.0;
   out_6670007426242957292[181] = 0.0;
   out_6670007426242957292[182] = 0.0;
   out_6670007426242957292[183] = 0.0;
   out_6670007426242957292[184] = 0.0;
   out_6670007426242957292[185] = 0.0;
   out_6670007426242957292[186] = 0.0;
   out_6670007426242957292[187] = 0.0;
   out_6670007426242957292[188] = 0.0;
   out_6670007426242957292[189] = 0.0;
   out_6670007426242957292[190] = 1.0;
   out_6670007426242957292[191] = 0.0;
   out_6670007426242957292[192] = 0.0;
   out_6670007426242957292[193] = 0.0;
   out_6670007426242957292[194] = 0.0;
   out_6670007426242957292[195] = 0.0;
   out_6670007426242957292[196] = 0.0;
   out_6670007426242957292[197] = 0.0;
   out_6670007426242957292[198] = 0.0;
   out_6670007426242957292[199] = 0.0;
   out_6670007426242957292[200] = 0.0;
   out_6670007426242957292[201] = 0.0;
   out_6670007426242957292[202] = 0.0;
   out_6670007426242957292[203] = 0.0;
   out_6670007426242957292[204] = 0.0;
   out_6670007426242957292[205] = 0.0;
   out_6670007426242957292[206] = 0.0;
   out_6670007426242957292[207] = 0.0;
   out_6670007426242957292[208] = 0.0;
   out_6670007426242957292[209] = 1.0;
   out_6670007426242957292[210] = 0.0;
   out_6670007426242957292[211] = 0.0;
   out_6670007426242957292[212] = 0.0;
   out_6670007426242957292[213] = 0.0;
   out_6670007426242957292[214] = 0.0;
   out_6670007426242957292[215] = 0.0;
   out_6670007426242957292[216] = 0.0;
   out_6670007426242957292[217] = 0.0;
   out_6670007426242957292[218] = 0.0;
   out_6670007426242957292[219] = 0.0;
   out_6670007426242957292[220] = 0.0;
   out_6670007426242957292[221] = 0.0;
   out_6670007426242957292[222] = 0.0;
   out_6670007426242957292[223] = 0.0;
   out_6670007426242957292[224] = 0.0;
   out_6670007426242957292[225] = 0.0;
   out_6670007426242957292[226] = 0.0;
   out_6670007426242957292[227] = 0.0;
   out_6670007426242957292[228] = 1.0;
   out_6670007426242957292[229] = 0.0;
   out_6670007426242957292[230] = 0.0;
   out_6670007426242957292[231] = 0.0;
   out_6670007426242957292[232] = 0.0;
   out_6670007426242957292[233] = 0.0;
   out_6670007426242957292[234] = 0.0;
   out_6670007426242957292[235] = 0.0;
   out_6670007426242957292[236] = 0.0;
   out_6670007426242957292[237] = 0.0;
   out_6670007426242957292[238] = 0.0;
   out_6670007426242957292[239] = 0.0;
   out_6670007426242957292[240] = 0.0;
   out_6670007426242957292[241] = 0.0;
   out_6670007426242957292[242] = 0.0;
   out_6670007426242957292[243] = 0.0;
   out_6670007426242957292[244] = 0.0;
   out_6670007426242957292[245] = 0.0;
   out_6670007426242957292[246] = 0.0;
   out_6670007426242957292[247] = 1.0;
   out_6670007426242957292[248] = 0.0;
   out_6670007426242957292[249] = 0.0;
   out_6670007426242957292[250] = 0.0;
   out_6670007426242957292[251] = 0.0;
   out_6670007426242957292[252] = 0.0;
   out_6670007426242957292[253] = 0.0;
   out_6670007426242957292[254] = 0.0;
   out_6670007426242957292[255] = 0.0;
   out_6670007426242957292[256] = 0.0;
   out_6670007426242957292[257] = 0.0;
   out_6670007426242957292[258] = 0.0;
   out_6670007426242957292[259] = 0.0;
   out_6670007426242957292[260] = 0.0;
   out_6670007426242957292[261] = 0.0;
   out_6670007426242957292[262] = 0.0;
   out_6670007426242957292[263] = 0.0;
   out_6670007426242957292[264] = 0.0;
   out_6670007426242957292[265] = 0.0;
   out_6670007426242957292[266] = 1.0;
   out_6670007426242957292[267] = 0.0;
   out_6670007426242957292[268] = 0.0;
   out_6670007426242957292[269] = 0.0;
   out_6670007426242957292[270] = 0.0;
   out_6670007426242957292[271] = 0.0;
   out_6670007426242957292[272] = 0.0;
   out_6670007426242957292[273] = 0.0;
   out_6670007426242957292[274] = 0.0;
   out_6670007426242957292[275] = 0.0;
   out_6670007426242957292[276] = 0.0;
   out_6670007426242957292[277] = 0.0;
   out_6670007426242957292[278] = 0.0;
   out_6670007426242957292[279] = 0.0;
   out_6670007426242957292[280] = 0.0;
   out_6670007426242957292[281] = 0.0;
   out_6670007426242957292[282] = 0.0;
   out_6670007426242957292[283] = 0.0;
   out_6670007426242957292[284] = 0.0;
   out_6670007426242957292[285] = 1.0;
   out_6670007426242957292[286] = 0.0;
   out_6670007426242957292[287] = 0.0;
   out_6670007426242957292[288] = 0.0;
   out_6670007426242957292[289] = 0.0;
   out_6670007426242957292[290] = 0.0;
   out_6670007426242957292[291] = 0.0;
   out_6670007426242957292[292] = 0.0;
   out_6670007426242957292[293] = 0.0;
   out_6670007426242957292[294] = 0.0;
   out_6670007426242957292[295] = 0.0;
   out_6670007426242957292[296] = 0.0;
   out_6670007426242957292[297] = 0.0;
   out_6670007426242957292[298] = 0.0;
   out_6670007426242957292[299] = 0.0;
   out_6670007426242957292[300] = 0.0;
   out_6670007426242957292[301] = 0.0;
   out_6670007426242957292[302] = 0.0;
   out_6670007426242957292[303] = 0.0;
   out_6670007426242957292[304] = 1.0;
   out_6670007426242957292[305] = 0.0;
   out_6670007426242957292[306] = 0.0;
   out_6670007426242957292[307] = 0.0;
   out_6670007426242957292[308] = 0.0;
   out_6670007426242957292[309] = 0.0;
   out_6670007426242957292[310] = 0.0;
   out_6670007426242957292[311] = 0.0;
   out_6670007426242957292[312] = 0.0;
   out_6670007426242957292[313] = 0.0;
   out_6670007426242957292[314] = 0.0;
   out_6670007426242957292[315] = 0.0;
   out_6670007426242957292[316] = 0.0;
   out_6670007426242957292[317] = 0.0;
   out_6670007426242957292[318] = 0.0;
   out_6670007426242957292[319] = 0.0;
   out_6670007426242957292[320] = 0.0;
   out_6670007426242957292[321] = 0.0;
   out_6670007426242957292[322] = 0.0;
   out_6670007426242957292[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_5677574138907018263) {
   out_5677574138907018263[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_5677574138907018263[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_5677574138907018263[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_5677574138907018263[3] = dt*state[12] + state[3];
   out_5677574138907018263[4] = dt*state[13] + state[4];
   out_5677574138907018263[5] = dt*state[14] + state[5];
   out_5677574138907018263[6] = state[6];
   out_5677574138907018263[7] = state[7];
   out_5677574138907018263[8] = state[8];
   out_5677574138907018263[9] = state[9];
   out_5677574138907018263[10] = state[10];
   out_5677574138907018263[11] = state[11];
   out_5677574138907018263[12] = state[12];
   out_5677574138907018263[13] = state[13];
   out_5677574138907018263[14] = state[14];
   out_5677574138907018263[15] = state[15];
   out_5677574138907018263[16] = state[16];
   out_5677574138907018263[17] = state[17];
}
void F_fun(double *state, double dt, double *out_3505296101455130334) {
   out_3505296101455130334[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3505296101455130334[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3505296101455130334[2] = 0;
   out_3505296101455130334[3] = 0;
   out_3505296101455130334[4] = 0;
   out_3505296101455130334[5] = 0;
   out_3505296101455130334[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3505296101455130334[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3505296101455130334[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3505296101455130334[9] = 0;
   out_3505296101455130334[10] = 0;
   out_3505296101455130334[11] = 0;
   out_3505296101455130334[12] = 0;
   out_3505296101455130334[13] = 0;
   out_3505296101455130334[14] = 0;
   out_3505296101455130334[15] = 0;
   out_3505296101455130334[16] = 0;
   out_3505296101455130334[17] = 0;
   out_3505296101455130334[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3505296101455130334[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3505296101455130334[20] = 0;
   out_3505296101455130334[21] = 0;
   out_3505296101455130334[22] = 0;
   out_3505296101455130334[23] = 0;
   out_3505296101455130334[24] = 0;
   out_3505296101455130334[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3505296101455130334[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3505296101455130334[27] = 0;
   out_3505296101455130334[28] = 0;
   out_3505296101455130334[29] = 0;
   out_3505296101455130334[30] = 0;
   out_3505296101455130334[31] = 0;
   out_3505296101455130334[32] = 0;
   out_3505296101455130334[33] = 0;
   out_3505296101455130334[34] = 0;
   out_3505296101455130334[35] = 0;
   out_3505296101455130334[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3505296101455130334[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3505296101455130334[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3505296101455130334[39] = 0;
   out_3505296101455130334[40] = 0;
   out_3505296101455130334[41] = 0;
   out_3505296101455130334[42] = 0;
   out_3505296101455130334[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3505296101455130334[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3505296101455130334[45] = 0;
   out_3505296101455130334[46] = 0;
   out_3505296101455130334[47] = 0;
   out_3505296101455130334[48] = 0;
   out_3505296101455130334[49] = 0;
   out_3505296101455130334[50] = 0;
   out_3505296101455130334[51] = 0;
   out_3505296101455130334[52] = 0;
   out_3505296101455130334[53] = 0;
   out_3505296101455130334[54] = 0;
   out_3505296101455130334[55] = 0;
   out_3505296101455130334[56] = 0;
   out_3505296101455130334[57] = 1;
   out_3505296101455130334[58] = 0;
   out_3505296101455130334[59] = 0;
   out_3505296101455130334[60] = 0;
   out_3505296101455130334[61] = 0;
   out_3505296101455130334[62] = 0;
   out_3505296101455130334[63] = 0;
   out_3505296101455130334[64] = 0;
   out_3505296101455130334[65] = 0;
   out_3505296101455130334[66] = dt;
   out_3505296101455130334[67] = 0;
   out_3505296101455130334[68] = 0;
   out_3505296101455130334[69] = 0;
   out_3505296101455130334[70] = 0;
   out_3505296101455130334[71] = 0;
   out_3505296101455130334[72] = 0;
   out_3505296101455130334[73] = 0;
   out_3505296101455130334[74] = 0;
   out_3505296101455130334[75] = 0;
   out_3505296101455130334[76] = 1;
   out_3505296101455130334[77] = 0;
   out_3505296101455130334[78] = 0;
   out_3505296101455130334[79] = 0;
   out_3505296101455130334[80] = 0;
   out_3505296101455130334[81] = 0;
   out_3505296101455130334[82] = 0;
   out_3505296101455130334[83] = 0;
   out_3505296101455130334[84] = 0;
   out_3505296101455130334[85] = dt;
   out_3505296101455130334[86] = 0;
   out_3505296101455130334[87] = 0;
   out_3505296101455130334[88] = 0;
   out_3505296101455130334[89] = 0;
   out_3505296101455130334[90] = 0;
   out_3505296101455130334[91] = 0;
   out_3505296101455130334[92] = 0;
   out_3505296101455130334[93] = 0;
   out_3505296101455130334[94] = 0;
   out_3505296101455130334[95] = 1;
   out_3505296101455130334[96] = 0;
   out_3505296101455130334[97] = 0;
   out_3505296101455130334[98] = 0;
   out_3505296101455130334[99] = 0;
   out_3505296101455130334[100] = 0;
   out_3505296101455130334[101] = 0;
   out_3505296101455130334[102] = 0;
   out_3505296101455130334[103] = 0;
   out_3505296101455130334[104] = dt;
   out_3505296101455130334[105] = 0;
   out_3505296101455130334[106] = 0;
   out_3505296101455130334[107] = 0;
   out_3505296101455130334[108] = 0;
   out_3505296101455130334[109] = 0;
   out_3505296101455130334[110] = 0;
   out_3505296101455130334[111] = 0;
   out_3505296101455130334[112] = 0;
   out_3505296101455130334[113] = 0;
   out_3505296101455130334[114] = 1;
   out_3505296101455130334[115] = 0;
   out_3505296101455130334[116] = 0;
   out_3505296101455130334[117] = 0;
   out_3505296101455130334[118] = 0;
   out_3505296101455130334[119] = 0;
   out_3505296101455130334[120] = 0;
   out_3505296101455130334[121] = 0;
   out_3505296101455130334[122] = 0;
   out_3505296101455130334[123] = 0;
   out_3505296101455130334[124] = 0;
   out_3505296101455130334[125] = 0;
   out_3505296101455130334[126] = 0;
   out_3505296101455130334[127] = 0;
   out_3505296101455130334[128] = 0;
   out_3505296101455130334[129] = 0;
   out_3505296101455130334[130] = 0;
   out_3505296101455130334[131] = 0;
   out_3505296101455130334[132] = 0;
   out_3505296101455130334[133] = 1;
   out_3505296101455130334[134] = 0;
   out_3505296101455130334[135] = 0;
   out_3505296101455130334[136] = 0;
   out_3505296101455130334[137] = 0;
   out_3505296101455130334[138] = 0;
   out_3505296101455130334[139] = 0;
   out_3505296101455130334[140] = 0;
   out_3505296101455130334[141] = 0;
   out_3505296101455130334[142] = 0;
   out_3505296101455130334[143] = 0;
   out_3505296101455130334[144] = 0;
   out_3505296101455130334[145] = 0;
   out_3505296101455130334[146] = 0;
   out_3505296101455130334[147] = 0;
   out_3505296101455130334[148] = 0;
   out_3505296101455130334[149] = 0;
   out_3505296101455130334[150] = 0;
   out_3505296101455130334[151] = 0;
   out_3505296101455130334[152] = 1;
   out_3505296101455130334[153] = 0;
   out_3505296101455130334[154] = 0;
   out_3505296101455130334[155] = 0;
   out_3505296101455130334[156] = 0;
   out_3505296101455130334[157] = 0;
   out_3505296101455130334[158] = 0;
   out_3505296101455130334[159] = 0;
   out_3505296101455130334[160] = 0;
   out_3505296101455130334[161] = 0;
   out_3505296101455130334[162] = 0;
   out_3505296101455130334[163] = 0;
   out_3505296101455130334[164] = 0;
   out_3505296101455130334[165] = 0;
   out_3505296101455130334[166] = 0;
   out_3505296101455130334[167] = 0;
   out_3505296101455130334[168] = 0;
   out_3505296101455130334[169] = 0;
   out_3505296101455130334[170] = 0;
   out_3505296101455130334[171] = 1;
   out_3505296101455130334[172] = 0;
   out_3505296101455130334[173] = 0;
   out_3505296101455130334[174] = 0;
   out_3505296101455130334[175] = 0;
   out_3505296101455130334[176] = 0;
   out_3505296101455130334[177] = 0;
   out_3505296101455130334[178] = 0;
   out_3505296101455130334[179] = 0;
   out_3505296101455130334[180] = 0;
   out_3505296101455130334[181] = 0;
   out_3505296101455130334[182] = 0;
   out_3505296101455130334[183] = 0;
   out_3505296101455130334[184] = 0;
   out_3505296101455130334[185] = 0;
   out_3505296101455130334[186] = 0;
   out_3505296101455130334[187] = 0;
   out_3505296101455130334[188] = 0;
   out_3505296101455130334[189] = 0;
   out_3505296101455130334[190] = 1;
   out_3505296101455130334[191] = 0;
   out_3505296101455130334[192] = 0;
   out_3505296101455130334[193] = 0;
   out_3505296101455130334[194] = 0;
   out_3505296101455130334[195] = 0;
   out_3505296101455130334[196] = 0;
   out_3505296101455130334[197] = 0;
   out_3505296101455130334[198] = 0;
   out_3505296101455130334[199] = 0;
   out_3505296101455130334[200] = 0;
   out_3505296101455130334[201] = 0;
   out_3505296101455130334[202] = 0;
   out_3505296101455130334[203] = 0;
   out_3505296101455130334[204] = 0;
   out_3505296101455130334[205] = 0;
   out_3505296101455130334[206] = 0;
   out_3505296101455130334[207] = 0;
   out_3505296101455130334[208] = 0;
   out_3505296101455130334[209] = 1;
   out_3505296101455130334[210] = 0;
   out_3505296101455130334[211] = 0;
   out_3505296101455130334[212] = 0;
   out_3505296101455130334[213] = 0;
   out_3505296101455130334[214] = 0;
   out_3505296101455130334[215] = 0;
   out_3505296101455130334[216] = 0;
   out_3505296101455130334[217] = 0;
   out_3505296101455130334[218] = 0;
   out_3505296101455130334[219] = 0;
   out_3505296101455130334[220] = 0;
   out_3505296101455130334[221] = 0;
   out_3505296101455130334[222] = 0;
   out_3505296101455130334[223] = 0;
   out_3505296101455130334[224] = 0;
   out_3505296101455130334[225] = 0;
   out_3505296101455130334[226] = 0;
   out_3505296101455130334[227] = 0;
   out_3505296101455130334[228] = 1;
   out_3505296101455130334[229] = 0;
   out_3505296101455130334[230] = 0;
   out_3505296101455130334[231] = 0;
   out_3505296101455130334[232] = 0;
   out_3505296101455130334[233] = 0;
   out_3505296101455130334[234] = 0;
   out_3505296101455130334[235] = 0;
   out_3505296101455130334[236] = 0;
   out_3505296101455130334[237] = 0;
   out_3505296101455130334[238] = 0;
   out_3505296101455130334[239] = 0;
   out_3505296101455130334[240] = 0;
   out_3505296101455130334[241] = 0;
   out_3505296101455130334[242] = 0;
   out_3505296101455130334[243] = 0;
   out_3505296101455130334[244] = 0;
   out_3505296101455130334[245] = 0;
   out_3505296101455130334[246] = 0;
   out_3505296101455130334[247] = 1;
   out_3505296101455130334[248] = 0;
   out_3505296101455130334[249] = 0;
   out_3505296101455130334[250] = 0;
   out_3505296101455130334[251] = 0;
   out_3505296101455130334[252] = 0;
   out_3505296101455130334[253] = 0;
   out_3505296101455130334[254] = 0;
   out_3505296101455130334[255] = 0;
   out_3505296101455130334[256] = 0;
   out_3505296101455130334[257] = 0;
   out_3505296101455130334[258] = 0;
   out_3505296101455130334[259] = 0;
   out_3505296101455130334[260] = 0;
   out_3505296101455130334[261] = 0;
   out_3505296101455130334[262] = 0;
   out_3505296101455130334[263] = 0;
   out_3505296101455130334[264] = 0;
   out_3505296101455130334[265] = 0;
   out_3505296101455130334[266] = 1;
   out_3505296101455130334[267] = 0;
   out_3505296101455130334[268] = 0;
   out_3505296101455130334[269] = 0;
   out_3505296101455130334[270] = 0;
   out_3505296101455130334[271] = 0;
   out_3505296101455130334[272] = 0;
   out_3505296101455130334[273] = 0;
   out_3505296101455130334[274] = 0;
   out_3505296101455130334[275] = 0;
   out_3505296101455130334[276] = 0;
   out_3505296101455130334[277] = 0;
   out_3505296101455130334[278] = 0;
   out_3505296101455130334[279] = 0;
   out_3505296101455130334[280] = 0;
   out_3505296101455130334[281] = 0;
   out_3505296101455130334[282] = 0;
   out_3505296101455130334[283] = 0;
   out_3505296101455130334[284] = 0;
   out_3505296101455130334[285] = 1;
   out_3505296101455130334[286] = 0;
   out_3505296101455130334[287] = 0;
   out_3505296101455130334[288] = 0;
   out_3505296101455130334[289] = 0;
   out_3505296101455130334[290] = 0;
   out_3505296101455130334[291] = 0;
   out_3505296101455130334[292] = 0;
   out_3505296101455130334[293] = 0;
   out_3505296101455130334[294] = 0;
   out_3505296101455130334[295] = 0;
   out_3505296101455130334[296] = 0;
   out_3505296101455130334[297] = 0;
   out_3505296101455130334[298] = 0;
   out_3505296101455130334[299] = 0;
   out_3505296101455130334[300] = 0;
   out_3505296101455130334[301] = 0;
   out_3505296101455130334[302] = 0;
   out_3505296101455130334[303] = 0;
   out_3505296101455130334[304] = 1;
   out_3505296101455130334[305] = 0;
   out_3505296101455130334[306] = 0;
   out_3505296101455130334[307] = 0;
   out_3505296101455130334[308] = 0;
   out_3505296101455130334[309] = 0;
   out_3505296101455130334[310] = 0;
   out_3505296101455130334[311] = 0;
   out_3505296101455130334[312] = 0;
   out_3505296101455130334[313] = 0;
   out_3505296101455130334[314] = 0;
   out_3505296101455130334[315] = 0;
   out_3505296101455130334[316] = 0;
   out_3505296101455130334[317] = 0;
   out_3505296101455130334[318] = 0;
   out_3505296101455130334[319] = 0;
   out_3505296101455130334[320] = 0;
   out_3505296101455130334[321] = 0;
   out_3505296101455130334[322] = 0;
   out_3505296101455130334[323] = 1;
}
void h_4(double *state, double *unused, double *out_4447281030058185614) {
   out_4447281030058185614[0] = state[6] + state[9];
   out_4447281030058185614[1] = state[7] + state[10];
   out_4447281030058185614[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_1530297457730267475) {
   out_1530297457730267475[0] = 0;
   out_1530297457730267475[1] = 0;
   out_1530297457730267475[2] = 0;
   out_1530297457730267475[3] = 0;
   out_1530297457730267475[4] = 0;
   out_1530297457730267475[5] = 0;
   out_1530297457730267475[6] = 1;
   out_1530297457730267475[7] = 0;
   out_1530297457730267475[8] = 0;
   out_1530297457730267475[9] = 1;
   out_1530297457730267475[10] = 0;
   out_1530297457730267475[11] = 0;
   out_1530297457730267475[12] = 0;
   out_1530297457730267475[13] = 0;
   out_1530297457730267475[14] = 0;
   out_1530297457730267475[15] = 0;
   out_1530297457730267475[16] = 0;
   out_1530297457730267475[17] = 0;
   out_1530297457730267475[18] = 0;
   out_1530297457730267475[19] = 0;
   out_1530297457730267475[20] = 0;
   out_1530297457730267475[21] = 0;
   out_1530297457730267475[22] = 0;
   out_1530297457730267475[23] = 0;
   out_1530297457730267475[24] = 0;
   out_1530297457730267475[25] = 1;
   out_1530297457730267475[26] = 0;
   out_1530297457730267475[27] = 0;
   out_1530297457730267475[28] = 1;
   out_1530297457730267475[29] = 0;
   out_1530297457730267475[30] = 0;
   out_1530297457730267475[31] = 0;
   out_1530297457730267475[32] = 0;
   out_1530297457730267475[33] = 0;
   out_1530297457730267475[34] = 0;
   out_1530297457730267475[35] = 0;
   out_1530297457730267475[36] = 0;
   out_1530297457730267475[37] = 0;
   out_1530297457730267475[38] = 0;
   out_1530297457730267475[39] = 0;
   out_1530297457730267475[40] = 0;
   out_1530297457730267475[41] = 0;
   out_1530297457730267475[42] = 0;
   out_1530297457730267475[43] = 0;
   out_1530297457730267475[44] = 1;
   out_1530297457730267475[45] = 0;
   out_1530297457730267475[46] = 0;
   out_1530297457730267475[47] = 1;
   out_1530297457730267475[48] = 0;
   out_1530297457730267475[49] = 0;
   out_1530297457730267475[50] = 0;
   out_1530297457730267475[51] = 0;
   out_1530297457730267475[52] = 0;
   out_1530297457730267475[53] = 0;
}
void h_10(double *state, double *unused, double *out_2806881049815821202) {
   out_2806881049815821202[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_2806881049815821202[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_2806881049815821202[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_624257207186408244) {
   out_624257207186408244[0] = 0;
   out_624257207186408244[1] = 9.8100000000000005*cos(state[1]);
   out_624257207186408244[2] = 0;
   out_624257207186408244[3] = 0;
   out_624257207186408244[4] = -state[8];
   out_624257207186408244[5] = state[7];
   out_624257207186408244[6] = 0;
   out_624257207186408244[7] = state[5];
   out_624257207186408244[8] = -state[4];
   out_624257207186408244[9] = 0;
   out_624257207186408244[10] = 0;
   out_624257207186408244[11] = 0;
   out_624257207186408244[12] = 1;
   out_624257207186408244[13] = 0;
   out_624257207186408244[14] = 0;
   out_624257207186408244[15] = 1;
   out_624257207186408244[16] = 0;
   out_624257207186408244[17] = 0;
   out_624257207186408244[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_624257207186408244[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_624257207186408244[20] = 0;
   out_624257207186408244[21] = state[8];
   out_624257207186408244[22] = 0;
   out_624257207186408244[23] = -state[6];
   out_624257207186408244[24] = -state[5];
   out_624257207186408244[25] = 0;
   out_624257207186408244[26] = state[3];
   out_624257207186408244[27] = 0;
   out_624257207186408244[28] = 0;
   out_624257207186408244[29] = 0;
   out_624257207186408244[30] = 0;
   out_624257207186408244[31] = 1;
   out_624257207186408244[32] = 0;
   out_624257207186408244[33] = 0;
   out_624257207186408244[34] = 1;
   out_624257207186408244[35] = 0;
   out_624257207186408244[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_624257207186408244[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_624257207186408244[38] = 0;
   out_624257207186408244[39] = -state[7];
   out_624257207186408244[40] = state[6];
   out_624257207186408244[41] = 0;
   out_624257207186408244[42] = state[4];
   out_624257207186408244[43] = -state[3];
   out_624257207186408244[44] = 0;
   out_624257207186408244[45] = 0;
   out_624257207186408244[46] = 0;
   out_624257207186408244[47] = 0;
   out_624257207186408244[48] = 0;
   out_624257207186408244[49] = 0;
   out_624257207186408244[50] = 1;
   out_624257207186408244[51] = 0;
   out_624257207186408244[52] = 0;
   out_624257207186408244[53] = 1;
}
void h_13(double *state, double *unused, double *out_8409369100465836073) {
   out_8409369100465836073[0] = state[3];
   out_8409369100465836073[1] = state[4];
   out_8409369100465836073[2] = state[5];
}
void H_13(double *state, double *unused, double *out_9140928666046968404) {
   out_9140928666046968404[0] = 0;
   out_9140928666046968404[1] = 0;
   out_9140928666046968404[2] = 0;
   out_9140928666046968404[3] = 1;
   out_9140928666046968404[4] = 0;
   out_9140928666046968404[5] = 0;
   out_9140928666046968404[6] = 0;
   out_9140928666046968404[7] = 0;
   out_9140928666046968404[8] = 0;
   out_9140928666046968404[9] = 0;
   out_9140928666046968404[10] = 0;
   out_9140928666046968404[11] = 0;
   out_9140928666046968404[12] = 0;
   out_9140928666046968404[13] = 0;
   out_9140928666046968404[14] = 0;
   out_9140928666046968404[15] = 0;
   out_9140928666046968404[16] = 0;
   out_9140928666046968404[17] = 0;
   out_9140928666046968404[18] = 0;
   out_9140928666046968404[19] = 0;
   out_9140928666046968404[20] = 0;
   out_9140928666046968404[21] = 0;
   out_9140928666046968404[22] = 1;
   out_9140928666046968404[23] = 0;
   out_9140928666046968404[24] = 0;
   out_9140928666046968404[25] = 0;
   out_9140928666046968404[26] = 0;
   out_9140928666046968404[27] = 0;
   out_9140928666046968404[28] = 0;
   out_9140928666046968404[29] = 0;
   out_9140928666046968404[30] = 0;
   out_9140928666046968404[31] = 0;
   out_9140928666046968404[32] = 0;
   out_9140928666046968404[33] = 0;
   out_9140928666046968404[34] = 0;
   out_9140928666046968404[35] = 0;
   out_9140928666046968404[36] = 0;
   out_9140928666046968404[37] = 0;
   out_9140928666046968404[38] = 0;
   out_9140928666046968404[39] = 0;
   out_9140928666046968404[40] = 0;
   out_9140928666046968404[41] = 1;
   out_9140928666046968404[42] = 0;
   out_9140928666046968404[43] = 0;
   out_9140928666046968404[44] = 0;
   out_9140928666046968404[45] = 0;
   out_9140928666046968404[46] = 0;
   out_9140928666046968404[47] = 0;
   out_9140928666046968404[48] = 0;
   out_9140928666046968404[49] = 0;
   out_9140928666046968404[50] = 0;
   out_9140928666046968404[51] = 0;
   out_9140928666046968404[52] = 0;
   out_9140928666046968404[53] = 0;
}
void h_14(double *state, double *unused, double *out_2303464499919243309) {
   out_2303464499919243309[0] = state[6];
   out_2303464499919243309[1] = state[7];
   out_2303464499919243309[2] = state[8];
}
void H_14(double *state, double *unused, double *out_5493538314069752004) {
   out_5493538314069752004[0] = 0;
   out_5493538314069752004[1] = 0;
   out_5493538314069752004[2] = 0;
   out_5493538314069752004[3] = 0;
   out_5493538314069752004[4] = 0;
   out_5493538314069752004[5] = 0;
   out_5493538314069752004[6] = 1;
   out_5493538314069752004[7] = 0;
   out_5493538314069752004[8] = 0;
   out_5493538314069752004[9] = 0;
   out_5493538314069752004[10] = 0;
   out_5493538314069752004[11] = 0;
   out_5493538314069752004[12] = 0;
   out_5493538314069752004[13] = 0;
   out_5493538314069752004[14] = 0;
   out_5493538314069752004[15] = 0;
   out_5493538314069752004[16] = 0;
   out_5493538314069752004[17] = 0;
   out_5493538314069752004[18] = 0;
   out_5493538314069752004[19] = 0;
   out_5493538314069752004[20] = 0;
   out_5493538314069752004[21] = 0;
   out_5493538314069752004[22] = 0;
   out_5493538314069752004[23] = 0;
   out_5493538314069752004[24] = 0;
   out_5493538314069752004[25] = 1;
   out_5493538314069752004[26] = 0;
   out_5493538314069752004[27] = 0;
   out_5493538314069752004[28] = 0;
   out_5493538314069752004[29] = 0;
   out_5493538314069752004[30] = 0;
   out_5493538314069752004[31] = 0;
   out_5493538314069752004[32] = 0;
   out_5493538314069752004[33] = 0;
   out_5493538314069752004[34] = 0;
   out_5493538314069752004[35] = 0;
   out_5493538314069752004[36] = 0;
   out_5493538314069752004[37] = 0;
   out_5493538314069752004[38] = 0;
   out_5493538314069752004[39] = 0;
   out_5493538314069752004[40] = 0;
   out_5493538314069752004[41] = 0;
   out_5493538314069752004[42] = 0;
   out_5493538314069752004[43] = 0;
   out_5493538314069752004[44] = 1;
   out_5493538314069752004[45] = 0;
   out_5493538314069752004[46] = 0;
   out_5493538314069752004[47] = 0;
   out_5493538314069752004[48] = 0;
   out_5493538314069752004[49] = 0;
   out_5493538314069752004[50] = 0;
   out_5493538314069752004[51] = 0;
   out_5493538314069752004[52] = 0;
   out_5493538314069752004[53] = 0;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_382365343426424134) {
  err_fun(nom_x, delta_x, out_382365343426424134);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3442262595017034864) {
  inv_err_fun(nom_x, true_x, out_3442262595017034864);
}
void pose_H_mod_fun(double *state, double *out_6670007426242957292) {
  H_mod_fun(state, out_6670007426242957292);
}
void pose_f_fun(double *state, double dt, double *out_5677574138907018263) {
  f_fun(state,  dt, out_5677574138907018263);
}
void pose_F_fun(double *state, double dt, double *out_3505296101455130334) {
  F_fun(state,  dt, out_3505296101455130334);
}
void pose_h_4(double *state, double *unused, double *out_4447281030058185614) {
  h_4(state, unused, out_4447281030058185614);
}
void pose_H_4(double *state, double *unused, double *out_1530297457730267475) {
  H_4(state, unused, out_1530297457730267475);
}
void pose_h_10(double *state, double *unused, double *out_2806881049815821202) {
  h_10(state, unused, out_2806881049815821202);
}
void pose_H_10(double *state, double *unused, double *out_624257207186408244) {
  H_10(state, unused, out_624257207186408244);
}
void pose_h_13(double *state, double *unused, double *out_8409369100465836073) {
  h_13(state, unused, out_8409369100465836073);
}
void pose_H_13(double *state, double *unused, double *out_9140928666046968404) {
  H_13(state, unused, out_9140928666046968404);
}
void pose_h_14(double *state, double *unused, double *out_2303464499919243309) {
  h_14(state, unused, out_2303464499919243309);
}
void pose_H_14(double *state, double *unused, double *out_5493538314069752004) {
  H_14(state, unused, out_5493538314069752004);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
