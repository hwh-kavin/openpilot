#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

double mass;

void set_mass(double x){ mass = x;}

double rotational_inertia;

void set_rotational_inertia(double x){ rotational_inertia = x;}

double center_to_front;

void set_center_to_front(double x){ center_to_front = x;}

double center_to_rear;

void set_center_to_rear(double x){ center_to_rear = x;}

double stiffness_front;

void set_stiffness_front(double x){ stiffness_front = x;}

double stiffness_rear;

void set_stiffness_rear(double x){ stiffness_rear = x;}
const static double MAHA_THRESH_25 = 3.8414588206941227;
const static double MAHA_THRESH_24 = 5.991464547107981;
const static double MAHA_THRESH_30 = 3.8414588206941227;
const static double MAHA_THRESH_26 = 3.8414588206941227;
const static double MAHA_THRESH_27 = 3.8414588206941227;
const static double MAHA_THRESH_29 = 3.8414588206941227;
const static double MAHA_THRESH_28 = 3.8414588206941227;
const static double MAHA_THRESH_31 = 3.8414588206941227;

/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_8253353176002921707) {
   out_8253353176002921707[0] = delta_x[0] + nom_x[0];
   out_8253353176002921707[1] = delta_x[1] + nom_x[1];
   out_8253353176002921707[2] = delta_x[2] + nom_x[2];
   out_8253353176002921707[3] = delta_x[3] + nom_x[3];
   out_8253353176002921707[4] = delta_x[4] + nom_x[4];
   out_8253353176002921707[5] = delta_x[5] + nom_x[5];
   out_8253353176002921707[6] = delta_x[6] + nom_x[6];
   out_8253353176002921707[7] = delta_x[7] + nom_x[7];
   out_8253353176002921707[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2074278415337607634) {
   out_2074278415337607634[0] = -nom_x[0] + true_x[0];
   out_2074278415337607634[1] = -nom_x[1] + true_x[1];
   out_2074278415337607634[2] = -nom_x[2] + true_x[2];
   out_2074278415337607634[3] = -nom_x[3] + true_x[3];
   out_2074278415337607634[4] = -nom_x[4] + true_x[4];
   out_2074278415337607634[5] = -nom_x[5] + true_x[5];
   out_2074278415337607634[6] = -nom_x[6] + true_x[6];
   out_2074278415337607634[7] = -nom_x[7] + true_x[7];
   out_2074278415337607634[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_2338615864478646488) {
   out_2338615864478646488[0] = 1.0;
   out_2338615864478646488[1] = 0;
   out_2338615864478646488[2] = 0;
   out_2338615864478646488[3] = 0;
   out_2338615864478646488[4] = 0;
   out_2338615864478646488[5] = 0;
   out_2338615864478646488[6] = 0;
   out_2338615864478646488[7] = 0;
   out_2338615864478646488[8] = 0;
   out_2338615864478646488[9] = 0;
   out_2338615864478646488[10] = 1.0;
   out_2338615864478646488[11] = 0;
   out_2338615864478646488[12] = 0;
   out_2338615864478646488[13] = 0;
   out_2338615864478646488[14] = 0;
   out_2338615864478646488[15] = 0;
   out_2338615864478646488[16] = 0;
   out_2338615864478646488[17] = 0;
   out_2338615864478646488[18] = 0;
   out_2338615864478646488[19] = 0;
   out_2338615864478646488[20] = 1.0;
   out_2338615864478646488[21] = 0;
   out_2338615864478646488[22] = 0;
   out_2338615864478646488[23] = 0;
   out_2338615864478646488[24] = 0;
   out_2338615864478646488[25] = 0;
   out_2338615864478646488[26] = 0;
   out_2338615864478646488[27] = 0;
   out_2338615864478646488[28] = 0;
   out_2338615864478646488[29] = 0;
   out_2338615864478646488[30] = 1.0;
   out_2338615864478646488[31] = 0;
   out_2338615864478646488[32] = 0;
   out_2338615864478646488[33] = 0;
   out_2338615864478646488[34] = 0;
   out_2338615864478646488[35] = 0;
   out_2338615864478646488[36] = 0;
   out_2338615864478646488[37] = 0;
   out_2338615864478646488[38] = 0;
   out_2338615864478646488[39] = 0;
   out_2338615864478646488[40] = 1.0;
   out_2338615864478646488[41] = 0;
   out_2338615864478646488[42] = 0;
   out_2338615864478646488[43] = 0;
   out_2338615864478646488[44] = 0;
   out_2338615864478646488[45] = 0;
   out_2338615864478646488[46] = 0;
   out_2338615864478646488[47] = 0;
   out_2338615864478646488[48] = 0;
   out_2338615864478646488[49] = 0;
   out_2338615864478646488[50] = 1.0;
   out_2338615864478646488[51] = 0;
   out_2338615864478646488[52] = 0;
   out_2338615864478646488[53] = 0;
   out_2338615864478646488[54] = 0;
   out_2338615864478646488[55] = 0;
   out_2338615864478646488[56] = 0;
   out_2338615864478646488[57] = 0;
   out_2338615864478646488[58] = 0;
   out_2338615864478646488[59] = 0;
   out_2338615864478646488[60] = 1.0;
   out_2338615864478646488[61] = 0;
   out_2338615864478646488[62] = 0;
   out_2338615864478646488[63] = 0;
   out_2338615864478646488[64] = 0;
   out_2338615864478646488[65] = 0;
   out_2338615864478646488[66] = 0;
   out_2338615864478646488[67] = 0;
   out_2338615864478646488[68] = 0;
   out_2338615864478646488[69] = 0;
   out_2338615864478646488[70] = 1.0;
   out_2338615864478646488[71] = 0;
   out_2338615864478646488[72] = 0;
   out_2338615864478646488[73] = 0;
   out_2338615864478646488[74] = 0;
   out_2338615864478646488[75] = 0;
   out_2338615864478646488[76] = 0;
   out_2338615864478646488[77] = 0;
   out_2338615864478646488[78] = 0;
   out_2338615864478646488[79] = 0;
   out_2338615864478646488[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_4794471467657306731) {
   out_4794471467657306731[0] = state[0];
   out_4794471467657306731[1] = state[1];
   out_4794471467657306731[2] = state[2];
   out_4794471467657306731[3] = state[3];
   out_4794471467657306731[4] = state[4];
   out_4794471467657306731[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_4794471467657306731[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_4794471467657306731[7] = state[7];
   out_4794471467657306731[8] = state[8];
}
void F_fun(double *state, double dt, double *out_4310723095704452947) {
   out_4310723095704452947[0] = 1;
   out_4310723095704452947[1] = 0;
   out_4310723095704452947[2] = 0;
   out_4310723095704452947[3] = 0;
   out_4310723095704452947[4] = 0;
   out_4310723095704452947[5] = 0;
   out_4310723095704452947[6] = 0;
   out_4310723095704452947[7] = 0;
   out_4310723095704452947[8] = 0;
   out_4310723095704452947[9] = 0;
   out_4310723095704452947[10] = 1;
   out_4310723095704452947[11] = 0;
   out_4310723095704452947[12] = 0;
   out_4310723095704452947[13] = 0;
   out_4310723095704452947[14] = 0;
   out_4310723095704452947[15] = 0;
   out_4310723095704452947[16] = 0;
   out_4310723095704452947[17] = 0;
   out_4310723095704452947[18] = 0;
   out_4310723095704452947[19] = 0;
   out_4310723095704452947[20] = 1;
   out_4310723095704452947[21] = 0;
   out_4310723095704452947[22] = 0;
   out_4310723095704452947[23] = 0;
   out_4310723095704452947[24] = 0;
   out_4310723095704452947[25] = 0;
   out_4310723095704452947[26] = 0;
   out_4310723095704452947[27] = 0;
   out_4310723095704452947[28] = 0;
   out_4310723095704452947[29] = 0;
   out_4310723095704452947[30] = 1;
   out_4310723095704452947[31] = 0;
   out_4310723095704452947[32] = 0;
   out_4310723095704452947[33] = 0;
   out_4310723095704452947[34] = 0;
   out_4310723095704452947[35] = 0;
   out_4310723095704452947[36] = 0;
   out_4310723095704452947[37] = 0;
   out_4310723095704452947[38] = 0;
   out_4310723095704452947[39] = 0;
   out_4310723095704452947[40] = 1;
   out_4310723095704452947[41] = 0;
   out_4310723095704452947[42] = 0;
   out_4310723095704452947[43] = 0;
   out_4310723095704452947[44] = 0;
   out_4310723095704452947[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_4310723095704452947[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_4310723095704452947[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4310723095704452947[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_4310723095704452947[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_4310723095704452947[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_4310723095704452947[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_4310723095704452947[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_4310723095704452947[53] = -9.8000000000000007*dt;
   out_4310723095704452947[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4310723095704452947[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4310723095704452947[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4310723095704452947[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4310723095704452947[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4310723095704452947[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4310723095704452947[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4310723095704452947[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4310723095704452947[62] = 0;
   out_4310723095704452947[63] = 0;
   out_4310723095704452947[64] = 0;
   out_4310723095704452947[65] = 0;
   out_4310723095704452947[66] = 0;
   out_4310723095704452947[67] = 0;
   out_4310723095704452947[68] = 0;
   out_4310723095704452947[69] = 0;
   out_4310723095704452947[70] = 1;
   out_4310723095704452947[71] = 0;
   out_4310723095704452947[72] = 0;
   out_4310723095704452947[73] = 0;
   out_4310723095704452947[74] = 0;
   out_4310723095704452947[75] = 0;
   out_4310723095704452947[76] = 0;
   out_4310723095704452947[77] = 0;
   out_4310723095704452947[78] = 0;
   out_4310723095704452947[79] = 0;
   out_4310723095704452947[80] = 1;
}
void h_25(double *state, double *unused, double *out_1108542113409831851) {
   out_1108542113409831851[0] = state[6];
}
void H_25(double *state, double *unused, double *out_6476779361774528901) {
   out_6476779361774528901[0] = 0;
   out_6476779361774528901[1] = 0;
   out_6476779361774528901[2] = 0;
   out_6476779361774528901[3] = 0;
   out_6476779361774528901[4] = 0;
   out_6476779361774528901[5] = 0;
   out_6476779361774528901[6] = 1;
   out_6476779361774528901[7] = 0;
   out_6476779361774528901[8] = 0;
}
void h_24(double *state, double *unused, double *out_6540167292037783551) {
   out_6540167292037783551[0] = state[4];
   out_6540167292037783551[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8702487145753397463) {
   out_8702487145753397463[0] = 0;
   out_8702487145753397463[1] = 0;
   out_8702487145753397463[2] = 0;
   out_8702487145753397463[3] = 0;
   out_8702487145753397463[4] = 1;
   out_8702487145753397463[5] = 0;
   out_8702487145753397463[6] = 0;
   out_8702487145753397463[7] = 0;
   out_8702487145753397463[8] = 0;
   out_8702487145753397463[9] = 0;
   out_8702487145753397463[10] = 0;
   out_8702487145753397463[11] = 0;
   out_8702487145753397463[12] = 0;
   out_8702487145753397463[13] = 0;
   out_8702487145753397463[14] = 1;
   out_8702487145753397463[15] = 0;
   out_8702487145753397463[16] = 0;
   out_8702487145753397463[17] = 0;
}
void h_30(double *state, double *unused, double *out_342089279219051316) {
   out_342089279219051316[0] = state[4];
}
void H_30(double *state, double *unused, double *out_5053274370443405960) {
   out_5053274370443405960[0] = 0;
   out_5053274370443405960[1] = 0;
   out_5053274370443405960[2] = 0;
   out_5053274370443405960[3] = 0;
   out_5053274370443405960[4] = 1;
   out_5053274370443405960[5] = 0;
   out_5053274370443405960[6] = 0;
   out_5053274370443405960[7] = 0;
   out_5053274370443405960[8] = 0;
}
void h_26(double *state, double *unused, double *out_2470462824050802067) {
   out_2470462824050802067[0] = state[7];
}
void H_26(double *state, double *unused, double *out_2735276042900472677) {
   out_2735276042900472677[0] = 0;
   out_2735276042900472677[1] = 0;
   out_2735276042900472677[2] = 0;
   out_2735276042900472677[3] = 0;
   out_2735276042900472677[4] = 0;
   out_2735276042900472677[5] = 0;
   out_2735276042900472677[6] = 0;
   out_2735276042900472677[7] = 1;
   out_2735276042900472677[8] = 0;
}
void h_27(double *state, double *unused, double *out_5894220590383233928) {
   out_5894220590383233928[0] = state[3];
}
void H_27(double *state, double *unused, double *out_4172677102830863920) {
   out_4172677102830863920[0] = 0;
   out_4172677102830863920[1] = 0;
   out_4172677102830863920[2] = 0;
   out_4172677102830863920[3] = 1;
   out_4172677102830863920[4] = 0;
   out_4172677102830863920[5] = 0;
   out_4172677102830863920[6] = 0;
   out_4172677102830863920[7] = 0;
   out_4172677102830863920[8] = 0;
}
void h_29(double *state, double *unused, double *out_4659583988441794329) {
   out_4659583988441794329[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4543043026129013776) {
   out_4543043026129013776[0] = 0;
   out_4543043026129013776[1] = 1;
   out_4543043026129013776[2] = 0;
   out_4543043026129013776[3] = 0;
   out_4543043026129013776[4] = 0;
   out_4543043026129013776[5] = 0;
   out_4543043026129013776[6] = 0;
   out_4543043026129013776[7] = 0;
   out_4543043026129013776[8] = 0;
}
void h_28(double *state, double *unused, double *out_3570554311319476923) {
   out_3570554311319476923[0] = state[0];
}
void H_28(double *state, double *unused, double *out_4422944647526639138) {
   out_4422944647526639138[0] = 1;
   out_4422944647526639138[1] = 0;
   out_4422944647526639138[2] = 0;
   out_4422944647526639138[3] = 0;
   out_4422944647526639138[4] = 0;
   out_4422944647526639138[5] = 0;
   out_4422944647526639138[6] = 0;
   out_4422944647526639138[7] = 0;
   out_4422944647526639138[8] = 0;
}
void h_31(double *state, double *unused, double *out_7275656153380966154) {
   out_7275656153380966154[0] = state[8];
}
void H_31(double *state, double *unused, double *out_6507425323651489329) {
   out_6507425323651489329[0] = 0;
   out_6507425323651489329[1] = 0;
   out_6507425323651489329[2] = 0;
   out_6507425323651489329[3] = 0;
   out_6507425323651489329[4] = 0;
   out_6507425323651489329[5] = 0;
   out_6507425323651489329[6] = 0;
   out_6507425323651489329[7] = 0;
   out_6507425323651489329[8] = 1;
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

void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
}
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<2, 3, 0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
}
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
}
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
}
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
}
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
}
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_31, H_31, NULL, in_z, in_R, in_ea, MAHA_THRESH_31);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_8253353176002921707) {
  err_fun(nom_x, delta_x, out_8253353176002921707);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2074278415337607634) {
  inv_err_fun(nom_x, true_x, out_2074278415337607634);
}
void car_H_mod_fun(double *state, double *out_2338615864478646488) {
  H_mod_fun(state, out_2338615864478646488);
}
void car_f_fun(double *state, double dt, double *out_4794471467657306731) {
  f_fun(state,  dt, out_4794471467657306731);
}
void car_F_fun(double *state, double dt, double *out_4310723095704452947) {
  F_fun(state,  dt, out_4310723095704452947);
}
void car_h_25(double *state, double *unused, double *out_1108542113409831851) {
  h_25(state, unused, out_1108542113409831851);
}
void car_H_25(double *state, double *unused, double *out_6476779361774528901) {
  H_25(state, unused, out_6476779361774528901);
}
void car_h_24(double *state, double *unused, double *out_6540167292037783551) {
  h_24(state, unused, out_6540167292037783551);
}
void car_H_24(double *state, double *unused, double *out_8702487145753397463) {
  H_24(state, unused, out_8702487145753397463);
}
void car_h_30(double *state, double *unused, double *out_342089279219051316) {
  h_30(state, unused, out_342089279219051316);
}
void car_H_30(double *state, double *unused, double *out_5053274370443405960) {
  H_30(state, unused, out_5053274370443405960);
}
void car_h_26(double *state, double *unused, double *out_2470462824050802067) {
  h_26(state, unused, out_2470462824050802067);
}
void car_H_26(double *state, double *unused, double *out_2735276042900472677) {
  H_26(state, unused, out_2735276042900472677);
}
void car_h_27(double *state, double *unused, double *out_5894220590383233928) {
  h_27(state, unused, out_5894220590383233928);
}
void car_H_27(double *state, double *unused, double *out_4172677102830863920) {
  H_27(state, unused, out_4172677102830863920);
}
void car_h_29(double *state, double *unused, double *out_4659583988441794329) {
  h_29(state, unused, out_4659583988441794329);
}
void car_H_29(double *state, double *unused, double *out_4543043026129013776) {
  H_29(state, unused, out_4543043026129013776);
}
void car_h_28(double *state, double *unused, double *out_3570554311319476923) {
  h_28(state, unused, out_3570554311319476923);
}
void car_H_28(double *state, double *unused, double *out_4422944647526639138) {
  H_28(state, unused, out_4422944647526639138);
}
void car_h_31(double *state, double *unused, double *out_7275656153380966154) {
  h_31(state, unused, out_7275656153380966154);
}
void car_H_31(double *state, double *unused, double *out_6507425323651489329) {
  H_31(state, unused, out_6507425323651489329);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
void car_set_mass(double x) {
  set_mass(x);
}
void car_set_rotational_inertia(double x) {
  set_rotational_inertia(x);
}
void car_set_center_to_front(double x) {
  set_center_to_front(x);
}
void car_set_center_to_rear(double x) {
  set_center_to_rear(x);
}
void car_set_stiffness_front(double x) {
  set_stiffness_front(x);
}
void car_set_stiffness_rear(double x) {
  set_stiffness_rear(x);
}
}

const EKF car = {
  .name = "car",
  .kinds = { 25, 24, 30, 26, 27, 29, 28, 31 },
  .feature_kinds = {  },
  .f_fun = car_f_fun,
  .F_fun = car_F_fun,
  .err_fun = car_err_fun,
  .inv_err_fun = car_inv_err_fun,
  .H_mod_fun = car_H_mod_fun,
  .predict = car_predict,
  .hs = {
    { 25, car_h_25 },
    { 24, car_h_24 },
    { 30, car_h_30 },
    { 26, car_h_26 },
    { 27, car_h_27 },
    { 29, car_h_29 },
    { 28, car_h_28 },
    { 31, car_h_31 },
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
    { 31, car_H_31 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
    { 31, car_update_31 },
  },
  .Hes = {
  },
  .sets = {
    { "mass", car_set_mass },
    { "rotational_inertia", car_set_rotational_inertia },
    { "center_to_front", car_set_center_to_front },
    { "center_to_rear", car_set_center_to_rear },
    { "stiffness_front", car_set_stiffness_front },
    { "stiffness_rear", car_set_stiffness_rear },
  },
  .extra_routines = {
  },
};

ekf_lib_init(car)
