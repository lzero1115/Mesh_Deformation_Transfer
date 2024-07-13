//
// Created by lex on 7/12/24.
//

#ifndef DEFORM_TRANSFER_H
#define DEFORM_TRANSFER_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <igl/triangle_triangle_adjacency.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycenter.h>

#include <iostream>
#include <vector>

class DeformXfer
{
    unsigned int n_verts_, n_faces_, n_dim_;
    std::vector<Eigen::Matrix<double,3,3>> T_inv_;
    Eigen::SparseMatrix<double> Ed_At_;  // displacement terms
    Eigen::SparseMatrix<double> Ed_AtA_;
    Eigen::SparseMatrix<double> Ed_AtC_;
    Eigen::SparseMatrix<double> Es_AtA_; // smoothness, Equation (11)
    Eigen::SparseMatrix<double> Es_AtC_;
    Eigen::SparseMatrix<double> Ei_AtA_; // deformation identity, Equation (12)
    Eigen::SparseMatrix<double> Ei_AtC_;
    Eigen::SparseMatrix<double> AtA_;
    Eigen::SparseMatrix<double> AtC_;
    Eigen::SparseMatrix<double> At_;

    void compute_Es_(const Eigen::MatrixXi&);
    void compute_Ei_(const Eigen::MatrixXi&);
    void compute_Ed_(
      const Eigen::MatrixXi&,
      const std::vector<std::pair<int, int> >&
    );

    bool fit_into_(
    const Eigen::MatrixXd&,
    const Eigen::MatrixXi&,
    const Eigen::MatrixXd&,
    const Eigen::MatrixXi&,
    const std::vector<std::pair<int, int> >&,
    std::vector<Eigen::MatrixXd>&
    );

    Eigen::SparseLU<Eigen::SparseMatrix<double> > solver_;

public:

    bool keep_progress = false;
    double max_distance = 100;
    double threshold = 10;
    double Wd = 1;
    double Ws = 1;
    double Wi = 0.001;
    std::vector<double> Wc = {1, 50, 1000, 5000};

    DeformXfer(
    const Eigen::MatrixXd& = Eigen::MatrixXd(),
    const Eigen::MatrixXi& = Eigen::MatrixXi()
    );

    bool initialize(
    const Eigen::MatrixXd&,
    const Eigen::MatrixXi&
    );

    bool map_correspondences(
    const Eigen::MatrixXd& SV,
    const Eigen::MatrixXi& SF,
    const Eigen::MatrixXd& TV,
    const Eigen::MatrixXi& TF,
    const std::vector<std::pair<int, int> >& markers,
    std::vector<Eigen::MatrixXd>& X, // progress of source morphing into target
    std::vector<std::pair<int, int> >& M // correspondences, Equation (5)
    );

    bool precompute(const Eigen::VectorXi& CI);

    bool transfer(
    const Eigen::MatrixXd&,
    const Eigen::MatrixXi&,
    const Eigen::MatrixXd&,
    const Eigen::MatrixXd&,
    const std::vector<std::pair<int, int> >&,
    Eigen::MatrixXd&
    );

};

void compute_inverse_transforms(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  std::vector<Eigen::Matrix<double, 3, 3> >& T
);

void compute_deform_rotations(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& U,
  std::vector<Eigen::Vector<double, 9> >& Q
);

inline void add_frobenius_row_(
  std::vector<Eigen::Triplet<double> >& coeffs,
  unsigned int row_off,
  unsigned int col_off,
  const Eigen::ArrayXi& verts,
  const Eigen::Matrix<double, 3, 3>& inv_xf,
  double mult_coeff = 1.0
);

bool compute_AtA(
  const Eigen::VectorXi& CI,
  const int n_dim,
  Eigen::SparseMatrix<double>& At,
  Eigen::SparseMatrix<double>& AtA
);

bool compute_AtC(
  const Eigen::MatrixXd& CC,
  const Eigen::SparseMatrix<double>& At,
  Eigen::SparseMatrix<double>& AtC
);

void find_correspondences(
  const Eigen::MatrixXd& SV,
  const Eigen::MatrixXi& SF,
  const Eigen::MatrixXd& TV,
  const Eigen::MatrixXi& TF,
  const double threshold,
  std::vector<std::pair<int, int> >& M
);

void closest_valid_points(
  const Eigen::MatrixXd& VX,
  const Eigen::MatrixXd& NX,
  const Eigen::MatrixXd& VY,
  const Eigen::MatrixXd& NY,
  const double max_distance,
  std::map<int, Eigen::Vector3d>& C
);


#endif //DEFORM_TRANSFER_H
