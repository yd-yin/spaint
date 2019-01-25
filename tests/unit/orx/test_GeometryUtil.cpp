#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/mpl/list.hpp>

#include <ORUtils/Math.h>
using namespace ORUtils;

#include <orx/geometry/GeometryUtil.h>
using namespace orx;

//#################### HELPER FUNCTIONS ####################

template <typename T>
void check_close(T a, T b, T TOL)
{
  BOOST_CHECK_CLOSE(a, b, TOL);
}

template <typename T>
void check_close(const T *v1, const T *v2, size_t size, T TOL)
{
  for(size_t i = 0; i < size; ++i) check_close(v1[i], v2[i], TOL);
}

template <typename T>
void check_close(const Vector3<T>& v1, const Vector3<T>& v2, T TOL)
{
  check_close(v1.v, v2.v, v1.size(), TOL);
}

template <typename T>
void check_close(const Vector4<T>& v1, const Vector4<T>& v2, T TOL)
{
  check_close(v1.v, v2.v, v1.size(), TOL);
}

template <typename T>
void check_close(const Matrix3<T>& m1, const Matrix3<T>& m2, T TOL)
{
  check_close(m1.m, m2.m, 9, TOL);
}

//#################### TESTS ####################

typedef boost::mpl::list<double,float> TS;

BOOST_AUTO_TEST_SUITE(test_GeometryUtil)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_blend_poses, T, TS)
{
  // Generate five poses around the identity pose by jittering the rotation angle and translation.
  std::vector<SE3Pose> inputPoses;
  const Vector3<T> up(0,0,1);

  for(float i = -2.0f; i <= 2.0f; ++i)
  {
    inputPoses.push_back(GeometryUtil::dual_quat_to_pose(
      DualQuaternion<T>::from_translation(Vector3<T>(i,0,0)) *
      DualQuaternion<T>::from_rotation(up, T(i * M_PI / 180))
    ));
  }

  // Check that the result of blending the poses is the identity pose.
  SE3Pose outputPose = GeometryUtil::blend_poses(inputPoses);
  BOOST_CHECK(DualQuaternion<T>::close(GeometryUtil::pose_to_dual_quat<T>(outputPose), DualQuaternion<T>::identity()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_dual_quat_to_pose, T, TS)
{
  DualQuaternion<T> dq = DualQuaternion<T>::from_rotation(Vector3<T>(0,0,1), T(M_PI_2));

  SE3Pose pose = GeometryUtil::dual_quat_to_pose(dq);
  Vector3f t = pose.GetT();
  Vector3f r = GeometryUtil::to_rotation_vector<float>(pose.GetR());

  BOOST_CHECK_SMALL(length(t), 1e-4f);
  BOOST_CHECK_SMALL(length(r - Vector3f(0,0,(float)M_PI_2)), 1e-4f);
  BOOST_CHECK(DualQuaternion<T>::close(GeometryUtil::pose_to_dual_quat<T>(pose), dq));
}

BOOST_AUTO_TEST_CASE(test_estimate_rigid_transform)
{
  Eigen::Matrix3f P;
  P(0,0) = 1; P(0,1) = 0; P(0,2) = 0;
  P(1,0) = 0; P(1,1) = 1; P(1,2) = 0;
  P(2,0) = 0; P(2,1) = 0; P(2,2) = 1;

  Eigen::Matrix3f Q;
  Q(0,0) = 1; Q(0,1) = 1; Q(0,2) = 0;
  Q(1,0) = 0; Q(1,1) = 1; Q(1,2) = 0;
  Q(2,0) = 1; Q(2,1) = 0; Q(2,2) = 0;

  Eigen::Matrix4f M = GeometryUtil::estimate_rigid_transform(P, Q);

  for(int i = 0; i < 3; ++i)
  {
    BOOST_CHECK_SMALL((M * P.col(i).homogeneous() - Q.col(i).homogeneous()).norm(), 1e-4f);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_find_best_hypothesis, T, TS)
{
  // Generate increasingly-large clusters of rotated poses around the z axis at 0, PI/2, PI and 3*PI/2 radians.
  const double rotThreshold = 20 * M_PI / 180;
  const float transThreshold = 0.05f;
  const Vector3<T> up(0,0,1);

  std::map<std::string,SE3Pose> inputPoses;
  size_t id = 0;
  for(float i = 0.0f; i < 4.0f; ++i)
  {
    for(float j = -i; j <= i; ++j)
    {
      float angle = static_cast<float>(i * M_PI_2 + j * M_PI / 180);
      inputPoses.insert(std::make_pair(
        boost::lexical_cast<std::string>(id++),
        GeometryUtil::dual_quat_to_pose(DualQuaternion<T>::from_rotation(up, angle))
      ));
    }
  }

  // Find the best hypothesis from these poses, and check that it is one of the poses around 3*PI/2.
  std::vector<SE3Pose> inliersForBestHypothesis;
  int bestHypothesisID = boost::lexical_cast<int>(GeometryUtil::find_best_hypothesis(inputPoses, inliersForBestHypothesis, rotThreshold, transThreshold));
  BOOST_CHECK_GT(bestHypothesisID, 1 + 3 + 5);

  // Check that blending the inliers for the best hypothesis together gives the 3*PI/2 pose.
  SE3Pose refinedPose = GeometryUtil::blend_poses(inliersForBestHypothesis);
  BOOST_CHECK(DualQuaternion<T>::close(GeometryUtil::pose_to_dual_quat<T>(refinedPose), DualQuaternion<T>::from_rotation(up, T(3 * M_PI_2))));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_pose_to_dual_quat, T, TS)
{
  Vector3<T> r(0,T(M_PI_4),0);
  Vector3<T> t(3,4,5);
  SE3Pose pose;
  pose.SetT(t.toFloat());
  pose.SetR(GeometryUtil::to_rotation_matrix<T,float>(r));

  DualQuaternion<T> dq = GeometryUtil::pose_to_dual_quat<T>(pose);

  BOOST_CHECK_SMALL(length(dq.get_rotation() - r), T(1e-4f));
  BOOST_CHECK_SMALL(length(dq.get_translation() - t), T(1e-4f));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_poses_are_similar, T, TS)
{
  const double rotThreshold = 20 * M_PI / 180;
  const float transThreshold = 0.05f;

  DualQuaternion<T> r1 = DualQuaternion<T>::from_rotation(Vector3<T>(0,0,1), 0.0f);
  DualQuaternion<T> r2 = DualQuaternion<T>::from_rotation(Vector3<T>(0,0,1), 19 * (float)M_PI / 180);
  DualQuaternion<T> r3 = DualQuaternion<T>::from_rotation(Vector3<T>(0,0,1), 21 * (float)M_PI / 180);

  BOOST_CHECK(GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(r1), GeometryUtil::dual_quat_to_pose(r2), rotThreshold, transThreshold));
  BOOST_CHECK(!GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(r1), GeometryUtil::dual_quat_to_pose(r3), rotThreshold, transThreshold));
  BOOST_CHECK(GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(r2), GeometryUtil::dual_quat_to_pose(r3), rotThreshold, transThreshold));

  DualQuaternion<T> t1 = DualQuaternion<T>::from_translation(Vector3<T>(0,0,0));
  DualQuaternion<T> t2 = DualQuaternion<T>::from_translation(Vector3<T>(0.04f,0,0));
  DualQuaternion<T> t3 = DualQuaternion<T>::from_translation(Vector3<T>(0.06f,0,0));

  BOOST_CHECK(GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(t1), GeometryUtil::dual_quat_to_pose(t2), rotThreshold, transThreshold));
  BOOST_CHECK(!GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(t1), GeometryUtil::dual_quat_to_pose(t3), rotThreshold, transThreshold));
  BOOST_CHECK(GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(t2), GeometryUtil::dual_quat_to_pose(t3), rotThreshold, transThreshold));

  DualQuaternion<T> t1r1 = t1 * r1, t1r3 = t1 * r3, t2r2 = t2 * r2, t3r1 = t3 * r1, t3r3 = t3 * r3;

  BOOST_CHECK(GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(t1r1), GeometryUtil::dual_quat_to_pose(t2r2), rotThreshold, transThreshold));
  BOOST_CHECK(!GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(t1r1), GeometryUtil::dual_quat_to_pose(t1r3), rotThreshold, transThreshold));
  BOOST_CHECK(!GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(t1r1), GeometryUtil::dual_quat_to_pose(t3r1), rotThreshold, transThreshold));
  BOOST_CHECK(!GeometryUtil::poses_are_similar(GeometryUtil::dual_quat_to_pose(t1r1), GeometryUtil::dual_quat_to_pose(t3r3), rotThreshold, transThreshold));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_to_rotation_matrix, T, TS)
{
  const T TOL = static_cast<T>(1e-4);

  Vector3<T> rotationVector(static_cast<T>(M_PI) / 4.0f, 0.0f, 0.0f);
  const T root2inv = static_cast<T>(1.0 / sqrt(2.0));
  Matrix3<T> rotationMatrix(T(1), T(0), T(0), T(0), root2inv, root2inv, T(0), -root2inv, root2inv);
  check_close(GeometryUtil::to_rotation_matrix(rotationVector), rotationMatrix, TOL);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_to_rotation_vector, T, TS)
{
  const T TOL = static_cast<T>(1e-4);

  const T root2inv = static_cast<T>(1.0 / sqrt(2.0));
  Matrix3<T> rotationMatrix(T(1), T(0), T(0), T(0), root2inv, root2inv, T(0), -root2inv, root2inv);
  Vector3<T> rotationVector(static_cast<T>(M_PI) / 4.0f, 0.0f, 0.0f);
  check_close(GeometryUtil::to_rotation_vector(rotationMatrix), rotationVector, TOL);
}

BOOST_AUTO_TEST_SUITE_END()
