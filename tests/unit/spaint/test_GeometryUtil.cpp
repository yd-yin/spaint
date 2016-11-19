#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <boost/mpl/list.hpp>

#include <ITMLib/Utils/ITMMath.h>
using namespace ORUtils;

#include <spaint/geometry/GeometryUtil.h>
using namespace spaint;

//#################### TESTS ####################

typedef boost::mpl::list<double,float> TS;

BOOST_AUTO_TEST_SUITE(test_GeometryUtil)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_dual_quat_to_pose, T, TS)
{
  DualQuaternion<T> dq = DualQuaternion<T>::from_rotation(Vector3<T>(0,0,1), T(M_PI_2));

  SE3Pose pose = GeometryUtil::dual_quat_to_pose(dq);
  Vector3f t, r;
  pose.GetParams(t, r);

  BOOST_CHECK_SMALL(length(t), 1e-4f);
  BOOST_CHECK_SMALL(length(r - Vector3f(0,0,(float)M_PI_2)), 1e-4f);
  BOOST_CHECK(DualQuaternion<T>::close(GeometryUtil::pose_to_dual_quat<T>(pose), dq));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_pose_to_dual_quat, T, TS)
{
  Vector3<T> r(0,T(M_PI_4),0);
  Vector3<T> t(3,4,5);
  SE3Pose pose((float)t.x, (float)t.y, (float)t.z, (float)r.x, (float)r.y, (float)r.z);

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

BOOST_AUTO_TEST_SUITE_END()
