import yaml
import casadi as ca

def calc_quat_cost(q_des, q_cur, weight):
  diff = multiply_quaternions(q_des, conjugate_quaternion(q_cur))
  cost = ca.transpose(diff) @ ca.diag(weight) @ diff
  return cost

def calc_vec_cost(des, curr, weight):
  diff = des - curr
  cost = ca.transpose(diff) @ ca.diag(weight) @ diff
  return cost

def normalize_quaternion(q):
  magnitude = ca.norm_2(q)
  return q / magnitude

def conjugate_quaternion(q):
  return ca.vertcat(q[0], -q[1], -q[2], -q[3])

def multiply_quaternions(q1, q2):
  qw = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
  qx = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
  qy = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
  qz = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
  return ca.vertcat(qw, qx, qy, qz)

def rotate_vector_by_quaternion(vec, q):
  p = ca.vertcat(0, vec)
  q_conjugate = conjugate_quaternion(q)
  rotated_vec = multiply_quaternions(multiply_quaternions(q, p), q_conjugate)
  return rotated_vec[1:]

def yaml_to_dict(path_to_yaml):
  with open(path_to_yaml, 'r') as stream:
    try:
      parsed_yaml = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
      print(exc)
  if '/**' in parsed_yaml:
    parsed_yaml = parsed_yaml['/**']['ros__parameters']
  return parsed_yaml
