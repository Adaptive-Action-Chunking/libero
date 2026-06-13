import numpy as np
import math
import cv2

def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def quat_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions (q1 * q0).

    E.g.:
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )
    
def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def compose_rotations(delta_rs):
    q_total = np.array([0.0, 0.0, 0.0, 1.0])
    for r in delta_rs:
        q = axisangle2quat(r)
        q_total = quat_multiply(q, q_total)
    return quat2axisangle(q_total)

def merge_cam_view(obs, cam_names, trial_id, sim_step, lang, chunk_to_execute=None):
    merge_view_list = []
    for _, show_cam in enumerate(cam_names):
        image_array = obs[show_cam + "_image"][::-1, ::-1, :]
        bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        merge_view_list.append(bgr_img)
        pass
    merge_view = np.concatenate(merge_view_list, axis=1)
    if chunk_to_execute:
        cv2.putText(merge_view, f"trial {trial_id}, sim_t {sim_step}, chunk:{chunk_to_execute}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    else:
        cv2.putText(merge_view, f"trial {trial_id}, sim_t {sim_step}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    cv2.putText(merge_view, f"Instruction: {lang}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return merge_view


def merge_cam_view_ac(obs, cam_names, trial_id, sim_step, lang, Status):
    merge_view_list = []
    for _, show_cam in enumerate(cam_names):
        image_array = obs[show_cam + "_image"][::-1, ::-1, :]
        bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        merge_view_list.append(bgr_img)
        pass
    merge_view = np.concatenate(merge_view_list, axis=1)

    cv2.putText(merge_view, f"trial {trial_id}, sim_t {sim_step}, {Status}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    cv2.putText(merge_view, f"Instruction: {lang}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return merge_view

def convert_action_dict_to_list(pred_action_dict):
    action_dict = {}
    action_dict["right"] = np.concatenate( (pred_action_dict["action.end_effector_position"],pred_action_dict["action.end_effector_rotation"]), axis = 1) # T, D
    action_dict["right_gripper"] = (pred_action_dict["action.gripper_close"]-0.5)*2# convert from [0, 1] to [-1, 1]
    action_array = np.concatenate([action_dict["right"], action_dict["right_gripper"][..., None]], axis=1) # T,D
    return action_array.tolist()

def cosine_similarity(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim