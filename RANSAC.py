import numpy as np
import random


def ground_remove(data,
                  distance_threshold=0.3,
                  p=0.99,
                  sample_size=3,
                  max_iterations=10000,
                  ):
    random.seed(12345)
    max_point_num = -1
    i = 0
    k = 10
    lang_data = len(data)
    range_data = range(lang_data)

    while i < k:

        s3 = random.sample(range_data, sample_size)

        if abs(data[s3[0], 1] - data[s3[1], 1]) < 3:
            continue

        # calculate coefficients of plane
        coeffs = estimate_plane(data[s3, :], normalize=False)
        if coeffs is None:
            continue

        # The norm of the normal vector
        r = np.sqrt(coeffs[0] ** 2 + coeffs[1] ** 2 + coeffs[2] ** 2)
        # Calculate the distance between each point and the plane
        d = np.divide(np.abs(np.matmul(coeffs[:3], data.T) + coeffs[3]), r)
        d_filt = np.array(d < distance_threshold)

        near_point_num = np.sum(d_filt, axis=0)

        if near_point_num > max_point_num:
            max_point_num = near_point_num

            best_model = coeffs
            best_filt = d_filt

            w = near_point_num / lang_data

            wn = np.power(w, 3)
            p_no_outliers = 1.0 - wn

            k = (np.log(1 - p) / np.log(p_no_outliers))

        i += 1

        if i > max_iterations:
            print(' RANSAC reached the maximum iterations.')
            break

    return np.argwhere(best_filt).flatten(), best_model


def estimate_plane(xyz, normalize=True):
    """
    :param xyz:  3*3 array
    x1 y1 z1
    x2 y2 z2
    x3 y3 z3
    :return: a b c d

      model_coefficients.resize (4);
      model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
      model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
      model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
      model_coefficients[3] = 0;
      // Normalize
      model_coefficients.normalize ();
      // ... + d = 0
      model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot (p0.matrix ()));

    """
    vector1 = xyz[1, :] - xyz[0, :]
    vector2 = xyz[2, :] - xyz[0, :]

    # check if three points on the same line
    if not np.all(vector1):
        print('will divide by zero..')
        return None
    dy1dy2 = vector2 / vector1

    if not ((dy1dy2[0] != dy1dy2[1]) or (dy1dy2[2] != dy1dy2[1])):
        return None

    a = (vector1[1] * vector2[2]) - (vector1[2] * vector2[1])
    b = (vector1[2] * vector2[0]) - (vector1[0] * vector2[2])
    c = (vector1[0] * vector2[1]) - (vector1[1] * vector2[0])
    # normalize
    if normalize:
        r = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        a = a / r
        b = b / r
        c = c / r
    d = -(a * xyz[0, 0] + b * xyz[0, 1] + c * xyz[0, 2])
    # return a,b,c,d
    return np.array([a, b, c, d])
