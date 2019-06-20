"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from . import backend


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1

''' GIoU
'''
def giou():
    """ Create a GIoU loss functor.

    Args

    Returns
        A functor for computing the GIoU loss given target data and predicted data.
    """

    def _giou(y_true, y_pred):
        """ Compute the GIoU loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). groundtruth of bbox(N,4)(x1,y1,x2,y2)
                                                 (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
                    The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4). predict of bbox(N,4)(x1,y1,x2,y2)

        Returns
            The GIoU loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        bbox_p      = y_pred
        bbox_g      = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices         = backend.where(keras.backend.equal(anchor_state, 1))
        bbox_p          = backend.gather_nd(bbox_p, indices)
        bbox_g          = backend.gather_nd(bbox_g, indices)
        
        # compute giou loss
        x1p = keras.backend.minimum(bbox_p[:, 0], bbox_p[:, 2])
        y1p = keras.backend.minimum(bbox_p[:, 1], bbox_p[:, 3])
        x2p = keras.backend.maximum(bbox_p[:, 0], bbox_p[:, 2])
        y2p = keras.backend.maximum(bbox_p[:, 1], bbox_p[:, 3])
        
        x1g = keras.backend.minimum(bbox_g[:, 0], bbox_g[:, 2])
        y1g = keras.backend.minimum(bbox_g[:, 1], bbox_g[:, 3])
        x2g = keras.backend.maximum(bbox_g[:, 0], bbox_g[:, 2])
        y2g = keras.backend.maximum(bbox_g[:, 1], bbox_g[:, 3])
        
        # calc area of Bg
        area_p = keras.backend.abs((x2p - x1p) * (y2p - y1p))
        # calc area of Bp
        area_g = keras.backend.abs((x2g - x1g) * (y2g - y1g))


        # cal intersection
        x1I = keras.backend.maximum(x1p, x1g)
        y1I = keras.backend.maximum(y1p, y1g)
        x2I = keras.backend.minimum(x2p, x2g)
        y2I = keras.backend.minimum(y2p, y2g)
        I = keras.backend.maximum((y2I - y1I), 0) * keras.backend.maximum((x2I - x1I), 0)

        # find enclosing box
        x1C = keras.backend.minimum(x1p, x1g)
        y1C = keras.backend.minimum(y1p, y1g)
        x2C = keras.backend.maximum(x2p, x2g)
        y2C = keras.backend.maximum(y2p, y2g)
        area_c = (x2C - x1C) * (y2C - y1C)

        # calc area of Bc 
        U = area_p + area_g - I
        iou = 1.0 * I / U

        # Giou
        giou = iou - 1.0 * (area_c - U) / area_c

        # loss_iou = 1 - iou, loss_giou = 1 - giou
        loss_giou = 1.0 - giou

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(loss_giou) / normalizer

    return _giou