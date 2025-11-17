import torch
import face_detection
from .model.cropping_model import build_crop_model
from .utils import (
    resize_image_op, color_normalization_op,
    enlarge_bbox, generate_bboxes, transpose_image_op
)


class AutoCropper(object):
    def __init__(self,
                 model: str = 'mobilenetv2',
                 cuda: bool = True,
                 model_path: str = "",
                 use_face_detector: bool = True):
        self.cuda = True if cuda and torch.cuda.is_available() else False
        self.cropper = build_crop_model(model=model, cuda=self.cuda, model_path=model_path)
        self.face_detector = None
        if use_face_detector:
            self.face_detector = face_detection.build_detector(
                "DSFDDetector", confidence_threshold=0.5, nms_iou_threshold=0.3
            )
        if self.cuda:
            self.cropper = torch.nn.DataParallel(self.cropper).cuda()

    def detect_face(self, rgb_img):
        return [enlarge_bbox(bbox) for bbox in self.face_detector.detect(rgb_img)]

    def eval_rois(self, image, roi):
        image = transpose_image_op(image)
        image = torch.unsqueeze(torch.as_tensor(image), 0).float()

        if self.cuda:
            image = image.cuda()

        roi = torch.tensor(roi, dtype=torch.float32, device=image.device)

        out = self.cropper(image, roi)
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)
        return id_out


    def eval_rois_batch(self, images, rois_batch):
        tensor_list = [torch.as_tensor(transpose_image_op(img), dtype=torch.float32) for img in images]
        batch_tensor = torch.stack(tensor_list, dim=0)  # (B, C, H, W)

        if self.cuda:
            batch_tensor = batch_tensor.cuda()

        all_rois = []
        for b_idx, rois in enumerate(rois_batch):
            for roi in rois:
                all_rois.append([b_idx, *roi])
        all_rois = torch.tensor(all_rois, dtype=torch.float32, device=batch_tensor.device)

        with torch.no_grad():
            out = self.cropper(batch_tensor, all_rois)

        results = [[] for _ in range(len(images))]
        offset = 0
        for b_idx, rois in enumerate(rois_batch):
            scores = out[offset: offset + len(rois)]
            id_out = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            results[b_idx] = id_out
            offset += len(rois)

        return results


    def generate_anchor_bboxes(self,
                               image,
                               scale_width,
                               scale_height,
                               crop_width,
                               crop_height,
                               face_bboxes,
                               single_face_center=True):
        return generate_bboxes(
            image,
            scale_width=scale_width,
            scale_height=scale_height,
            crop_width=crop_width,
            crop_height=crop_height,
            face_bboxes=face_bboxes,
            single_face_center=single_face_center,
        )

    def crop(self,
             rgb_image,
             topK=1,
             crop_height=None,
             crop_width=None,
             filter_face=True,
             single_face_center=True):
        input_img, scale_height, scale_width = resize_image_op(rgb_image)
        face_bboxes = []
        if self.face_detector and filter_face:
            face_bboxes = self.detect_face(input_img)

        trans_bboxes, source_bboxes = self.generate_anchor_bboxes(
            input_img,
            scale_width=scale_width,
            scale_height=scale_height,
            crop_height=crop_height,
            crop_width=crop_width,
            face_bboxes=face_bboxes,
            single_face_center=single_face_center,
        )
        if not trans_bboxes:
            raise ValueError("No suitable candidate box")

        roi = [[0, *tbbox] for tbbox in trans_bboxes]
        input_img = color_normalization_op(input_img)

        id_out = self.eval_rois(input_img, roi)
        select_bboxes = [[int(round(x)) for x in source_bboxes[i]] for i in id_out[:topK]]
        return select_bboxes

    def crop_batch(self,
                   rgb_images,
                   topK=1,
                   crop_height=None,
                   crop_width=None,
                   filter_face=True,
                   single_face_center=True):
        """
        批量裁剪接口
        :param rgb_images: List[np.ndarray], 多张 HWC RGB 图
        :return: List[List[list[4]]], 每张图的裁剪结果
        """
        all_input_imgs, all_rois, all_source_bboxes = [], [], []

        for rgb_image in rgb_images:
            input_img, scale_height, scale_width = resize_image_op(rgb_image)
            face_bboxes = []
            if self.face_detector and filter_face:
                face_bboxes = self.detect_face(input_img)

            trans_bboxes, source_bboxes = self.generate_anchor_bboxes(
                input_img,
                scale_width=scale_width,
                scale_height=scale_height,
                crop_height=crop_height,
                crop_width=crop_width,
                face_bboxes=face_bboxes,
                single_face_center=single_face_center,
            )
            if not trans_bboxes:
                raise ValueError("No suitable candidate box for one image")

            rois = [[0, *tbbox] for tbbox in trans_bboxes]
            all_input_imgs.append(color_normalization_op(input_img))
            all_rois.append(rois)
            all_source_bboxes.append(source_bboxes)

        id_outs = self.eval_rois_batch(all_input_imgs, all_rois)

        results = []
        for ids, source_bboxes in zip(id_outs, all_source_bboxes):
            select_bboxes = [[int(round(x)) for x in source_bboxes[i]] for i in ids[:topK]]
            results.append(select_bboxes)

        return results
