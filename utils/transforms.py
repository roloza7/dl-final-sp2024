from torchvision.transforms import functional as VF

class ResizeTransform:
    def __init__(self, new_width, new_height, instances=False): # instances if using bounding boxes set
        self.new_width = new_width
        self.new_height = new_height
        self.instances = instances

    def __call__(self, image, target):
        original_width, original_height = image.size
        scale_w = self.new_width / original_width
        scale_h = self.new_height / original_height

        resized_image = VF.resize(image, (self.new_height, self.new_width))

        if self.instances:
            for box in target:
                bbox = box["bbox"]
                new_x_min = bbox[0] * scale_w
                new_y_min = bbox[1] * scale_h
                new_x_max = bbox[2] * scale_w
                new_y_max = bbox[3] * scale_h
                box["bbox"] = [new_x_min, new_y_min, new_x_max, new_y_max]

        return resized_image, target